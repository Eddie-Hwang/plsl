import os
import json
import torch
import importlib
from collections import Counter
from tokenizers import SentencePieceBPETokenizer
from tokenizers.normalizers import NFKC, Sequence, Strip
from tokenizers.processors import TemplateProcessing
from modules import external_metrics_sacrebleu
from modules import mscoco_rouge


def instantiate_from_config(config):
    if not 'target' in config:
        raise KeyError('Expected key "target" to instatiate.')
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def sliding_window(tensor, window_size, overlap_size):
    # tensor: [batch_size, n_frames, n_dim]
    # Get step size
    step_size = window_size - overlap_size

    # Apply sliding window
    windows = tensor.unfold(1, window_size, step_size)

    return windows


def create_mask(seq_lengths, device="cpu"):
    max_len = max(seq_lengths)
    mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(seq_lengths, device=device)[:, None]
    return mask.to(torch.bool)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_vocab(file_path, min_freq=1):
    data = get_data(file_path)
    vocab = build_vocab(data, min_freq)
    return vocab


def save_vocab(vocab, file_path):
    with open(file_path, "w") as f:
        json.dump(vocab, f, indent=4)


def load_vocab(file_path):
    with open(file_path, "r") as f:
        vocab = json.load(f)
    return vocab


def get_data(path):
    with open(path, "r") as file:
        return [line.strip() for line in file]
    

def build_vocab(data, min_freq):
    """
    Build a vocabulary from the given data.

    Args:
        data (list[str]): A list of strings, where each string represents a line of data.
        min_freq (int): The minimum frequency for a token to be included in the vocabulary.

    Returns:
        dict: A dictionary mapping tokens to their indices in the vocabulary.
    """
    tokens = [token for line in data for token in line.split()]
    token_counts = Counter(tokens)
    token_list = [token for token, count in token_counts.items() if count >= min_freq]

    # Add special tokens to the vocabulary
    special_tokens = ['<sos>', '<eos>', '<pad>']
    vocab = {token: idx for idx, token in enumerate(special_tokens + token_list)}

    return vocab


def strings_to_indices(strings, vocab):
    """
    Convert a list of strings to a list of lists of indices based on the given vocabulary.

    Args:
        strings (list[str]): The input list of strings to be converted.
        vocab (dict): The vocabulary mapping tokens to indices.

    Returns:
        list[list[int]]: A list of lists of indices representing the input strings.
    """
    indices_list = []
    for string in strings:
        tokens = string.split()
        indices = [vocab["<sos>"]] + [vocab[token] for token in tokens if token in vocab] + [vocab["<eos>"]]
        indices_list.append(indices)
    return indices_list


def indices_to_strings(indices_list, vocab, sos_token='<sos>', eos_token='<eos>', pad_token='<pad>'):
    """
    Convert a list of lists of indices to a list of strings based on the given vocabulary.

    Args:
        indices_list (list[list[int]]): The input list of lists of indices to be converted.
        vocab (dict): The vocabulary mapping tokens to indices.

    Returns:
        list[str]: A list of strings representing the input indices.
    """
    inv_vocab = {idx: token for token, idx in vocab.items()}
    special_indices = [vocab[token] for token in [sos_token, eos_token, pad_token]]
    strings = []
    for indices in indices_list:
        indices = indices.tolist()
        tokens = [inv_vocab[idx] for idx in indices if idx not in special_indices]
        string = ' '.join(tokens)
        strings.append(string)
    return strings


def init_tokenizer(text_corpus, gloss_corpus, min_freq, tokenizer_path):
    tokenizer = SentencePieceBPETokenizer()

    # Set up a normalizer
    # normalizer = Sequence([BertNormalizer(clean_text=True, handle_chinese_chars=False, lowercase=True), NFKC()])
    normalizer = Sequence([Strip(), NFKC()])
    tokenizer._tokenizer.normalizer = normalizer
    
    vocab_filename = os.path.join(tokenizer_path, "vocab.json")
    merges_filename = os.path.join(tokenizer_path, "merges.txt")

    if not(os.path.exists(vocab_filename)):
        if gloss_corpus is not None: # Shared vocab
            _file = [text_corpus, gloss_corpus]
        else:
            _file = [text_corpus]
        tokenizer.train(files=_file, vocab_size=32000, min_frequency=min_freq, special_tokens=["<unk>", "<s>", "<pad>", "</s>"])
        tokenizer.save_model(tokenizer_path)

    # Load tokenizer
    tokenizer = tokenizer.from_file(vocab_filename=vocab_filename, merges_filename=merges_filename)
    tokenizer._tokenizer.add_special_tokens(["<s>", "<pad>", "</s>"])

    # Post processing
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        # pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    
    return tokenizer


def evaluate_results(predictions, references, tokenizer=None, split="train"):
    log_dicts = {}
    
    bleu_scores = external_metrics_sacrebleu.raw_corpus_bleu(
        sys_stream=predictions, ref_streams=[references]
    ).scores
    for n in range(len(bleu_scores)):
        log_dicts[f"{split}/bleu" + str(n + 1)] = bleu_scores[n]

    rouge_score = 0
    n_seq = len(predictions)
    for h, r in zip(predictions, references):
        rouge_score += mscoco_rouge.calc_score(hypotheses=[h], references=[r]) / n_seq

    log_dicts[f"{split}/rouge"] = rouge_score * 100
    
    return log_dicts