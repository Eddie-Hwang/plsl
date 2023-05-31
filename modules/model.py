import pytorch_lightning as pl
import torch
import torch.nn as nn
from modules.utils import *
from modules.positional import PositionalEncoding
from modules.data import get_vocab, strings_to_indices, indices_to_strings
import modules.external_metrics_sacrebleu as external_metrics_sacrebleu
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
import numpy as np
from einops import rearrange


def vocab(file_path, cache, min_freq=1):
    if os.path.exists(cache):
        vocab = load_vocab(file_path=cache)
    else:
        vocab = get_vocab(file_path, min_freq)
        save_vocab(vocab, file_path=cache)

    return vocab


def create_mask(seq_lengths, device="cpu"):
    max_len = max(seq_lengths)
    mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(seq_lengths, device=device)[:, None]
    return mask.bool()



def get_bleus(references, hypotheses, split="train"):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    bleu_scores = external_metrics_sacrebleu.raw_corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references]
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores[f"{split}/bleu" + str(n + 1)] = bleu_scores[n]
    return scores



class TransformerT2G(pl.LightningModule):
    def __init__(
        self, 
        monitor, 
        text_file_path, 
        gloss_file_path,
        text_vocab_cache,
        gloss_vocab_cache,
        dim_feedforward=1024, 
        d_model=512, 
        nhead=4, 
        dropout=0.1, 
        activation="relu", 
        n_layers=3,
        max_len=150,
        vocab_freq=1,
        emb_dim=512,
        base_learning_rate=0.001, 
        label_smoothing=0.1,
        **kwargs
    ):
        super().__init__()

        self.lr = base_learning_rate
        self.monitor = monitor
        self.max_len = max_len
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.label_smoothing = label_smoothing

        self.text_vocab = vocab(text_file_path, text_vocab_cache, vocab_freq)
        self.gloss_vocab = vocab(gloss_file_path, gloss_vocab_cache, vocab_freq)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layers)
        
        self.src_emb = nn.Embedding(num_embeddings=len(self.text_vocab), embedding_dim=emb_dim)
        self.trg_emb = nn.Embedding(num_embeddings=len(self.gloss_vocab), embedding_dim=emb_dim)

        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        self.outs = nn.Linear(d_model, len(self.gloss_vocab))
    
    def get_inputs(self, batch):
        text = batch["text"]
        gloss = batch["gloss"]
        keypoints = batch["keypoints"]
        frame_lengths = batch["frame_lengths"]

        text = strings_to_indices(text, self.text_vocab)
        gloss = strings_to_indices(gloss, self.gloss_vocab)
        
        text = [torch.tensor(t) for t in text]
        gloss = [torch.tensor(g) for g in gloss]

        text = pad_sequence(text, batch_first=True, padding_value=self.text_vocab["<pad>"])
        gloss = pad_sequence(gloss, batch_first=True, padding_value=self.gloss_vocab["<pad>"])
        keypoints = pad_sequence(keypoints, batch_first=True, padding_value=0.)
        
        text_mask = (text != self.text_vocab["<pad>"])
        gloss_mask = (gloss != self.gloss_vocab["<pad>"])
        keypoints_mask = create_mask(frame_lengths, device=self.device)
        
        # Map tensor to device
        text, gloss, keypoints = map(lambda tensor: tensor.to(self.device), [text, gloss, keypoints])
        text_mask, gloss_mask, keypoints_mask = map(lambda tensor: tensor.to(self.device), [text_mask, gloss_mask, keypoints_mask])

        return (text, text_mask), (gloss, gloss_mask), (keypoints, keypoints_mask)

    def encode(self, src, mask):
        embed = self.src_emb(src)
        embed = self.pos_encoding(embed)
        outs = self.encoder(embed, src_key_padding_mask=~mask)
        
        return outs
    
    def decode(self, enc_outs, trg, mask):
        embed = self.trg_emb(trg)
        embed = self.pos_encoding(embed)
        ar_mask = nn.Transformer.generate_square_subsequent_mask(trg.shape[1]).to(self.device)
        
        outs = self.decoder(embed, enc_outs, tgt_mask=ar_mask, tgt_key_padding_mask=~mask)
        
        return outs
    
    def generate(self, enc_outs, max_len=100, sos_token='<sos>', eos_token='<eos>'):
        '''
        Greedy search
        '''
        batch_size = enc_outs.size(0)
        sos_index = self.gloss_vocab[sos_token]
        eos_index = self.gloss_vocab[eos_token]

        # Initialize the output tensor with SOS tokens
        trg = torch.LongTensor(batch_size, 1).fill_(sos_index).to(enc_outs.device)

        # Autoregressive decoding
        for _ in range(max_len - 1):
            mask = (trg != self.gloss_vocab["<pad>"])
            outs = self.decode(enc_outs, trg, mask) 
            outs = self.outs(outs)
            next_word = outs.argmax(dim=-1)[:, -1:]  # Get the word with the highest probability
            trg = torch.cat((trg, next_word), dim=1)  # Append the predicted word to the target sequence

            # Stop decoding when all sequences in the batch have generated the EOS token
            if (trg == eos_index).any(dim=1).all().item():
                break
        
        return trg
    
    def forward(self, src, src_mask, trg, trg_mask):
        enc_outs = self.encode(src, src_mask)
        dec_outs = self.decode(enc_outs, trg, trg_mask)
        outs = self.outs(dec_outs)
        return outs

    def training_step(self, batch, batch_idx):
        text, gloss, keypoint = self.get_inputs(batch)
        
        text_input, text_mask = text[0], text[1]
        gloss_input, gloss_mask = gloss[0], gloss[1]
        
        outs = self(text_input, text_mask, gloss_input[:, :-1], gloss_mask[:, :-1])

        # Calculate the cross-entropy loss using right-shifted targets and F.cross_entropy
        loss = F.cross_entropy(outs.reshape(-1, outs.size(-1)), 
                               gloss_input[:, 1:].reshape(-1), 
                               label_smoothing=self.label_smoothing, ignore_index=self.gloss_vocab["<pad>"])

        self.log("train/loss", loss, batch_size=outs.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        text, gloss, keypoint = self.get_inputs(batch)
        
        text_input, text_mask = text[0], text[1]
        gloss_input, gloss_mask = gloss[0], gloss[1]

        outs = self(text_input, text_mask, gloss_input[:, :-1], gloss_mask[:, :-1])

        # greedy search
        enc_outs = self.encode(text_input, text_mask)
        generated = self.generate(enc_outs, max_len=self.max_len)
        generated_strings = indices_to_strings(generated, self.gloss_vocab)
        reference_strings = indices_to_strings(gloss_input, self.gloss_vocab)

        # Calculate the cross-entropy loss using right-shifted targets and F.cross_entropy
        loss = F.cross_entropy(outs.reshape(-1, outs.size(-1)), 
                               gloss_input[:, 1:].reshape(-1), 
                               label_smoothing=self.label_smoothing, ignore_index=self.gloss_vocab["<pad>"])

        bleu_dict = get_bleus(generated_strings, reference_strings, split="valid")

        self.log("valid/loss", loss, batch_size=outs.shape[0])
        self.log_dict(bleu_dict, logger=True, batch_size=outs.shape[0])
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            list(self.encoder.parameters())+
            list(self.decoder.parameters())+
            list(self.src_emb.parameters())+
            list(self.trg_emb.parameters())+
            list(self.outs.parameters()), 
            lr=self.lr,
        )
        return opt
    

class TransformerP2T(TransformerT2G):
    def __init__(self, num_joints=120, num_feats=2, noise_config=None, **kwargs):
        super().__init__(**kwargs)

        self.num_joints = num_joints
        self.num_feats = num_feats

        self.src_emb = nn.Linear(num_joints * num_feats, self.d_model)
        self.trg_emb = nn.Embedding(num_embeddings=len(self.text_vocab), embedding_dim=self.emb_dim)
        self.outs = nn.Linear(self.d_model, len(self.text_vocab))

        self.noise_scheduler = instantiate_from_config(noise_config)

    def add_noise(self, x, noise_std=0.5):
        # noise = torch.randn_like(x) * noise_std * x.mean()
        noise = torch.randn_like(x) * noise_std
        return x + noise

    def training_step(self, batch, batch_idx):
        text, gloss, keypoint = self.get_inputs(batch)
        
        text_input, text_mask = text[0], text[1]
        gloss_input, gloss_mask = gloss[0], gloss[1]
        keypoints, keypoints_mask = keypoint[0], keypoint[1]
        
        keypoints = rearrange(keypoints, "b f v c -> b f (v c)")

        noise_std = self.noise_scheduler.step(self.global_step)
        
        outs = self(self.add_noise(keypoints, noise_std), keypoints_mask, text_input[:, :-1], text_mask[:, :-1])
        
        # Calculate the cross-entropy loss using right-shifted targets and F.cross_entropy
        loss = F.cross_entropy(outs.reshape(-1, outs.size(-1)), text_input[:, 1:].reshape(-1))

        self.log("train/loss", loss, batch_size=outs.shape[0])

        return loss
    
    def validation_step(self, batch, batch_idx):
        text, gloss, keypoint = self.get_inputs(batch)
        
        text_input, text_mask = text[0], text[1]
        gloss_input, gloss_mask = gloss[0], gloss[1]
        keypoints, keypoints_mask = keypoint[0], keypoint[1]

        keypoints = rearrange(keypoints, "b f v c -> b f (v c)")

        outs = self(keypoints, keypoints_mask, text_input[:, :-1], text_mask[:, :-1])
        
        # Calculate the cross-entropy loss using right-shifted targets and F.cross_entropy
        loss = F.cross_entropy(outs.reshape(-1, outs.size(-1)), text_input[:, 1:].reshape(-1))

        self.log("valid/loss", loss, batch_size=outs.shape[0])

        return {"keypoint": keypoint, "text": text}
    
    def validation_epoch_end(self, outputs):
        generated_string_list = []
        reference_string_list = []
        for out in outputs:
            keypoints, keypoints_mask = out["keypoint"][0], out["keypoint"][1]
            text, text_mask = out["text"][0], out["text"][1]

            keypoints = rearrange(keypoints, "b f v c -> b f (v c)")

            enc_outs = self.encode(keypoints, keypoints_mask)
            generated = self.generate(enc_outs, max_len=self.max_len)

            generated_strings = indices_to_strings(generated, self.text_vocab)
            generated_string_list += generated_strings

            reference_strings = indices_to_strings(text, self.text_vocab)
            reference_string_list += reference_strings

        bleu_scores_dict = get_bleus(reference_string_list, generated_string_list)
        self.log_dict(bleu_scores_dict, logger=True)