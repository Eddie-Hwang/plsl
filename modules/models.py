import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from modules.utils import init_tokenizer, evaluate_results, instantiate_from_config
from modules.positional import PositionalEncoding
from modules.beam_search import AutoRegressiveBeamSearch
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class TransformerEncoderDecoder(pl.LightningModule):
    def __init__(
        self,
        lr,
        monitor,
        max_len,
        d_model,
        emb_dim,
        label_smoothing,
        text_corpus,
        gloss_corpus,
        min_freq,
        tokenizer_path,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        n_layers,
        beam_size,
        scheduler_config,
    ):
        super().__init__()

        self.lr = lr
        self.monitor = monitor
        self.max_len = max_len
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.label_smoothing = label_smoothing
        self.scheduler_config = scheduler_config
        self.beam_size = beam_size

        # Sentencepiece tokenizer with shared vocab
        self.tokenizer = init_tokenizer(text_corpus, gloss_corpus, min_freq, tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()

        # Source and target embedding layer
        self.src_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)
        self.trg_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)

        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                activation=activation, 
                batch_first=True
            ), 
            num_layers=n_layers
        )

        # Transformer decoder
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                activation=activation, 
                batch_first=True
            ), 
            num_layers=n_layers
        )

        # Output linear layer
        self.outs = nn.Linear(self.d_model, self.vocab_size)

        # Special tokens
        self.bos_id = self.tokenizer.token_to_id('<s>')
        self.eos_id = self.tokenizer.token_to_id('</s>')
        self.pad_id = self.tokenizer.token_to_id('<pad>')

        self.beam_search = AutoRegressiveBeamSearch(self.eos_id, beam_size=beam_size)

    def get_inputs(self, batch):
        raise NotImplementedError
    
    def encode(self, src, mask=None):
        embed = self.src_emb(src)
        embed = self.pos_encoding(embed)
        if mask is not None:
            outs = self.encoder(embed, src_key_padding_mask=~mask)
        else:
            outs = self.encoder(embed)
        
        return outs
    
    def decode(self, enc_outs, trg, mask):
        '''
        Autoregressive decoder
        '''
        embed = self.trg_emb(trg)
        embed = self.pos_encoding(embed)
        ar_mask = nn.Transformer.generate_square_subsequent_mask(trg.shape[1]).to(self.device)
        mask = (~mask).to(torch.float32)
        outs = self.decoder(embed, enc_outs, tgt_mask=ar_mask, tgt_key_padding_mask=mask)
        
        return outs

    def forward(self, src, src_mask, trg, trg_mask):
        enc_outs = self.encode(src, src_mask)
        dec_outs = self.decode(enc_outs, trg, trg_mask)
        outs = self.outs(dec_outs)
        return outs

    @torch.no_grad()
    def beam_decode(self, encoder_out, partial_text):
        B, L, C = encoder_out.shape
        beam_size = int(partial_text.shape[0] / B)
        if beam_size > 1:
            # repeat encoder output for batched beam decoding
            encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1)
            encoder_out = encoder_out.view(B * beam_size, L, C)

        if len(partial_text.size()) == 2:
            # not first timestep, pad [BOS] to partial_text
            bos_padding = partial_text.new_full((partial_text.shape[0], 1), self.bos_id).long()
            partial_text = torch.cat((bos_padding, partial_text), dim=1)
        
        mask = (partial_text != self.pad_id).bool()

        if len(partial_text.size()) == 1:
            mask = mask.unsqueeze(-1)
            partial_text = partial_text.unsqueeze(-1)
        
        logits = self.decode(encoder_out, partial_text, mask)
        logits = self.outs(logits)

        # return the logits for the last timestep
        return logits[:, -1, :]

    @torch.no_grad()
    def greedy_decode(self, enc_outs, max_len=100):
        batch_size = enc_outs.size(0)
        
        sos_index = self.bos_id
        eos_index = self.eos_id
        pad_index = self.pad_id

        trg = torch.LongTensor(batch_size, 1).fill_(sos_index).to(enc_outs.device)
        seq_end = torch.zeros(batch_size, dtype=torch.bool).to(enc_outs.device)  # To track which sequences have ended
        
        for _ in range(max_len - 1):
            mask = (trg != pad_index) & ~seq_end.unsqueeze(1)
            import IPython; IPython.embed(); exit(1)
            outs = self.decode(enc_outs, trg, mask)
            outs = self.outs(outs)

            outs = outs[:, -1, :]
            topv, topi = outs.data.topk(1, dim=-1)
            trg = torch.cat((trg, topi), dim=1)

            # Check which sequences have generated the EOS token
            seq_end |= (topi.squeeze() == eos_index)
        
            # Stop decoding if all sequences have generated the EOS token
            if seq_end.all().item():
                break
        
        return trg

    def generate(self, **kwargs):
        if self.beam_size > 1:
            encoder_out = kwargs["enc_outs"]
            B, L, C = encoder_out.shape
            start_predictions = encoder_out.new_full((B,), self.bos_id).long()
            decoding_step = partial(self.beam_decode, encoder_out)
            outputs, _ = self.beam_search.search(start_predictions, decoding_step)
        else:
            outputs = self.greedy_decode(**kwargs)
        
        return outputs

    def share_step(self, inputs, split="train"):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        inputs = self.get_inputs(batch)
        batch_size = inputs["text"].shape[0]
        loss, log_dict = self.share_step(inputs, "train")
        self.log_dict(log_dict, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs = self.get_inputs(batch)
        batch_size = inputs["text"].shape[0]
        _, log_dict = self.share_step(inputs, "valid")
        self.log_dict(log_dict, batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        inputs = self.get_inputs(batch)
        batch_size = inputs["text"].shape[0]
        _, log_dict = self.share_step(inputs, "test")
        self.log_dict(log_dict, batch_size=batch_size)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            list(self.encoder.parameters())+
            list(self.decoder.parameters())+
            list(self.src_emb.parameters())+
            list(self.trg_emb.parameters())+
            list(self.outs.parameters()), 
            lr=self.lr,
        )

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }
            ]
            return [opt], scheduler
        return opt


class GlossToText(TransformerEncoderDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_inputs(self, batch):
        text = batch["text"]
        gloss = batch["gloss"]

        # Text
        encoded_text = self.tokenizer.encode_batch(text)
        text_ids, text_mask = [], []
        for encodings in encoded_text:
            text_ids.append(torch.tensor(encodings.ids))
            text_mask.append(torch.tensor(encodings.attention_mask))
        text_ids = pad_sequence(text_ids, padding_value=self.pad_id, batch_first=True)
        text_mask = pad_sequence(text_mask, padding_value=0, batch_first=True).to(torch.bool)
        text_ids, text_mask = map(lambda tensor: tensor.to(self.device), [text_ids, text_mask])

        # Gloss
        encoded_gloss = self.tokenizer.encode_batch(gloss)
        gloss_ids, gloss_mask = [], []
        for encodings in encoded_gloss:
            gloss_ids.append(torch.tensor(encodings.ids))
            gloss_mask.append(torch.tensor(encodings.attention_mask))
        gloss_ids = pad_sequence(gloss_ids, padding_value=self.pad_id, batch_first=True)
        gloss_mask = pad_sequence(gloss_mask, padding_value=0, batch_first=True).to(torch.bool)
        gloss_ids, gloss_mask = map(lambda tensor: tensor.to(self.device), [gloss_ids, gloss_mask])
        
        return {
            "text": text_ids,
            "text_mask": text_mask,
            "gloss": gloss_ids,
            "gloss_mask": gloss_mask,
        }
    
    def share_step(self, inputs, split="train"):
        log_dict = {}

        gloss = inputs["gloss"]
        gloss_mask = inputs["gloss_mask"]
        text = inputs["text"]
        text_mask = inputs["text_mask"]

        # Forward
        outs = self(gloss, gloss_mask, text[:, :-1], text_mask[:, :-1])
        
        # Loss
        loss = F.cross_entropy(outs.reshape(-1, outs.size(-1)), text[:, 1:].reshape(-1), 
                               ignore_index=self.pad_id, label_smoothing=self.label_smoothing)
        
        # Logging
        log_dict[f"{split}/loss"] = loss
        
        # Predict
        if split != "train":
            enc_outs = self.encode(gloss, gloss_mask)
            generated = self.generate(enc_outs=enc_outs, max_len=self.max_len)

            generated_strings = self.tokenizer.decode_batch(generated.tolist(), skip_special_tokens=True)
            reference_strings = self.tokenizer.decode_batch(text.tolist(), skip_special_tokens=True)

            log_dict = evaluate_results(predictions=generated_strings, references=reference_strings, split=split)
            
        return loss, log_dict
    