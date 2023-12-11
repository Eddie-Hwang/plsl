import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from transformers import BertConfig, EncoderDecoderModel, EncoderDecoderConfig
from modules.utils import init_tokenizer, evaluate_results, instantiate_from_config, create_mask
from transformers import VivitImageProcessor, VivitModel, VivitConfig

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class HuggingFaceEncoderDecoder(pl.LightningModule):
    def __init__(
        self,
        lr,
        monitor,
        max_len,
        d_model,
        emb_dim,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        n_layers,
        beam_size,
        label_smoothing,
        text_corpus,
        gloss_corpus,
        min_freq,
        tokenizer_path,
        scheduler_config,
        repetition_penalty,
        temperature,
        pos_encoding,
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
        self.n_layers = n_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.pos_encoding = pos_encoding

        # Sentencepiece tokenizer with shared vocab
        self.tokenizer = init_tokenizer(text_corpus, gloss_corpus, min_freq, tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()

        # Special tokens
        self.bos_id = self.tokenizer.token_to_id('<s>')
        self.eos_id = self.tokenizer.token_to_id('</s>')
        self.pad_id = self.tokenizer.token_to_id('<pad>')

        model_config = BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.d_model,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.nhead,
            intermediate_size=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            position_embedding_type=self.pos_encoding
        )
        config = EncoderDecoderConfig.from_encoder_decoder_configs(model_config, model_config)        
        self.model = EncoderDecoderModel(config=config)
        self.model.config.decoder.add_cross_attention = True
        self.model.config.decoder.is_decoder = True

    def get_inputs(self, batch):
        raise NotImplementedError
    
    def forward(self, src, src_mask, trg, trg_mask):
        outs = self.model(
            input_ids=src,
            attention_mask=src_mask,
            decoder_input_ids=trg, 
            decoder_attention_mask=trg_mask,
        )
        return outs.logits
    
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
            list(self.model.parameters()),
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
    

class GlossToText(HuggingFaceEncoderDecoder):
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
            generated = self.model.generate(
                input_ids=gloss,
                attention_mask=gloss_mask,
                num_beams=self.beam_size,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                pad_token_id=self.pad_id,
                bos_token_id=self.bos_id,
                eos_token_id=self.eos_id,
                max_length=self.max_len
            )

            generated_strings = self.tokenizer.decode_batch(generated.tolist(), skip_special_tokens=True)
            reference_strings = self.tokenizer.decode_batch(text.tolist(), skip_special_tokens=True)

            log_dict = evaluate_results(predictions=generated_strings, references=reference_strings, split=split)
            
        return loss, log_dict
    

class SignToText(HuggingFaceEncoderDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.pre = nn.Linear(768, self.d_model)

    def get_inputs(self, batch):
        text = batch["text"]
        video_feature = batch["feature"]

        encoded_text = self.tokenizer.encode_batch(text)
        text_ids, text_mask = [], []
        for encodings in encoded_text:
            text_ids.append(torch.tensor(encodings.ids))
            text_mask.append(torch.tensor(encodings.attention_mask))
        text_ids = pad_sequence(text_ids, padding_value=self.pad_id, batch_first=True)
        text_mask = pad_sequence(text_mask, padding_value=0, batch_first=True).to(torch.bool)
        text_ids, text_mask = map(lambda tensor: tensor.to(self.device), [text_ids, text_mask])

        video_mask = [feature.shape[0] for feature in video_feature]
        video_mask = create_mask(video_mask, device=self.device)

        video_feature = pad_sequence(video_feature, batch_first=True, padding_value=0.)
        
        return {
            "text": text_ids,
            "text_mask": text_mask,
            "video_feature": video_feature,
            "video_mask": video_mask
        }
        
    def forward(self, src, src_mask, trg, trg_mask):
        outs = self.model(
            inputs_embeds=self.pre(src),
            attention_mask=src_mask,
            decoder_input_ids=trg, 
            decoder_attention_mask=trg_mask,
        )
        return outs.logits

    def share_step(self, inputs, split="train"):
        log_dict = {}

        text = inputs["text"]
        text_mask = inputs["text_mask"]
        video_feature = inputs["video_feature"]
        video_mask = inputs["video_mask"]
        
        # Forward
        outs = self(self.pre(video_feature), video_mask, text[:, :-1], text_mask[:, :-1])
        
        # Loss
        loss = F.cross_entropy(outs.reshape(-1, outs.size(-1)), text[:, 1:].reshape(-1), 
                               ignore_index=self.pad_id, label_smoothing=self.label_smoothing)
        
        # Logging
        log_dict[f"{split}/loss"] = loss
        
        # Predict
        if split != "train":
            generated = self.model.generate(
                inputs_embeds=self.pre(video_feature),
                attention_mask=video_mask,
                num_beams=self.beam_size,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                pad_token_id=self.pad_id,
                bos_token_id=self.bos_id,
                eos_token_id=self.eos_id,
                max_length=self.max_len
            )

            generated_strings = self.tokenizer.decode_batch(generated.tolist(), skip_special_tokens=True)
            reference_strings = self.tokenizer.decode_batch(text.tolist(), skip_special_tokens=True)

            log_dict = evaluate_results(predictions=generated_strings, references=reference_strings, split=split)
            
        return loss, log_dict
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            list(self.model.parameters()) +
            list(self.pre.parameters()),
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
        

    