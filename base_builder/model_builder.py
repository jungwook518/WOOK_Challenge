import torch
import torch.nn as nn
from omegaconf import DictConfig
from astropy.modeling import ParameterError
from dataloader.vocabulary import Vocabulary
from base_model.base_model import Base_model,AV_Transformer
import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(config,vocab):

    input_size = config.audio.n_mels
    input_vid_size = config.video.input_feat

    if config.model.architecture.lower() == 'base_model':
        model = build_base_model(num_classes=len(vocab))

    if config.model.architecture.lower() == 'transformer_model':
        model = build_Transformer_model(
            num_classes=len(vocab),
            input_dim=input_size,
            input_vid_dim=input_vid_size,
            d_model=config.model.d_model,
            d_ff=config.model.d_ff,
            num_heads=config.model.num_heads,
            pad_id=vocab.pad_id,
            sos_id=vocab.sos_id,
            eos_id=vocab.eos_id,
            max_length=config.model.max_len,
            num_encoder_layers=config.model.num_encoder_layers,
            num_decoder_layers=config.model.num_decoder_layers,
            dropout_p=config.model.dropout,
        )
        

    print("model parameter ")
    print(count_parameters(model))
    return model

def build_base_model(num_classes):
    return Base_model(num_classes=num_classes)

def build_Transformer_model(num_classes,
            input_dim,
            input_vid_dim,
            d_model,
            d_ff,
            num_heads,
            pad_id,
            sos_id,
            eos_id,
            max_length,
            num_encoder_layers,
            num_decoder_layers,
            dropout_p
            ):
    return AV_Transformer(input_vid_dim =input_vid_dim ,
            input_dim=input_dim,
            num_classes=num_classes,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            encoder_dropout_p = dropout_p,
            decoder_dropout_p=dropout_p,
            d_model=d_model,
            d_ff=d_ff,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
            num_heads=num_heads,
            max_length=max_length)




