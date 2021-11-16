import torch
import torch.nn as nn
from omegaconf import DictConfig
from astropy.modeling import ParameterError
from dataloader.vocabulary import Vocabulary
from base_model.base_model import Base_model
import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(config,vocab):
    if config.model.architecture.lower() == 'base_model':
        model = build_base_model(num_classes=len(vocab))
    print("model parameter ")
    print(count_parameters(model))
    return model

def build_base_model(num_classes):
    return Base_model(num_classes=num_classes)




