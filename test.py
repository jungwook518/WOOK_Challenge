import os,random,warnings,time,math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dataloader.data_loader import prepare_dataset, _collate_fn
from builder.model_builder import build_model
from dataloader.vocabulary import Vocabulary,KsponSpeechVocabulary
from configparser import ConfigParser
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from metric.metric import CharacterErrorRate,WordErrorRate
from checkpoint.checkpoint import Checkpoint
from typing import Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb

def test(config):
    os.environ["CUDA_VISIBLE_DEVICES"]= config.train.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vocab = KsponSpeechVocabulary(config.train.vocab_label)
    test_model = config.train.test_model
    model = torch.load(test_model, map_location=lambda storage, loc: storage)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model = model.to(device)
    model.eval()

    test_metric = CharacterErrorRate(vocab)
    test_metric_wer = WordErrorRate(vocab)
    testset = prepare_dataset(config,config.train.transcripts_path_test,vocab,Train=False)                                
    test_loader = torch.utils.data.DataLoader(dataset=testset,batch_size =config.train.batch_size,
                                shuffle=False,collate_fn = _collate_fn, num_workers=config.train.num_workers)

    model.eval()
    print('test start!!!')

    for i, (video_inputs,audio_inputs,targets,video_input_lengths,audio_input_lengths,target_lengths) in enumerate(tqdm(test_loader)):
        
        video_inputs = video_inputs.to(device)
        audio_inputs = audio_inputs.to(device)
        targets = targets.to(device)
        video_input_lengths = video_input_lengths.to(device)
        audio_input_lengths = audio_input_lengths.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        
        outputs = model.recognize(video_inputs, 
                                video_input_lengths, 
                                audio_inputs,
                                audio_input_lengths,
                                )
        
        y_hats = outputs.max(-1)[1]
        cer = test_metric(targets[:, 1:], y_hats)
        print(cer)
        
        wer = test_metric_wer(targets[:, 1:], y_hats)
        print(wer)
    
    print('Test Over')
    print('Mean CER : ', cer)
    print('Mean WER : ', wer)

if __name__ == '__main__':
    # pdb.set_trace()
    config = OmegaConf.load('test.yaml')
    test(config)
    

