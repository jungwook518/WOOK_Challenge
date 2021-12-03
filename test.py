import os,random,warnings,time,math
import torch
import torch.nn as nn
from dataloader.data_loader import prepare_dataset, _collate_fn
from base_builder.model_builder import build_model
from dataloader.vocabulary import KsponSpeechVocabulary
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from metric.metric import CharacterErrorRate
from checkpoint.checkpoint import Checkpoint
from torch.utils.data import DataLoader
import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def test(config):

    os.environ["CUDA_VISIBLE_DEVICES"]= config.train.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    vocab = KsponSpeechVocabulary(config.train.vocab_label)
    test_model = config.train.test_model
    model = torch.load(test_model, map_location=lambda storage, loc: storage).to(device)


    model.eval()
    test_metric = CharacterErrorRate(vocab)
    print(model)
    print(count_parameters(model))
    pdb.set_trace()
    model.eval()  


    print(model)


    testset = prepare_dataset(config, config.train.transcripts_path_test,vocab, Train=False)

    test_loader = torch.utils.data.DataLoader(dataset=testset,batch_size =config.train.batch_size,
                                shuffle=False,collate_fn = _collate_fn, num_workers=config.train.num_workers)
    

    start_time = time.time()
    with torch.no_grad():
        for i, (video_inputs,audio_inputs,targets,video_input_lengths,audio_input_lengths,target_lengths) in enumerate(test_loader):

            video_inputs = video_inputs.to(device)
            audio_inputs = audio_inputs.to(device)
            targets = targets.to(device)
            video_input_lengths = video_input_lengths.to(device)
            audio_input_lengths = audio_input_lengths.to(device)
            target_lengths = torch.as_tensor(target_lengths).to(device)
            model = model
            outputs = model(video_inputs, 
                                    video_input_lengths, 
                                    audio_inputs,
                                    audio_input_lengths,
                                    targets,
                                    target_lengths,
                                    )

            y_hats = outputs.max(-1)[1]
            cer = test_metric(targets[:, 1:], y_hats)
            print(cer)
    
    print("Total Time")
    print(time.time() - start_time)

            

        

if __name__ == '__main__':
    config = OmegaConf.load('test.yaml')
    test(config)
    

