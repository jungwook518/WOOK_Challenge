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


def train(config):

    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)
    os.environ["CUDA_VISIBLE_DEVICES"]= config.train.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vocab = KsponSpeechVocabulary(config.train.vocab_label)

    if not config.train.resume: # 학습한 경우가 없으면,
        model = build_model(config, vocab)
        start_epoch =0
        if config.train.multi_gpu == True:
            model = nn.DataParallel(model)
        model = model.to(device)
   
    else: # 학습한 경우가 있으면,
        checkpoint = Checkpoint(config=config)
        latest_checkpoint_path = checkpoint.get_latest_checkpoint()
        resume_checkpoint = checkpoint.load(latest_checkpoint_path)
        model = resume_checkpoint.model
        optimizer = resume_checkpoint.optimizer
        start_epoch = resume_checkpoint.epoch+1
        if isinstance(model, nn.DataParallel):
            model = model.module
        if config.train.multi_gpu == True:
            model = nn.DataParallel(model)
        model = model.to(device)    


    print(model)

    train_metric,val_metric = CharacterErrorRate(vocab),CharacterErrorRate(vocab)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=2,min_lr=0.000001)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id,reduction='sum')
    

    tensorboard_path = 'outputs/tensorboard/'+str(config.model.architecture)+'/'+str(config.train.exp_day)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    summary = SummaryWriter(tensorboard_path)

    trainset = prepare_dataset(config, config.train.transcripts_path_train,vocab, Train=True)
    validset = prepare_dataset(config, config.train.transcripts_path_valid,vocab, Train=False)

    train_loader = torch.utils.data.DataLoader(dataset=trainset,batch_size =config.train.batch_size,
                                shuffle=True,collate_fn = _collate_fn, num_workers=config.train.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset=validset,batch_size =config.train.batch_size,
                                shuffle=False,collate_fn = _collate_fn, num_workers=config.train.num_workers)
    
    log_format = "epoch: {:4d}/{:4d}, step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    train_begin_time = time.time()
    for epoch in range(start_epoch, config.train.num_epochs):
        #######################################           Train             ###############################################
        print('Train %d epoch start' % epoch)
        cer = 1.0
        epoch_loss_total = 0.
        total_num = 0
        timestep = 0
        model.train()
        begin_time = epoch_begin_time = time.time() #모델 학습 시작 시간
        for i, (video_inputs,audio_inputs,targets,video_input_lengths,audio_input_lengths,target_lengths) in enumerate(train_loader):
            optimizer.zero_grad()
            video_inputs = video_inputs.to(device)
            audio_inputs = audio_inputs.to(device)
            targets = targets.to(device)
            video_input_lengths = video_input_lengths.to(device)
            audio_input_lengths = audio_input_lengths.to(device)
            target_lengths = torch.as_tensor(target_lengths).to(device)
            model = model
            outputs, encoder_output_lengths = model(video_inputs, 
                                    video_input_lengths, 
                                    audio_inputs,
                                    audio_input_lengths,
                                    targets,
                                    target_lengths,
                                    )

            # pdb.set_trace()
            loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), targets[:, 1:].contiguous().view(-1))
            y_hats = outputs.max(-1)[1]

            cer = train_metric(targets[:, 1:], y_hats)
            loss.backward()
            optimizer.step()


            total_num += int(audio_input_lengths.sum())  
            epoch_loss_total += loss.item()

            timestep += 1
            torch.cuda.empty_cache()
            
            if timestep % config.train.print_every == 0:
                current_time = time.time()
                elapsed = current_time - begin_time
                epoch_elapsed = (current_time - epoch_begin_time) / 60.0
                train_elapsed = (current_time - train_begin_time) / 3600.0
                
                print(log_format.format(epoch,config.train.num_epochs,
                    timestep, len(train_loader), loss,
                    cer, elapsed, epoch_elapsed, train_elapsed,
                    optimizer.state_dict()['param_groups'][0]['lr'],
                ))
                begin_time = time.time()
            summary.add_scalar('iter_training/loss',loss,epoch*len(train_loader)+i)
            summary.add_scalar('iter_training/cer',cer,epoch*len(train_loader)+i)
            pdb.set_trace()
        print('Train %d completed' % epoch)
        Checkpoint(model, optimizer, epoch, config=config).save()
        model, train_loss, train_cer = model, epoch_loss_total / total_num, cer
        summary.add_scalar('training/loss',train_loss,epoch)
        summary.add_scalar('training/cer',train_cer,epoch)
        print('Epoch %d Training Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

        train_metric.reset()
        

if __name__ == '__main__':
    config = OmegaConf.load('train.yaml')
    train(config)
    

