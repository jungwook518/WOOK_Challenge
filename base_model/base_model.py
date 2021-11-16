from torch import Tensor
from typing import Tuple
import torch
import torch.nn as nn
import pdb

class Base_model(nn.Module):

    def __init__(self,num_classes):
        
        super(Base_model, self).__init__()

        self.num_classes = num_classes
        self.visual_frontend = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5,7,7), stride=(1,2,2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),
            )
        self.fc1 = nn.Linear(96,128)
        self.fc2 = nn.Linear(80,128)
        self.fc3 = nn.Linear(256,self.num_classes)


    def forward(
            self,
            video_inputs: Tensor,
            video_input_lengths: Tensor,
            audio_inputs: Tensor,
            audio_input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
    ):
        video_inputs = self.visual_frontend(video_inputs)
        video_inputs = video_inputs.mean(3)
        video_inputs = video_inputs.mean(3)
        video_inputs = video_inputs.permute(0,2,1)
        vid_encoder_outputs = self.fc1(video_inputs)
        aud_encoder_outputs = self.fc2(audio_inputs)
        #B T F --> B F T
        vid_encoder_outputs = vid_encoder_outputs.permute(0,2,1)
        aud_encoder_outputs = aud_encoder_outputs.permute(0,2,1)
        vid_encoder_outputs_upsample = torch.nn.functional.interpolate(
                vid_encoder_outputs,
                aud_encoder_outputs.size(2)
                )
            
        aud_encoder_outputs = aud_encoder_outputs.permute(0,2,1)
        vid_encoder_outputs_upsample = vid_encoder_outputs_upsample.permute(0,2,1)
        fusion_encoder_out = torch.cat((aud_encoder_outputs,vid_encoder_outputs_upsample),2)
        fusion_encoder_out = self.fc3(fusion_encoder_out)

        
        return fusion_encoder_out.log_softmax(dim=-1)



