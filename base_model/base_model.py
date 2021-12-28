from torch import Tensor
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
import math
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

    def inference(
            self,
            video_inputs: Tensor,
            video_input_lengths: Tensor,
            audio_inputs: Tensor,
            audio_input_lengths: Tensor,
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





###############################################
#####          Transformer         ############
###############################################

class VGGExtractor_fairseq(nn.Module):

    def __init__(
            self,
            input_dim: int,
            in_channels: int = 1,
            out_channels: int or tuple = (64, 128),
            activation: str = 'relu',
    ):
        super(VGGExtractor_fairseq, self).__init__()
        self.in_channels = in_channels
        self.input_dim = input_dim
        self.out_channels = out_channels
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[0]),
                nn.ReLU(),
                nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[0]),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[1]),
                nn.ReLU(),
                nn.Conv2d(out_channels[1], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[1]),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
            )
    def get_output_dim(self):
        if isinstance(self, VGGExtractor_fairseq):
            output_dim = (self.input_dim - 1) << 5 if self.input_dim % 2 else self.input_dim << 5

        return output_dim
    
    def get_output_lengths(self, seq_lengths: Tensor):
        assert self.conv is not None, "self.conv should be defined"

        for module in self.conv:
            if isinstance(module, nn.Conv2d):
                numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
                seq_lengths = numerator.float() / float(module.stride[1])
                seq_lengths = seq_lengths.int() + 1

            elif isinstance(module, nn.MaxPool2d):
                seq_lengths >>= 1

        return seq_lengths.int()

    def forward(self, inputs, input_lengths):
        # pdb.set_trace()
        outputs = self.conv(inputs.unsqueeze(1))
        
        output_lengths = self.get_output_lengths(input_lengths)

        batch_size, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = outputs.view(batch_size, seq_lengths, channels * dimension)
        return outputs, output_lengths

class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """
    def __init__(
            self,
            module: nn.Module,
            module_factor: float = 1.0,
            input_factor: float = 1.0,
    ):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor):
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor):
        return self.linear(x)


class View(nn.Module):
    """ Wrapper class of torch.view() for Sequential module. """
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, inputs):
        if self.contiguous:
            inputs = inputs.contiguous()
        return inputs.view(*self.shape)


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, inputs: Tensor):
        return inputs.transpose(*self.shape)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout_p: float = 0.3) :
        super(PositionwiseFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            Linear(d_model, d_ff),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            Linear(d_ff, d_model),
            nn.Dropout(dropout_p),
        )

    def forward(self, inputs: Tensor):
        return self.feed_forward(inputs)

def get_attn_pad_mask(inputs, input_lengths, expand_length):
    """ mask position is set to 1 """
    # pdb.set_trace()
    def get_transformer_non_pad_mask(inputs: Tensor, input_lengths: Tensor) -> Tensor:
        """ Padding position is set to 0, either use input_lengths or pad_id """
        batch_size = inputs.size(0)
        #pdb.set_trace()
        if len(inputs.size()) == 2:
            non_pad_mask = inputs.new_ones(inputs.size())  # B x T
        elif len(inputs.size()) == 3:
            non_pad_mask = inputs.new_ones(inputs.size()[:-1])  # B x T
        else:
            raise ValueError(f"Unsupported input shape {inputs.size()}")

        for i in range(batch_size):
            non_pad_mask[i, input_lengths[i]:] = 0

        return non_pad_mask

    non_pad_mask = get_transformer_non_pad_mask(inputs, input_lengths)
    pad_mask = non_pad_mask.lt(1)
    attn_pad_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_pad_mask


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)

    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask

class ScaledDotProductAttention(nn.Module):

    def __init__(self, dim, scale = True):
        super(ScaledDotProductAttention, self).__init__()
        if scale:
            self.sqrt_dim = np.sqrt(dim)
        else:
            self.sqrt_dim = 1

    def forward(self,query,key,value,mask = None):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn

class MultiHeadAttention(nn.Module):

    def __init__(self, dim = 512, num_heads = 8):
        super(MultiHeadAttention, self).__init__()

        assert dim % num_heads == 0, "hidden_dim % num_heads should be zero."

        self.d_head = int(dim / num_heads)
        self.num_heads = num_heads
        self.query_proj = Linear(dim, self.d_head * num_heads)
        self.key_proj = Linear(dim, self.d_head * num_heads)
        self.value_proj = Linear(dim, self.d_head * num_heads)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head, scale=True)

    def forward(self,query,key,value,mask = None):
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)

        return context, attn

class TransformerEncoderLayer(nn.Module):

    def __init__(
            self,
            d_model: int = 512,             # dimension of model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.attention_prenorm = nn.LayerNorm(d_model)
        self.feed_forward_prenorm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    def forward(self, inputs: Tensor, self_attn_mask: Tensor = None):
        # pdb.set_trace()
        residual = inputs
        inputs = self.attention_prenorm(inputs)
        outputs, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs, attn

class AVTransformerEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,                         # dimension of feature vector
            d_model: int = 512,                     # dimension of model
            d_ff: int = 2048,                       # dimension of feed forward network
            num_layers: int = 6,                    # number of enc oder layers
            num_heads: int = 8,                     # number of attention heads
            dropout_p: float = 0.3,                 # probability of dropout
            num_classes: int = None,                # number of classification
    ):
        super(AVTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.d_ff = d_ff
        self.dropout_p = dropout_p
        self.extractor = VGGExtractor_fairseq(input_dim)
        self.conv_output_dim = self.extractor.get_output_dim()

        self.input_proj = Linear(self.conv_output_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
            ) for _ in range(num_layers)
        ])

    def forward(self, inputs, input_lengths):
        # pdb.set_trace()
        conv_outputs, output_lengths = self.extractor(inputs, input_lengths)

        self_attn_mask = get_attn_pad_mask(conv_outputs, output_lengths, conv_outputs.size(1))
        outputs = self.input_norm(self.input_proj(conv_outputs))
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs, attn = layer(outputs, self_attn_mask)

        return outputs, output_lengths


class TransformerDecoderLayer(nn.Module):
    """
    DecoderLayer is made up of self-attention, multi-head attention and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)
    """

    def __init__(
            self,
            d_model: int = 512,             # dimension of model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention_prenorm = nn.LayerNorm(d_model)
        self.encoder_attention_prenorm = nn.LayerNorm(d_model)
        self.feed_forward_prenorm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    def forward(
            self,
            inputs: Tensor,
            encoder_outputs: Tensor,
            self_attn_mask: Tensor = None,
            encoder_outputs_mask: Tensor = None
    ) :
        # pdb.set_trace()
        residual = inputs
        inputs = self.self_attention_prenorm(inputs)
        outputs, self_attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs += residual

        residual = outputs
        outputs = self.encoder_attention_prenorm(outputs)
        outputs, encoder_attn = self.encoder_attention(outputs, encoder_outputs, encoder_outputs, encoder_outputs_mask)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs, self_attn, encoder_attn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.
    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, d_model: int = 512, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int):
        return self.pe[:, :length]


class Embedding(nn.Module):
    """
    Embedding layer. Similarly to other sequence transduction models, transformer use learned embeddings
    to convert the input tokens and output tokens to vectors of dimension d_model.
    In the embedding layers, transformer multiply those weights by sqrt(d_model)
    """
    def __init__(self, num_embeddings: int, pad_id: int, d_model: int = 512):
        super(Embedding, self).__init__()
        self.sqrt_dim = math.sqrt(d_model)
        self.embedding = nn.Embedding(num_embeddings, d_model, padding_idx=pad_id)

    def forward(self, inputs: Tensor):
        return self.embedding(inputs) * self.sqrt_dim

class TransformerDecoder(nn.Module):


    def __init__(
            self,
            num_classes: int,               # number of classes
            d_model: int = 512,             # dimension of model
            d_ff: int = 512,                # dimension of feed forward network
            num_layers: int = 6,            # number of decoder layers
            num_heads: int = 8,             # number of attention heads
            dropout_p: float = 0.3,         # probability of dropout
            pad_id: int = 0,                # identification of pad token
            sos_id: int = 1,                # identification of start of sentence token
            eos_id: int = 2,                # identification of end of sentence token
            max_length: int = 400,          # max length of decoding
    ) :
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.embedding = Embedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
            ) for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            Linear(d_model, num_classes, bias=False),
        )

    def forward_step(
            self,
            decoder_inputs,
            decoder_input_lengths,
            encoder_outputs,
            encoder_output_lengths,
            positional_encoding_length,
    ) :
        #pdb.set_trace()
        dec_self_attn_pad_mask = get_attn_pad_mask(
            decoder_inputs, decoder_input_lengths, decoder_inputs.size(1)
        )
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(decoder_inputs)
        self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        encoder_attn_mask = get_attn_pad_mask(
            encoder_outputs, encoder_output_lengths, decoder_inputs.size(1)
        )

        outputs = self.embedding(decoder_inputs) + self.positional_encoding(positional_encoding_length)
        outputs = self.input_dropout(outputs)
        #pdb.set_trace()
        for layer in self.layers:
            outputs, self_attn, memory_attn = layer(
                inputs=outputs,
                encoder_outputs=encoder_outputs,
                self_attn_mask=self_attn_mask,
                encoder_outputs_mask=encoder_attn_mask,
            )

        return outputs

    def forward(
            self,
            targets: Tensor,
            encoder_outputs: Tensor,
            encoder_output_lengths: Tensor,
            target_lengths: Tensor,
    ) :
        #pdb.set_trace()
        batch_size = encoder_outputs.size(0)

        targets = targets[targets != self.eos_id].view(batch_size, -1)
        target_length = targets.size(1)

        outputs = self.forward_step(
            decoder_inputs=targets,
            decoder_input_lengths=target_lengths,
            encoder_outputs=encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            positional_encoding_length=target_length,
        )
        return self.fc(outputs).log_softmax(dim=-1)

    @torch.no_grad()
    def decode(self, encoder_outputs: Tensor, encoder_output_lengths: Tensor) -> Tensor:

        logits = list()
        batch_size = encoder_outputs.size(0)

        input_var = encoder_outputs.new_zeros(batch_size, self.max_length).long()
        input_var = input_var.fill_(self.pad_id)
        input_var[:, 0] = self.sos_id

        for di in range(1, self.max_length):
            input_lengths = torch.IntTensor(batch_size).fill_(di)
            # pdb.set_trace()
            outputs = self.forward_step(
                decoder_inputs=input_var[:, :di],
                decoder_input_lengths=input_lengths,
                encoder_outputs=encoder_outputs,
                encoder_output_lengths=encoder_output_lengths,
                positional_encoding_length=di,
            )
            # pdb.set_trace()
            step_output = self.fc(outputs).log_softmax(dim=-1)

            logits.append(step_output[:, -1, :])
            input_var[:,di] = logits[-1].topk(1)[1].squeeze(1)

        return torch.stack(logits, dim=1)










class AV_Transformer(nn.Module):
    
    def __init__(
            self,
            input_vid_dim: int,
            input_dim: int,
            num_classes: int,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            encoder_dropout_p: float = 0.2,
            decoder_dropout_p: float = 0.2,
            d_model: int = 512,
            d_ff: int = 2048,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            num_heads: int = 8,
            max_length: int = 400,
            ) :
        super(AV_Transformer, self).__init__()
        
        self.video_encoder = AVTransformerEncoder(
            input_dim=input_vid_dim,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout_p=encoder_dropout_p,
            num_classes=num_classes,
        )
        self.audio_encoder = AVTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout_p=encoder_dropout_p,
            num_classes=num_classes,
        )
        self.decoder = TransformerDecoder(
            num_classes=num_classes,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout_p=decoder_dropout_p,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
            max_length=max_length,
        )
        

        self.num_classes = num_classes
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_length = max_length

        self.visual_frontend = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),
            )

        self.fc1 = nn.Sequential(
                    nn.Linear(d_model*2,d_model*4),
                    Transpose(shape=(1, 2)),
                    nn.BatchNorm1d(d_model*4),
                    Transpose(shape=(1, 2)),
                    nn.ReLU(),
                    Linear(d_model*4,d_model),
                    )


    def forward(
            self,
            video_inputs: Tensor,
            video_input_lengths: Tensor,
            audio_inputs: Tensor,
            audio_input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
    ):
        #pdb.set_trace()
        video_inputs = self.visual_frontend(video_inputs)
        video_inputs = video_inputs.mean(3)
        video_inputs = video_inputs.mean(3)
        video_inputs = video_inputs.permute(0,2,1) # B T F
        # pdb.set_trace()
        vid_encoder_outputs, vid_output_lengths = self.video_encoder(video_inputs, video_input_lengths)
        aud_encoder_outputs, aud_output_lengths = self.audio_encoder(audio_inputs, audio_input_lengths)
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
        fusion_encoder_out = self.fc1(fusion_encoder_out)

        predicted_log_probs = self.decoder(targets, fusion_encoder_out, aud_output_lengths, target_lengths)

        return predicted_log_probs, aud_output_lengths

    @torch.no_grad()
    def recognize(
            self,
            video_inputs: Tensor,
            video_input_lengths: Tensor,
            audio_inputs: Tensor,
            audio_input_lengths: Tensor,
    ):
        #pdb.set_trace()
        video_inputs = self.visual_frontend(video_inputs)
        video_inputs = video_inputs.mean(3)
        video_inputs = video_inputs.mean(3)
        video_inputs = video_inputs.permute(0,2,1)
        vid_encoder_outputs, vid_output_lengths,_ = self.video_encoder(video_inputs, video_input_lengths)
        aud_encoder_outputs, aud_output_lengths,_ = self.audio_encoder(audio_inputs, audio_input_lengths)
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
        fusion_encoder_out = self.fc1(fusion_encoder_out)
        return self.decoder.decode(fusion_encoder_out, aud_output_lengths)


# class Transformer_aonly(EncoderDecoderModel):

#     def __init__(
#             self,
#             input_dim: int,
#             num_classes: int,
#             extractor: str,
#             num_encoder_layers: int = 12,
#             num_decoder_layers: int = 6,
#             encoder_dropout_p: float = 0.2,
#             decoder_dropout_p: float = 0.2,
#             d_model: int = 512,
#             d_ff: int = 2048,
#             pad_id: int = 0,
#             sos_id: int = 1,
#             eos_id: int = 2,
#             num_heads: int = 8,
#             joint_ctc_attention: bool = False,
#             max_length: int = 400,
#     ) :
#         assert d_model % num_heads == 0, "d_model % num_heads should be zero."
#         encoder = TransformerEncoder(
#             input_dim=input_dim,
#             extractor=extractor,
#             d_model=d_model,
#             d_ff=d_ff,
#             num_layers=num_encoder_layers,
#             num_heads=num_heads,
#             dropout_p=encoder_dropout_p,
#             joint_ctc_attention=joint_ctc_attention,
#             num_classes=num_classes,
#         )
#         decoder = TransformerDecoder(
#             num_classes=num_classes,
#             d_model=d_model,
#             d_ff=d_ff,
#             num_layers=num_decoder_layers,
#             num_heads=num_heads,
#             dropout_p=decoder_dropout_p,
#             pad_id=pad_id,
#             sos_id=sos_id,
#             eos_id=eos_id,
#             max_length=max_length,
#         )
#         super(Transformer_aonly, self).__init__(encoder, decoder)

#         self.num_classes = num_classes
#         self.joint_ctc_attention = joint_ctc_attention
#         self.sos_id = sos_id
#         self.eos_id = eos_id
#         self.pad_id = pad_id
#         self.max_length = max_length

#     def forward(
#             self,
#             video_inputs: Tensor,
#             video_input_lengths: Tensor,
#             audio_inputs: Tensor,
#             audio_input_lengths: Tensor,
#             targets: Tensor,
#             target_lengths: Tensor,
#     ):
#         # pdb.set_trace()
#         encoder_outputs, output_lengths, encoder_log_probs  = self.encoder(audio_inputs, audio_input_lengths)
#         predicted_log_probs = self.decoder(targets, encoder_outputs, output_lengths, target_lengths)
        

#         return predicted_log_probs, output_lengths,encoder_log_probs
    
#     @torch.no_grad()
#     def recognize(
#             self,
#             video_inputs: Tensor,
#             video_input_lengths: Tensor,
#             audio_inputs: Tensor,
#             audio_input_lengths: Tensor,
#     ):
#         encoder_outputs, output_lengths, encoder_log_probs = self.encoder(audio_inputs, audio_input_lengths)
#         return self.decoder.decode(encoder_outputs, output_lengths)