import math
import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn
import numpy as np


def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 3:
        (n_out, n_in, height) = layer.weight.size()
        n = n_in * height
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_lstm(layer):
    """
    Initialises the hidden layers in the LSTM - H0 and C0.

    Input
        layer: torch.Tensor - The LSTM layer
    """
    n_i1, n_i2 = layer.weight_ih_l0.size()
    n_i = n_i1 * n_i2

    std = math.sqrt(2. / n_i)
    scale = std * math.sqrt(3.)
    layer.weight_ih_l0.data.uniform_(-scale, scale)

    if layer.bias_ih_l0 is not None:
        layer.bias_ih_l0.data.fill_(0.)

    n_h1, n_h2 = layer.weight_hh_l0.size()
    n_h = n_h1 * n_h2

    std = math.sqrt(2. / n_h)
    scale = std * math.sqrt(3.)
    layer.weight_hh_l0.data.uniform_(-scale, scale)

    if layer.bias_hh_l0 is not None:
        layer.bias_hh_l0.data.fill_(0.)


def init_att_layer(layer):
    """
    Initilise the weights and bias of the attention layer to 1 and 0
    respectively. This is because the first iteration through the attention
    mechanism should weight each time step equally.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """
    layer.weight.data.fill_(1.)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """
    Initialize a Batchnorm layer.

    Input
        bn: torch.Tensor - The batch normalisation layer
    """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock1d(nn.Module):
    """
    Creates an instance of a 1D convolutional layer. This includes the
    convolutional filter but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, kernel, stride, pad,
                 normalisation, dil=1):
        super(ConvBlock1d, self).__init__()
        self.norm = normalisation
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel,
                               stride=stride,
                               padding=pad,
                               dilation=dil)
        if self.norm == 'bn':
            self.bn1 = nn.BatchNorm1d(out_channels)
        elif self.norm == 'wn':
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
        else:
            self.conv1 = self.conv1
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input):
        """
        Passes the input through the convolutional filter

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm == 'bn':
            x = self.relu(self.bn1(self.conv1(x)))
        else:
            x = self.relu(self.conv1(x))

        return x


class ConvBlock2d(nn.Module):
    """
    Creates an instance of a 2D convolutional layer. This includes the
    convolutional filter but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, kernel, stride, pad,
                 normalisation, att=None):
        super(ConvBlock2d, self).__init__()
        self.norm = normalisation
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel,
                               stride=stride,
                               padding=pad)
        if self.norm == 'bn':
            self.bn1 = nn.BatchNorm2d(out_channels)
        elif self.norm == 'wn':
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
        else:
            self.conv1 = self.conv1
        self.att = att
        if not self.att:
            self.act = nn.ReLU()
        else:
            self.norm = None
            if self.att == 'softmax':
                self.act = nn.Softmax(dim=-1)
            elif self.att == 'global':
                self.act = None
            else:
                self.act = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        if self.att:
            init_att_layer(self.conv1)
        else:
            init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input):
        """
        Passes the input through the convolutional filter

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.att:
            x = self.conv1(x)
            if self.act():
                x = self.act(x)
        else:
            if self.norm == 'bn':
                x = self.act(self.bn1(self.conv1(x)))
            else:
                x = self.act(self.conv1(x))

        return x


class FullyConnected(nn.Module):
    """
    Creates an instance of a fully-connected layer. This includes the
    hidden layers but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, activation, normalisation,
                 att=None):
        super(FullyConnected, self).__init__()
        self.att = att
        self.norm = normalisation
        self.fc = nn.Linear(in_features=in_channels,
                            out_features=out_channels)
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
            self.norm = None
        elif activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
            self.norm = None
        elif activation == 'global':
            self.act = None
            self.norm = None
        else:
            self.act = nn.ReLU()
            if self.norm == 'bn':
                self.bnf = nn.BatchNorm1d(out_channels)
            elif self.norm == 'wn':
                self.wnf = nn.utils.weight_norm(self.fc, name='weight')

        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        if self.att:
            init_att_layer(self.fc)
        else:
            init_layer(self.fc)
        if self.norm == 'bn':
            init_bn(self.bnf)

    def forward(self, input):
        """
        Passes the input through the fully-connected layer

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm is not None:
            if self.norm == 'bn':
                x = self.act(self.bnf(self.fc(x)))
            else:
                x = self.act(self.wnf(x))
        else:
            if self.att:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)
            else:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)        

        return x


class ConvLSTM_Audio(nn.Module):
    def __init__(self, input_dim, output_dim, conv_hidden, lstm_hidden, num_layers, activation, norm, dropout):
        super(ConvLSTM_Audio, self).__init__()
        self.conv = ConvBlock1d(in_channels=input_dim,      # 80
                                out_channels=conv_hidden,   # 128
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')         # ['bn', 'wn', else]
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(dropout)                     # 0.2
        self.lstm = nn.LSTM(input_size=conv_hidden,         # 128
                            hidden_size=lstm_hidden,        # 128
                            num_layers=num_layers,          # 2
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)
        self.fc = FullyConnected(in_channels=lstm_hidden*2,   # 128
                                 out_channels=output_dim,   # 2
                                 activation=activation,     # ['sigmoid', 'softmax', 'global', else]
                                 normalisation=norm)        # ['bn', 'wn']: nn.BatchNorm1d, nn.utils.weight_norm
        self.informer = Informer(lstm_hidden,lstm_hidden*2,4,4,0.1,600)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()
        # x, _ = self.lstm(x)   # output shape: (batch, width//stride(pool), lstm_hidden*2) 5x600x128
        x = self.informer(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))         # output shape: (batch, output_dim)

        return x

# ==== 1. Positional Encoding ====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

# ==== 2. ProbSparse Attention ====
class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(ProbSparseSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        Q, K, V = map(lambda t: t.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), qkv)

        # Attention mechanism
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        prob_attention = torch.softmax(scores, dim=-1)
        attention = torch.matmul(prob_attention, V)

        attention = attention.permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.fc_out(attention)

# ==== 3. Informer Encoder Layer ====
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.self_attn(x)
        x = self.norm1(x + self.dropout1(x2))
        x2 = self.fc(x)
        x = self.norm2(x + self.dropout2(x2))
        return x

# ==== 4. Informer Encoder ====
class InformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout):
        super(InformerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ==== 5. Informer ====
class Informer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dropout, seq_len):
        super(Informer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_len=seq_len)
        self.encoder = InformerEncoder(num_layers, d_model, num_heads, dropout)
        # self.fc_out = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.position_encoding(x)
        x = self.encoder(x)
        # Take the last step's representation for prediction
        # x = self.fc_out(x[:, -1, :])
        return x