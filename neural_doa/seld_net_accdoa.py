# The SELDnet architecture
# Based on https://github.com/sharathadavanne/doa-net with modifications.
# Combines ideas from:
# - Shimada et al. (2020): "ACCDOA: Activity-Coupled Cartesian Direction of Arrival Representation for Sound Event Localization and Detection" 
# - Adavanne et al. (2018): "Sound Event Localization and Detection of Overlapping Sources Using Convolutional Recurrent Neural Networks"


import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

warnings.filterwarnings('ignore')
EPS = torch.finfo(torch.float32).eps

def print_memory_usage(layer_name):
    allocated = torch.cuda.memory_allocated() / 1024**2  
    reserved = torch.cuda.memory_reserved() / 1024**2    
    print(f"{layer_name} - Allocated: {allocated:.2f} MiB, Reserved: {reserved:.2f} MiB")


class SELD_net(torch.nn.Module):
    """SELD Net with multi acticity-coupled cartesian direction of arrival (multi-accdoa) output"""
    
    def __init__(self,
                 #feature_extractor,
                 input_shape,
                 n_conv_filters = 64,
                 pooling_size = 8, #8,
                 conv_dropout = 0.2,
                 rnn_type = "gru",
                 num_rnn_layers = 2,
                 hidden_size = 128,
                 rnn_dropout = 0.0,
                 bidirectional = True,
                 linear_dim = 128,
                 fc_dropout = 0.0,
                 sequence_length=512,
                 n_doa_dims = 3, #2,
                 n_classes = 1,
                 n_tracks = 1): #2
        super(SELD_net, self).__init__()

        """ init of SELD-Net

        Parameters: feature_extractor: class
                            class for feature extraction, must be child of FeatureExtractor class
                    n_conv_filters: int
                            number of filters in convolutional layer
                    pooling_size: int
                            size of first dimension of pooling kernel
                    conv_dropout: float 
                            dropout of convolutional layer
                    rnn_type: str ("gru","lstm","rnn")
                            type of recurrent neural network. Default to gru
                    num_rnn_layers: int
                            number of recurrent layer
                    reduced_bins: int

                    hidden_size: int
                            size of recurrent hidden layers
                    rnn_dropout: float
                            dropout of recurrent layers
                    bidirectional: boolean 
                            if true, bidirectional recurrent layer is used
                    linear_dim: int
                            dimension of fc layer for SED and DOA output
                    fc_dropout: float
                            dropout of fc layer
                    sequence_length: int
                            number of time frames processed as a chunk
                    n_doa_dims: int 2,3
                            number of DOA output dimension, 2 for 2-dimensional or 3 for 3-dimensional
                    n_classes: int
                            number of SED outputs, equal to number of classes
                    n_tracks: int
                            number of tracks, i.e. number of possible overlapping events of same class
        """
        #self.feature_extractor = feature_extractor
        self.n_conv_filters = n_conv_filters
        self.pooling_size = pooling_size
        self.conv_dropout = conv_dropout
        self.rnn_type = rnn_type
        self.num_rnn_layers = num_rnn_layers
        #self.reduced_bins = reduced_bins
        self.in_rnn_size = int(input_shape[-1]/(pooling_size**3))*n_conv_filters
        self.hidden_size = hidden_size
        self.rnn_dropout = rnn_dropout
        self.bidirectional = bidirectional
        self.linear_dim = linear_dim
        self.fc_dropout = fc_dropout
        self.sequence_length = sequence_length
        self.n_doa_dims = n_doa_dims
        self.n_classes = n_classes
        self.n_tracks = n_tracks    

        #first conv layer
        self.conv1 = Conv_layer(input_shape[1]*input_shape[2], self.n_conv_filters, kernel_size=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,self.pooling_size))
        self.dropout1 = nn.Dropout(self.conv_dropout)
        
        #second conv layer
        self.conv2 = Conv_layer(self.n_conv_filters, self.n_conv_filters, kernel_size=(3, 3))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,self.pooling_size))
        self.dropout2 = nn.Dropout(self.conv_dropout)
        
        #Third conv layer
        self.conv3 = Conv_layer(self.n_conv_filters, self.n_conv_filters, kernel_size=(3, 3))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1,self.pooling_size))
        # self.maxpool3 = nn.MaxPool2d(kernel_size=(1,self.pooling_size))
        self.dropout3 = nn.Dropout(self.conv_dropout)
        
        #RNN layers
        self.rnn_type = self.rnn_type.upper()
        if self.rnn_type not in ['RNN', 'LSTM', 'GRU']:
            raise ValueError("Unsupported rnn type: {}".format(self.rnn_type))
        #self.rnn = getattr(torch.nn, self.rnn_type)(self.n_conv_filters*reduced_bins, self.hidden_size, self.num_rnn_layers, batch_first=True, dropout=self.rnn_dropout, bidirectional=self.bidirectional)
        self.rnn = getattr(torch.nn, self.rnn_type)(self.in_rnn_size, self.hidden_size, self.num_rnn_layers, batch_first=True, dropout=self.rnn_dropout, bidirectional=self.bidirectional)
        self.rnn.flatten_parameters()

        #Linear layers for DOA
        self.doa = nn.Sequential(nn.Linear(self.hidden_size if self.bidirectional else self.hidden_size, linear_dim),
                                 nn.Dropout(p=self.fc_dropout),
                                 nn.Linear(linear_dim, self.n_tracks*self.n_classes*self.n_doa_dims))
        self.non_linear_doa = nn.Tanh()


        
    def forward(self, features, lengths, hidden_state=None):
        """forward function to process data
        
        Parameters: signal: torch tensor
                        input audio features (magnitude and phase) or GCC-PHAT

        Returns:    sed: torch tensor
                        estimated sound event detection per block
                    doa: torch tensor
                        estimated direction of arrival as 2D or 3D coordinates per block
        """
        
        # transpose channels with samples
        #signal = torch.transpose(signal,2,1)

        # do max normalization
        #signal = signal / torch.max(torch.abs(signal))
        
        # extract features
        #features = self.feature_extractor(signal)

        # transpose features for model input
        #features = torch.transpose(features,3,2)

        # add additional dimension if no batch is used
        #if features.dim() == 1:
        #    x = torch.unsqueeze(features, 0)

        f_shape = features.shape
        # add additional dimension if no batch is used
        if len(f_shape) != 5:
            features = torch.unsqueeze(features, 0)
            f_shape = features.shape
        '''(batch_size, time_steps, channel, n_mic_pairs, freq_bins) to (batch_size, n_mic_pairs, time_steps, freq_bins)''' 
        x = features.permute(0, 2, 3, 1, 4).contiguous().view(f_shape[0], f_shape[2]*f_shape[3], f_shape[1], f_shape[4])
        # make chunks of features
        # y_chunk = torch.split(features, self.sequence_length, dim=2)
        
        #processing over elements of tuples
        # d_full = []
        # for y in y_chunk:
        '''input: (batch_size, channels, time_steps, freq_bins)'''
        y = self.conv1(x)
        y = self.maxpool1(y)
        y = self.dropout1(y)
        y = self.conv2(y)
        y = self.maxpool2(y)
        y = self.dropout2(y)
        y = self.conv3(y)
        y = self.maxpool3(y)
        y = self.dropout3(y)
        y = torch.transpose(y, 1, 2)
        #y = y.reshape(y.shape[0], y.shape[1], self.hidden_size)
       
        y = y.reshape(y.shape[0], y.shape[1], -1)
        #rnn layer forwarding
        y = pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
        z, hidden_state = self.rnn(y, hidden_state)
        z, _ = pad_packed_sequence(z, batch_first=True)
        z = torch.tanh(z)
        if self.bidirectional:
            z = z[:, :, z.shape[-1]//2:] * z[:, :, :z.shape[-1]//2] # Combining bidirectional GRU outputs

        #doa layer
        d = self.doa(z)
        doa = self.non_linear_doa(d)
        doa = pack_padded_sequence(doa, lengths, batch_first=True, enforce_sorted=False)
        doa, _ = pad_packed_sequence(doa, batch_first=True)
            # d_full.append(d)
        #print(d.shape)
        # doa = torch.cat(d_full, dim=1)
        return doa, hidden_state
        

    def min_max_norm(self,x):
        x = (x-x.min()) / (x.max()-x.min())  
        return x      

    def disturb(self, std):
        for p in self.parameters():
            noise_element = torch.zeros_like(p).normal_(0, std)
            p.data.add_(noise_element)


#conv2d layer implementation
class Conv_layer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3,3), dilation=(1, 1)):
        super(Conv_layer, self).__init__()
        self.cnn = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1,
                             dilation=dilation, padding=tuple(d*(k-1) // 2 for k, d in zip(kernel_size, dilation)))
        self.bn = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        x = self.bn(self.cnn(x))
        return F.relu(x)

    

