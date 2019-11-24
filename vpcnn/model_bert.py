"""
This module contains CNN network that directly eat bert embeddings, with similar function as model.py
"""
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch import autograd

class CNN_Embed(nn.Module):
    """
    This class directly takes word embedding level representation as it's input
    """
    def __init__(self,
                 class_num=361,
                 kernel_num=300,
                 kernel_sizes=[3,4,5],
                 embed_dim=3072,
                 dropout=0.5,
                 conv_init='default',
                 fc_init='default'):
        """

        :param class_num: int, number of different class labels
        :param kernel_num: int, number of kernel for each size, correspond to the output channel
        :param kernel_sizes: list of int, each element represent length of a kind of kernels
        :param embed_dim: int, dimensionality of a single word embedding, also used as kernel width
        :param dropout: float, probability of dropout an pattern for Dropout layer
        :param conv_init: string; initialize method for weight of convolutional layers, could be ortho, uniform or
        default
        :param fc_init: string; initialize method for output linear layer, could be ortho, normal or default
        """
        super(CNN_Embed, self).__init__()
        # initialize convolutional layers
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=1,
                                               out_channels=kernel_num,
                                               kernel_size=(kz, embed_dim)) for kz in kernel_sizes])
        for layer in self.convs1:
            if conv_init == 'ortho':
                init.orthogonal(layer.weight.data)
                layer.bias.data.zero_()
            elif conv_init == 'uniform':
                layer.weight.data.uniform_(-0.01, 0.01)
                layer.bias.data.zero_()
            elif conv_init == 'default':
                pass

        self.dropout = nn.Dropout(dropout)
        # output layer weight matrix
        # the input dimensionality of output layer is len(kernel_sizes)* kernel_num, which mean for each single kernel
        # it correspond to one dimension in output layer after pooling
        # the output dimensionality of output layer is equal to class_num
        # later a softmax will be applied to this layer
        self.fc1 = nn.Linear(len(kernel_sizes)*kernel_num,class_num)
        if fc_init == 'ortho':
            init.orthogonal(self.fc1.weight.data)
            self.fc1.bias.data.zero_()
        elif fc_init == 'normal':
            init.normal(self.fc1.weight.data)
            self.fc1.bias.data.zero_()
        elif fc_init == 'default':
            pass

    def forward(self, x):
        x = self.confidence(x)
        # logit = F.log_softmax(x)
        logit = F.softmax(x)
        return logit

    def confidence(self, x):
        x = autograd.Variable(x.float()).cuda()
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x,1)
        x = self.dropout(x)
        linear_out = self.fc1(x)
        return linear_out

