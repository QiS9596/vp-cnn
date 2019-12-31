"""
This module contains CNN network that directly eat bert embeddings, with similar function as model.py
"""
import copy
from abc import ABC, abstractmethod

import bert_train
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.nn import init


class CNN_Embed(nn.Module):
    """
    This class directly takes word embedding level representation as it's input
    """

    def __init__(self,
                 class_num=361,
                 kernel_num=300,
                 kernel_sizes=[3, 4, 5],
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
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, class_num)
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
        x = torch.cat(x, 1)
        x = self.dropout(x)
        linear_out = self.fc1(x)
        return linear_out


class BaseAutoEncoderDecoder(nn.Module, ABC):
    """
    Base class of autoencoders, should provide forward, encode, decode and training method
    """

    def __init__(self, layers=[3072, 1500, 800, 399]):
        super(BaseAutoEncoderDecoder, self).__init__()
        self.neuron_num = layers

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass

    @staticmethod
    def pre_train(mdl, train, optimizer='adadelta', lr=1e-4, batch_size=50, use_cuda=True):
        """
        Train the model
        :param mdl: the autoencoder decoder to be trained
        :param train: training data attributes
        :param optimizer: optimizer to be used
        :param lr: learning rate
        :param batch_size: batch size
        :param use_cuda: weather to use cuda
        :return: a deep copy of model
        """
        pass


class AutoEncoderDecoder(BaseAutoEncoderDecoder):
    """
    A simple autoencoder-decoder built with fully-connected feed-forward network
    """

    def __init__(self, layers=[3072, 1500, 800, 300]):
        super(AutoEncoderDecoder, self).__init__()
        _layers = []
        for i in range(len(layers) - 1):
            _layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
            _layers.append(nn.ReLU(True))
        self.encoder = nn.Sequential(*_layers)
        _layers = []
        layers.reverse()
        for i in range(len(layers) - 1):
            _layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
            _layers.append(nn.ReLU(True))
        self.decoder = nn.Sequential(*_layers)

    def encode(self, x):
        x = autograd.Variable(x.float()).cuda()
        return self.encoder(x)

    def decode(self, x):
        return self.decode(x)

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)

    @staticmethod
    def pre_train(mdl, train, optimizer='adadelta', lr=1e-4, batch_size=50, use_cuda=True, epochs=50):
        """
        Definition of parameters see super class
        Train the auto-encoder-decoder in a simple manner
        """
        if use_cuda:
            mdl.cuda()
        if optimizer == 'adadelta':
            _optimizer = torch.optim.Adadelta(mdl.parameters(), rho=0.95)
        elif optimizer == 'sgd':
            _optimizer = torch.optim.SGD(mdl.parameters(), lr=lr)
        elif optimizer == 'adam':
            _optimizer = torch.optim.Adam(mdl.parameters(), lr=lr)
        mdl.train()
        best_loss = 1000
        best_mdl = None

        train_batches_ = bert_train.generate_batches(dataset=train, batch_size=batch_size, shuffle=False,
                                                     drop_last=False)
        train_batches = []
        for batch in train_batches_:
            train_batches.append(copy.deepcopy(batch))
        for epoch in range(epochs + 1):
            # fit model on batch
            pass
            # TODO: train autoencoder decoder


class CNN_shirnk_dim(nn.Module):
    def __init__(self,
                 class_num=361,
                 kernel_num=300,
                 kernel_sizes=[3, 4, 5],
                 embed_dim=3072,
                 dropout=0.5,
                 conv_init='default',
                 fc_init='default',
                 shrinked_dim=300,
                 shrink_method='s_encoder'):
        self.cnn_model = CNN_Embed(class_num=class_num,
                                   kernel_num=kernel_num,
                                   kernel_sizes=kernel_sizes,
                                   embed_dim=shrinked_dim,
                                   dropout=dropout,
                                   conv_init=conv_init,
                                   fc_init=fc_init
                                   )
        if embed_dim == 3072:
            auto_encoder_layers = [3072, 1500, 800, 300]
        elif embed_dim == 768:
            auto_encoder_layers = [768, 500, 300]
        if shrink_method == 's_encoder':
            self.auto_encoder = AutoEncoderDecoder(auto_encoder_layers)
