"""
This module contains CNN network that directly eat bert embeddings, with similar function as model.py
"""
import copy
from abc import ABC, abstractmethod
import vp_dataset_bert
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
    def pre_train(mdl, train, optimizer='adadelta', lr=1e-4, batch_size=50, early_stop_loss=1e-2, use_cuda=True,
                  epochs=50):
        """
        Train the model
        :param mdl: the autoencoder decoder to be trained
        :param train: vp_dataset_bert.VPDataset_bert_embedding: training data
        :param optimizer: optimizer to be used
        :param lr: learning rate
        :param batch_size: batch size
        :param early_stop_loss: loss for early stopping, the value is based on mean square loss
        :param use_cuda: weather to use cuda
        :return: tuple of loss and a deep copy of trained model
        """
        return None, None


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
    def pre_train(mdl, train, optimizer='adadelta', lr=1e-4, batch_size=50, early_stop_loss=1e-2, use_cuda=True,
                  epochs=50):
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
        else:
            _optimizer = torch.optim.Adam(mdl.parameters(), lr=lr)
        mdl.train()
        # best_loss = 1000
        # best_mdl = None

        train_batches_ = bert_train.generate_batches(dataset=train, batch_size=batch_size, shuffle=False,
                                                     drop_last=False)
        train_batches = []
        for batch in train_batches_:
            train_batches.append(copy.deepcopy(batch))
        for epoch in range(epochs + 1):
            # fit model on batch
            batch_idx = 0
            avg_loss = 0
            for batch in train_batches:
                feature = batch['embed']
                target = batch['embed']
                # step 1: set optimizer to zero grad
                _optimizer.zero_grad()
                # step 2: make prediction
                output = mdl(feature)
                target = autograd.Variable(target).cuda()
                # step 3: compute loss, using mse
                loss = F.mse_loss(input=output, target=target)
                # step 4: use loss to produce gradient
                loss.backward()
                # step 5: update weights
                _optimizer.step()
                batch_idx += 1
                avg_loss += loss
            avg_loss /= len(train_batches)
            mdl_ = copy.deepcopy(mdl)
            if avg_loss < early_stop_loss:
                break
        return avg_loss, mdl_


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
            auto_encoder_layers = [3072, 1500, 800, shrinked_dim]
        elif embed_dim == 768:
            auto_encoder_layers = [768, 500, shrinked_dim]
        if shrink_method == 's_encoder':
            self.auto_encoder = AutoEncoderDecoder(auto_encoder_layers)

    def pre_train(self, train, optimizer='adadelta', lr=1e-4, batch_size=50, early_stop_loss=1e-2, use_cuda=True,
                  epochs=50):
        """
        Pre-train the autoencoder part for dimensionality reduction on embeddings.
        :param train: here refers to the vp_dataset_bert.VPDataset_bert_embedding object
        parameter definition see BaseAutoEncoderDecoder.pre_train
        """
        # get AutoEncoderPretrainDataset object from the CNN dataset
        train_ = vp_dataset_bert.AutoEncoderPretrainDataset.from_VPDataset_bert_embedding(train)
        loss, self.auto_encoder = BaseAutoEncoderDecoder.pre_train(self.auto_encoder,
                                                                   train=train_,
                                                                   optimizer=optimizer,
                                                                   lr=lr,
                                                                   batch_size=batch_size,
                                                                   early_stop_loss=early_stop_loss,
                                                                   use_cuda=use_cuda,
                                                                   epochs=epochs)
        print('Pre-training of auto-encoder ends, with loss of ' + str(loss))

    def forward(self, x):
        """
        Forward and predict output
        the function first use the autoencoder to reduce the dimensionality of each word vector to shrinked_dim,
        then feed the shirnked embedding to CNN classifier
        :param x: input sentence embedding
        :return: predicted class (softmax output to meet one hot encoding)
        """
        x = self.auto_encoder.encode(x)
        y = self.cnn_model(x)
        return y

    @staticmethod
    def mdl_train(train, dev, model, optimizer='adam', use_cuda=True, lr=1e-3, l2=1e-6, epochs=25, batch_size=50,
                  max_norm=3.0, no_always_norm=False):
        """
        Training method for CNN_shirnk_dim
        :param train: training dataset object
        :param dev: development dataset object
        :param model: model object to be trained
        :param optimizer: str; for choosing optimizer
        :param use_cuda: bool; for using cuda or not
        :param lr: float; learning rate
        :param l2: l2 regularization of optimizer
        :param epochs: int; max number of epochs
        :param batch_size: int; batch size
        :param max_norm: float; l2 constraint for parameters
        :param no_always_norm: boolean
        :return: tuple of acc and copy of best model
        """

        if use_cuda:
            model.cuda()
        if optimizer == 'adam':
            optimizer_ = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
        elif optimizer == 'sgd':
            optimizer_ = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2)
        elif optimizer == 'adadelta':
            optimizer_ = torch.optim.Adadelta(model.prameters(), rho=0.95)
        # set trainable
        model.train()
        model.auto_encoder.train(mode=False)
        model.cnn_model.train()
        # initialize some placeholder for best model
        best_acc = 0
        best_model = None
        # generate batchs
        train_batchs_ = bert_train.generate_batches(dataset=train, batch_size=batch_size, shuffle=False, drop_last=False)
        train_batchs = []
        for batch in train_batchs_:
            train_batchs.append(copy.deepcopy(batch))
        # begin training
        for epoch in range(1, epochs+1):
            batch_idx = 0
            for batch in train_batchs:
                feature = batch['embed']
                target = batch['label']

                # step 1: set optimizer to zero grad
                optimizer_.zero_grad()
                # step 2: make prediction
                logit = model(feature)
                target = autograd.Variable(target).cuda()
                # step 3: compute loss, cross entropy is used
                loss = F.cross_entropy(input=logit, target=target)
                # step 4: use loss to produce gradient
                loss.backward()
                # step 4: update weights
                optimizer_.step()

                batch_idx += 1

                if max_norm > 0:
                    if not no_always_norm:
                        for row in model.cnn_model.fc1.weight.data:
                            norm = row.norm() + 1e-7
                            row.div_(norm).mul_(max_norm)
                    else:
                        model.cnn_model.fc1.weight.data.renorm_(2, 0, max_norm)
            acc = bert_train.eval(dev, model, batch_size, use_cuda)
            if acc > best_acc:
                best_model = copy.deepcopy(model)
        # end training
        model = best_model
        acc = bert_train.eval(dev, model, batch_size, use_cuda)
        return acc, model

