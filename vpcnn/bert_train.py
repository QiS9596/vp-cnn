"""
The model provides training and evaluation method similar to train.py but compatible with bert cnn models
"""
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import copy
import numpy as np
from torch.utils.data import DataLoader
def generate_batches(dataset, batch_size, shuffle=False, drop_last=False, device='cuda'):
    """
    generator function wraps pytorch dataloader
    generate a sequence of dictionary, each one contains data for each batch
    :param dataset: dataset object
    :param batch_size: int; size of each mini batch
    :param shuffle: bool; if to random shuffle the dataset
    :param drop_last:
    :param device:
    :return: a generator which generates a sequence of batch dictionary
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            if name=='embed__':
                print("type of a batch")
                print(type(data_dict[name]))
                print("type of an dataobject")
                print(type(data_dict[name][0]))
                out_data_dict[name] = torch.FloatTensor(data_dict[name])
            else:
                out_data_dict[name] = data_dict[name]
            # if device == 'cuda':
                # out_data_dict[name] = data_dict[name].to('cuda')
        yield out_data_dict

def train(train, dev, model, optimizer='adam', use_cuda=True, lr=1e-3, l2=1e-6, epochs=25, batch_size=50,
          max_norm=3.0, no_always_norm=False):
    """
    training single bert cnn model
    :param train: training dataset object
    :param dev: development dataset object
    :param model: model object to be trained
    :param optimizer: str; for choosing optimizer
    :param use_cuda: bool; for using cuda or not
    :param lr: float; learning rate
    :param l2: l2 regularization of optimizer
    :param epochs: int; max number of epochs
    :param batch_size: int; batch size
    :param max_norm: float, l2 constraint for parameters
    :param no_always_norm: boolean; if true then do some black magic to norm of output layer
    :return: tuple of acc and copy of best model
    """
    device = 'cpu'
    if use_cuda:
        model.cuda()
        device = 'cuda'
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters, lr=lr, weight_decay=l2)
    elif optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95)
    steps = 0
    model.train()
    best_acc = 0
    best_model = None
    # train_batchs is a sqeuence of python dicts obtained based on the given dataset
    train_batchs_ = generate_batches(dataset=train, batch_size=batch_size, shuffle=False, drop_last=False,device=device)
    train_batchs = []
    for batch in train_batchs_:
        train_batchs.append(copy.deepcopy(batch))
    # begin training
    for epoch in range(1, epochs+1):
        # fit model on batch
        batch_idx = 0
        for batch in train_batchs:
            feature = batch['embed']
            target = batch['label']
            before = copy.deepcopy(list(model.parameters())[0])
            # step 1: set optimizer to zero grad
            optimizer.zero_grad()
            # step 2: make prediction
            logit = model(feature)
            target = autograd.Variable(target).cuda()
            # step 3: compute loss, here negative log likelihood is employed
            loss = F.cross_entropy(input=logit, target=target)
            # step 4: use loss to produce gradient
            loss.backward()
            # step 5: update weights
            optimizer.step()
            after = copy.deepcopy(list(model.parameters())[0])
            #print(torch.equal(before.data, after.data))
            batch_idx += 1
            # max norm constraint
            # Qi: the code is directly copy paste from original one, dont completely sure the process
            if max_norm > 0:
                if not no_always_norm:
                    for row in model.fc1.weight.data:
                        norm = row.norm() +1e-7
                        row.div_(norm).mul_(max_norm)
                else:
                    model.fc1.weight.data.renorm_(2,0,max_norm)
        acc = eval(dev, model, batch_size, use_cuda)
        if acc > best_acc:
            best_model = copy.deepcopy(model)
    model = best_model
    acc = eval(dev, model, batch_size, use_cuda)
    return acc, model


def eval(data_iter, model, batch_size, use_cuda=True):
    """
    evaluation method for single model
    :param data_iter: dataset object for evaluation
    :param model: model object to be evaluated
    :param batch_size: size of mini batch
    :param use_cuda: bool; if to use cuda
    :return: accuracy measured on the given dataset
    """
    model.eval()
    corrects, avg_loss = 0, 0
    batchs_ = generate_batches(dataset=data_iter, batch_size=batch_size, shuffle=False, drop_last=False)
    if use_cuda:
        model.cuda()
    # batchs = []
    # for batch in batchs_:
    #     batchs.append(copy.deepcopy(batch))
    for batch in batchs_:
        feature = batch['embed']
        target = batch['label']
        target = autograd.Variable(target).cuda()
        logit = model(feature)
        loss = F.cross_entropy(input=logit, target=target)
        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
    avg_loss /= len(data_iter)
    accuracy = corrects/len(data_iter) * 100.0
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       len(data_iter)))
    return accuracy

def ensemble_predict(batch_feature, batch_target, models):
    """
    This function takes a stack of bert cnn models and evaluate the ensemble on the given batch
    :param batch_feature: the features field, the word embedding per se
    :param batch_target: the training target
    :param models: a list of model, serves as an embedding
    :return:
    """
