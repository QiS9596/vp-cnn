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
    :param dataset:
    :param batch_size:
    :param shuffle:
    :param drop_last:
    :param device:
    :return:
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

    :param train:
    :param dev:
    :param model:
    :param optimizer:
    :param use_cuda:
    :param lr:
    :param l2: l2 regularization of optimizer
    :param epochs: int; max number of epochs
    :param batch_size: int; batch size
    :param max_norm: float, l2 constraint for parameters
    :param no_always_norm: boolean; if true then do some black magic to norm of output layer
    :return:
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
        eval(dev, model, batch_size, use_cuda)

    # TODO eval
#TODO predict,eval and ensemble train functions
def eval(data_iter, model, batch_size, use_cuda=True):
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
    accuracy = corrects/len(data_iter)
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       len(data_iter)))
    return accuracy
