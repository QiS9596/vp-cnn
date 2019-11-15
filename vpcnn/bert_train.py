import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import copy
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
    train_batchs = generate_batches(dataset=train, batch_size=batch_size, shuffle=False, drop_last=False,device=device)
    # begin training
    for epoch in range(1, epochs+1):
        # fit model on batch
        for batch in train_batchs:
            feature = batch['embed']
            target = batch['label']
            # step 1: set optimizer to zero grad
            optimizer.zero_grad()
            # step 2: make prediction
            logit = model(feature)
            # step 3: compute loss, here negative log likelihood is employed
            loss = F.nll_loss(input=logit, target=target)
            # step 4: use loss to produce gradient
            loss.backward()
            # step 5: update weights
            optimizer.step()
            # max norm constraint
            # Qi: the code is directly copy paste from original one, dont completely sure the process
            if max_norm > 0:
                if not no_always_norm:
                    for row in model.fc1.weight.data:
                        norm = row.norm() +1e-7
                        row.div_(norm).mul_(max_norm)
                else:
                    model.fc1.weight.data.renorm_(2,0,max_norm)
    print(loss)
    # TODO eval
#TODO predict,eval and ensemble train functions