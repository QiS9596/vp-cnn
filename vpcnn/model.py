import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch import autograd

class CNN_Text(nn.Module):
    
    def __init__(self, 
                 class_num=None,
                 kernel_num=None,
                 kernel_sizes=None,
                 embed_num=None,
                 embed2_num=2,
                 embed_dim=None, 
                 dropout=0.5,
                 conv_init='default',
                 fc_init='default',
                 two_ch=False,
                 static=False,
                 vectors=None):
        super(CNN_Text,self).__init__()
        
        self.static = static
        self.two_ch = two_ch
        
        if (embed_num is None or 
            embed_dim is None or 
            class_num is None or 
            kernel_sizes is None or 
            kernel_num is None):
            raise TypeError("Required keyword argument not provided")

        V = embed_num
        V2 = embed2_num
        D = embed_dim
        C = class_num
        Ci = 2 if two_ch else 1
        Co = kernel_num
        Ks = kernel_sizes

        self.embed = nn.Embedding(V, D) #, padding_idx=1)
        self.embed2 = nn.Embedding(V2, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        if vectors is not None:
            self.embed.weight.data = vectors

        for layer in self.convs1:
            if conv_init == 'ortho':
                init.orthogonal(layer.weight.data)
                layer.bias.data.zero_()
            elif conv_init == 'uniform':
                layer.weight.data.uniform_(-0.01, 0.01)
                layer.bias.data.zero_()
            elif conv_init == 'default':
                # nothing to do, it's already initialized
                # (but still wanted to check that a valid
                # option was passed).
                pass 
            else:
                raise ValueError("Unknown initialization option")
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        if fc_init == 'ortho':
            init.orthogonal(self.fc1.weight.data)
            self.fc1.bias.data.zero_()
        elif fc_init == 'normal':
            init.normal(self.fc1.weight.data)
            self.fc1.weight.data.mul_(0.01)
            self.fc1.bias.data.zero_()
        elif fc_init == 'default':
            # nothing to do, it's already initialized
            # (but still wanted to check that a valid
            # option was passed).
            pass 
        else:
            raise ValueError("Unknown initialization option")
        # print(V, D, C, Ci, Co, Ks, self.convs1, self.fc1)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, y = None):
        x = self.confidence(x, y)
        logit = F.log_softmax(x) # (N,C)
        return logit

    def confidence(self, x, y):
        x = self.embed(x)  # (N,W,D)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        if self.two_ch:
            if y is not None:
                y = self.embed2(y)
                y = y.unsqueeze(1)
                x = torch.stack([x, y], dim=1)
            else:
                raise ValueError("two-channel model requires two inputs")

        if self.static:
            # default initialize to requires_grad=False
            x = autograd.Variable(x.data)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        # print([x_p.size() for x_p in x])

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N,len(Ks)*Co)
        linear_out = self.fc1(x)
        return linear_out

class SimpleLogistic(nn.Module):
    def __init__(self, args):
        super(SimpleLogistic, self).__init__()
        self.args = args
        self.input_size = self.args.class_num * 2
        self.output_size = self.args.class_num
        self.layer_num = self.args.layer_num
        self.layers = nn.ModuleList([nn.Linear(self.input_size, self.input_size) if x < self.layer_num - 1 else
                       nn.Linear(self.input_size, self.output_size) for x in range(self.layer_num)])

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x

class StackingNet(nn.Module):
    def __init__(self, args):
        super(StackingNet, self).__init__()
        self.args = args
        self.params = nn.ParameterList([nn.Parameter(torch.rand(1)) for i in range(2)])

    def forward(self, inputs):
        output = 0
        for index, input in enumerate(inputs):
            output += input * self.params[index].expand(input.size())
        output = F.log_softmax(output)
        return output
