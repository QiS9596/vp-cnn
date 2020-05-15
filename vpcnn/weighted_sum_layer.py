import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules import Module

class Weighted_Sum(Module):
    """
    applies a weighted sum of input data
    we first split the input data into several trunks, and return weighted sum of them
    the sameple of input should be `(N, embed-dim*layers)`
    output: `(N, embed-dim)`
    in which in_features should be integer times of out_features
    attributes:
    weight: the trainable weight of the model of shape(in_features/out_features,1)
    """

    def __init__(self, embed_dim, layers):
        super(Weighted_Sum, self).__init__()
        self.embed_dim = embed_dim
        self.layers = layers
        self.weight_length = layers
        self.weight = Parameter(torch.Tensor(self.weight_length, 1))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        print(type(input))
        print(input.size())
        print(type(self.weight))
        print(self.weight.size())
        return torch.mv(input, self.weight)

    def __repr__(self):
        return self.__class__.__name__ + '  (' + str(self.embed_dim) + ' * ' + str(self.layers) +' -> '\
            + str(self.embed_dim) + ')'
