import torch as th
import operations
import torch.nn as nn

class HyperbolicLinearLinear(nn.Module):
    def __init__(self, ninputs, noutputs, bias=False):
        super(HyperbolicLinearLinear, self).__init__()
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.weights = th.rand((ninputs, noutputs))
        self.W = nn.Parameter(self.weights, requires_grad=True)
        self.bias = False
        if bias:
            self.bias = th.rand(1)
            self.b = nn.Parameter(self.bias, requires_grad=True)

    def forward(self, input):
        res = operations.mat_mult(self.W, input)
        if self.bias:
            bias = th.ones_like(res) * self.bias
            res = res + bias
        return res


