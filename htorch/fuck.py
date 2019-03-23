import torch as th
import nn
import numpy as np
import operations

inputs = th.rand((5, 3), requires_grad=True)
linear = nn.HyperbolicLinearLinear(3, 2, False)

linear(inputs)
