import nn
import rsgd



class LinearModel(object):
    def __init__(self, n_in, n_out, bias=False):
        self.esimator = nn.