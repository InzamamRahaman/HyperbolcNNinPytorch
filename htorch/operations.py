import torch as th
import numpy as np
import torch.nn as nn
import random
from numpy import linalg as LA
from numpy import random as np_random

C = 1.0
PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = th.tensor(15.0)


# important utilities

def atanh(x):
    x = th.min(x.float(), th.tensor(1.0 - EPS))
    numer = 1 + x
    denom = 1 - x
    inner = numer / denom
    inner = th.log(inner)
    res = inner / 2
    return res

def tanh(x):
    x = th.max(x, -MAX_TANH_ARG)
    x = th.min(x, MAX_TANH_ARG)
    return th.tanh(x)

def dot(x, y):
    product = x * y
    return th.sum(product, dim=0, keepdim=True)

def norm(x):
    return th.norm(x, dim=0, keepdim=True)


def project_hyp_vecs(x, c=C):
    normed = norm(x)
    #print(normed)
    #print(1.0 - PROJ_EPS / np.sqrt(c))
    bound = 0.99999 #1.0 - PROJ_EPS / np.sqrt(c)
    desired = th.clamp(normed, 0, bound)
    y = x * (desired / (EPS + normed))
    return y


# basic operations

def add(u, v, c=C):
    v = v + EPS
    dot_uv = 2 * c * dot(u, v)
    norm_u_sq = c * dot(u, u)
    norm_v_sq = c * dot(v, v)
    denom = 1 + dot_uv + norm_v_sq * norm_u_sq
    res =(1. + dot_uv + norm_v_sq) / denom * u + (1. - norm_u_sq) / denom * v
    res = project_hyp_vecs(res, c)
    return res

def distance(u, v, c=C):
    sqrt_c = np.sqrt(c)
    m = add(-u, v, c,) + EPS
    atanh_x = sqrt_c * norm(m)
    dist = 2 / sqrt_c * atanh(atanh_x)
    return dist

def distance_squared(u, v, c=C):
    dist = distance(u, v ,c)
    return dist * dist

def scalar_mult(r, v, c=C):
    v = v + EPS
    norm_v = norm(v)
    nomin = tanh(r * atanh(np.sqrt(c) * norm_v))
    result = nomin / (np.sqrt(c) * norm_v) * v
    return project_hyp_vecs(result, c)

def lambda_x(x, c=C):
    return 2 / (1 - c * dot(x, x))

def unit_speed_geo(x, v, t, c=C):
    a = tanh(th.tensor(np.sqrt(c) * t / 2))
    b = np.sqrt(c) * norm(v)
    second_term = a / b * v
    res = add(x, second_term, c)
    #print('Unit speed ', res)
    return res

def exp_map_x(x, v, c=C):
    v = v + EPS
    norm_v = norm(v)
    sqrt_c = th.tensor(np.sqrt(c))
    inner = sqrt_c * lambda_x(x, c) * norm_v
    inner = inner / 2
    inner = inner[0]
    inner = inner.float()
    a = tanh(inner)
    b = sqrt_c * norm_v
    second_term = a / b * v
    res = add(x, second_term, c)
    return res

def log_map_x(x, v, c=C):
    diff = add(-x, v, c) + EPS
    #norm_diff = norm(diff)
    lam = lambda_x(x, c).float()

    sqrt_c = th.tensor(np.sqrt(c)).float()

    f1 = 2.0 / (sqrt_c * lam)
    f2 = atanh(sqrt_c * norm(diff)) / norm(diff)
    f3 = diff
    return (f1 * f2 * f3)


def exp_map_zero(v, c=C):
    v = v + EPS # Perturbe v to avoid dealing with v = 0
    norm_v = norm(v)
    result = tanh(th.tensor(np.sqrt(c) * norm_v)) / (np.sqrt(c) * norm_v) * v
    return project_hyp_vecs(result, c)

def log_map_zero(y, c=C):
    diff = y + EPS
    norm_diff = norm(diff)
    return 1. / np.sqrt(c) * atanh(np.sqrt(c) * norm_diff) / norm_diff * diff


def mat_mult(M, x, c=C):
    x = x + EPS
    #print('M is ', M)
    #print('x is ', x)
    Mx = x @ M
    #th.matmul(M, x)
        #th.matmul(M, x) + EPS
    MX_norm = norm(Mx)
    x_norm = norm(x)

    #print(Mx)

    print('Mx norm2 ', MX_norm)
    print('x norm2', x_norm)

    a1 = MX_norm / x_norm
    a2 = atanh(np.sqrt(c) * x_norm)

    a = tanh(a1 * a2)
    b = 1. / np.sqrt(c)
    c = a / MX_norm

    # print(a)
    # print(b)
    # print(c)
    # print(Mx)

    result = b * c * Mx
    #print(result)
    return project_hyp_vecs(result, c)



# x is hyperbolic, u is Euclidean. Computes diag(u) \otimes x.
def mob_pointwise_prod(x, u, c=C):
    x = x + EPS
    Mx = x * u + EPS
    MX_norm = norm(Mx)
    x_norm = norm(x)
    result = 1. / np.sqrt(c) * tanh(MX_norm / x_norm * atanh(np.sqrt(c) * x_norm)) / MX_norm * Mx
    return project_hyp_vecs(result, c)


#########################
def riemannian_gradient_c(u, c=C):
    return ((1. - c * dot(u,u)) ** 2) / 4.0


#########################
def eucl_non_lin(eucl_h, non_lin):
    if non_lin == 'id':
        return eucl_h
    elif non_lin == 'relu':
        return nn.relu(eucl_h)
    elif non_lin == 'tanh':
        return tanh(eucl_h)
    elif non_lin == 'sigmoid':
        return nn.sigmoid(eucl_h)
    return eucl_h

# Applies a non linearity sigma to a hyperbolic h using: exp_0(sigma(log_0(h)))
def tf_hyp_non_lin(hyp_h, non_lin, hyp_output, c=C):
    if non_lin == 'id':
        if hyp_output:
            return hyp_h
        else:
            return log_map_zero(hyp_h, c)

    eucl_h = eucl_non_lin(log_map_zero(hyp_h, c), non_lin)

    if hyp_output:
        return exp_map_zero(eucl_h, c)
    else:
        return eucl_h

