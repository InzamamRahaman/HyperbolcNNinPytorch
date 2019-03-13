import torch as th
import numpy as np
import random
from numpy import linalg as LA
from numpy import random as np_random

C = 1.0
PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0

def project_hyp_vecs(x, c=C):
    pass

# important utilities

def atanh(x):
    x = th.min(x, 1.0 - EPS)
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
    return th.sum(product, dim=1, keepdim=True)

def norm(x):
    return th.norm(x, dim=1, keepdim=True)


# basic operations

def add(u, v, c=C):
    v = v + EPS
    dot_uv = 2 * c * dot(u, v)
    norm_u_sq = c * dot(u, u)
    norm_v_sq = c * th.dot(v, v)
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
    second_term = 1/(tanh(np.sqrt(c) * t / 2)) * (np.sqrt(c) * norm(v)) * v
    res = add(x, second_term, c)
    return res

def exp_map_x(x, v, c=C):
    v = v + EPS
    norm_v = norm(v)
    second_term = (tanh(np.sqrt(c) * lambda_x(x, c) * norm_v / 2) / (np.sqrt(c) * norm_v)) * v
    res = add(x, second_term, c)
    return res

def log_map_x(x, v, c=C):
    


