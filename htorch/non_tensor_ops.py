import logging
import numpy as np
from numpy import linalg as LA
from numpy import random as np_random
import os
import random

PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0



def mob_add(u, v, c):
    numerator = (1.0 + 2.0 * c * np.dot(u,v) + c * LA.norm(v)**2) * u + (1.0 - c * LA.norm(u)**2) * v
    denominator = 1.0 + 2.0 * c * np.dot(u,v) + c**2 * LA.norm(v)**2 * LA.norm(u)**2
    return numerator / denominator




def poinc_dist_sq(u, v, c):
    sqrt_c = np.sqrt(c)
    atanh_x = sqrt_c * LA.norm(mob_add(-u, v, c))
    dist_poincare = 2.0 / sqrt_c * np.arctanh(atanh_x)
    return dist_poincare ** 2



def euclid_dist_sq(u, v):
    return LA.norm(u - v)



def mob_scalar_mul(r, v, c):
    norm_v = LA.norm(v)
    nomin = np.tanh(r * np.arctanh(np.sqrt(c) * norm_v)) * v
    return nomin / (np.sqrt(c) * norm_v)



def n_lambda_x(x, c):
    return 2. / (1 - c * LA.norm(x)**2)


#########################
def unit_speed_geo(x, v, t, c):
    second_term = np.tanh(np.sqrt(c) * t / 2) / (np.sqrt(c) * LA.norm(v)) * v
    return mob_add(x, second_term, c)

def n_exp_map_x(x, v, c):
    second_term = np.tanh(np.sqrt(c) * n_lambda_x(x, c) * LA.norm(v) / 2) / (np.sqrt(c) * LA.norm(v)) * v
    return mob_add(x, second_term, c)

def n_log_map_x(x, y, c):
    diff = mob_add(-x, y, c)
    lam = n_lambda_x(x, c)
    return 2. / (np.sqrt(c) * lam) * np.arctanh(np.sqrt(c) * LA.norm(diff)) / (LA.norm(diff)) * diff




#########################
def mob_mat_mul(M, x, c):
    Mx = M.dot(x)
    MX_norm = LA.norm(Mx)
    x_norm = LA.norm(x)
    return 1. / np.sqrt(c) * np.tanh(MX_norm / x_norm * np.arctanh(np.sqrt(c) * x_norm)) / MX_norm * Mx


# x is hyperbolic, u is Euclidean. Computes diag(u) \otimes x.

