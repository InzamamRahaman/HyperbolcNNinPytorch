import torch as th
import numpy as np
import random
from operations import *
from non_tensor_ops import *
import torch.random as th_random

def random_tensor(size=(1, 10)):
    data = np.random.uniform(low=-0.01, high=0.01, size=size)
    tensor = th.tensor(data)
    return tensor

def addition_left_cancelation_test():
    a = None
    b = None
    res = None
    for i in range(0, 10000):
        a = random_tensor((10, 10))
        b = random_tensor((10, 10))
        c = random.random()
        res = add(-a, add(a, b, c=c), c=c)
        diff = th.sum(th.abs(b - res))
        if diff > 1e-10:
            print('Invalid :/')
            print('b: ')
            print(b)
            print('res: ')
            print(res)
            exit()
    print('Test left cancelation passed!')

def addition_zero_b_test():
    for i in range(0, 10000):
        a = th.tensor(np.zeros(10))
        b = random_tensor(10)

        res = add(a, b, c=1.0)
        diff = th.sum(th.abs(res - b))
        if diff > 1e-10:
            print('Test 5 FAILED at trial %d :/' % i)
            print('res: ')
            print(res)
            print('b: ')
            print(b)
            exit()

    print('Test 0 + b passed!')


def addition_negative_test():
    for i in range(0, 10000):
        a = random_tensor((10, 10))
        b = random_tensor((10, 10))

        c = random.random()
        res1 = add(-a, -b, c)
        res2 = -add(a, b, c)
        diff = th.sum(th.abs(res1 - res2))

        if diff > 1e-10:
            print('Test 6 FAILED at trial %d :/' % i)
            print('res1: ')
            print(res1)
            print('res2: ')
            print(res2)
            exit()

    print('Test a+b = -a + -b passed!')

def addition_infinity_test():
    for i in range(0, 10000):
        a = random_tensor(10)#np.random.uniform(low=-0.01, high=0.01, size=10)
        b = random_tensor(10)#np.random.uniform(low=-0.01, high=0.01, size=10)

        a = a / norm(a)

        res = add(a, b, c=1.0)
        diff = norm(a - res)
        diff = diff[0]
        diff = float(diff)

        if diff > 1e-10:
            print('Test 7a FAILED at trial %d :/' % i)
            print('res: ')
            print(res)
            print('diff: ')
            print(diff)
            exit()

        res = add(b, a, c=1.0)
        diff = th.abs(1 - norm(res))
        diff = diff[0]
        diff = float(diff)
        if diff > 1e-10:
            print('Test 7b FAILED at trial %d :/' % i)
            print('res: ')
            print(res)
            print('diff: ')
            print(diff)
            exit()


    print('Test mob add at infinity passed!')

def mobius_unit_speed_geo_test():

    emb_dim = 5
    c = 0.76

    x = (np_random.uniform(-.5, .5, (emb_dim)).astype(np.float64))
    v = np_random.uniform(-.5, .5, (emb_dim)).astype(np.float64)

    x = th.tensor(x)
    v = th.tensor(v)

    x = x * 0.54321 / norm(x)
    v = v / (norm(v) * lambda_x(x, c))

    t = 1e-6

    a = unit_speed_geo(x, v, 0, c)
    b =  unit_speed_geo(x, v, t, c)
    print('a is ', a)
    print('b is ', b)
    d = (-a + b) / t
    print(d)
    print(v)
    diff = norm(d - v)
    print(diff)
    assert  diff < 1e-5
    print('Test unit speed geodesic passed!')


def exp_map_test():
    emb_dim = 5
    c = 0.76

    x = np_random.uniform(-.5, .5, (emb_dim)).astype(np.float32)
    v = np_random.uniform(-.5, .5, (emb_dim)).astype(np.float32)
    x = th.tensor(x)
    v = th.tensor(v)

    x = x * 0.54321 / norm(x)

    inner = exp_map_x(x, v, c)
    inner1 = log_map_x(x, inner, c)
    inner2 = inner1 - v
    normed_inner2 = norm(inner2)
    #print(normed_inner2)
    assert norm(inner2) < 1e-5

    r = np_random.random() * 10
    assert norm(exp_map_x(0, r * log_map_x(0, x, c), c) - scalar_mult(r, x, c)) < 1e-8

    #print('Test exp map passed!')

def mobius_mat_mul_test():
    M = np.random.rand(5, 8)
    x = np.random.rand(8)
    x = x / LA.norm(x) * 0.789
    c = 1.0
    assert LA.norm(mob_mat_mul(M, x, c) - exp_map_x(0, M.dot(log_map_x(0, x, c)), c)) < 1e-5

    for i in range(10):
        c = random.random()
        assert LA.norm(mob_mat_mul(M, x, c) - exp_map_x(0, M.dot(log_map_x(0, x, c)), c)) < 1e-5
        assert LA.norm(mob_mat_mul(M, x, 1e-10) - M.dot(x)) < 1e-5

        M_prime = np.random.rand(7, 5)
        assert LA.norm(mob_mat_mul(M_prime.dot(M), x, c) - mob_mat_mul(M_prime, mob_mat_mul(M,x,c),c)) < 1e-5

        r = random.random() * 10
        assert LA.norm(mob_mat_mul(r * M, x, c) - mob_scalar_mul(r, mob_mat_mul(M,x,c),c)) < 1e-5

        assert LA.norm(mob_mat_mul(M, x, c) / LA.norm(mob_mat_mul(M, x, c)) - M.dot(x) / LA.norm(M.dot(x))) < 1e-5

    print('Mobius mat mul test passed!')

def get_numpy_values(v1, v2, v3, r, M):
    res = {
        'distance': poinc_dist_sq(v1, v2, 1),
        'add': mob_add(v1, v2, 1),
        'mult': mob_scalar_mul(r, v1, 1),
        'diff': mob_add(-v1, v2, 1),
        'exp_map_x': n_exp_map_x(v1, v2, 1),
        'log_map_x': n_log_map_x(v1, v2, 1),
        'lambda': n_lambda_x(v1, 1),
        'matmul1': mob_mat_mul(M, v1, 1),
        'matmul2': mob_mat_mul(M, v2, 1),
        'matmul3': mob_mat_mul(M, v3, 1),
    }
    return res

def get_th_values(v1, v2, v3, r, M):
    res = {
        'distance': distance_squared(v1, v2, 1),
        'add': add(v1, v2, 1),
        'mult': scalar_mult(r, v1, 1),
        'diff': add(-v1, v2, 1),
        'exp_map_x': exp_map_x(v1, v2, 1),
        'log_map_x': log_map_x(v1, v2, 1),
        'lambda': lambda_x(v1, 1),
        'matmul1': mat_mult(M, v1, 1),
        'matmul2': mat_mult(M, v2, 1),
        #'matmul3': mat_mult(M, v3, 1),
    }
    return res


def run():
    c = 1
    emb_dim = 20
    bs = 1
    r = np_random.random() * 10
    v1 = np_random.uniform(-0.5, 0.5, emb_dim).astype(np.float32)
    v2 = np_random.uniform(-0.5, 0.5, emb_dim).astype(np.float32)
    v3 = np_random.uniform(-0.5, 0.5, (10, emb_dim)).astype(np.float32)
    v1 = v1 * 0.59999 / LA.norm(v1)
    v2 = v2 * 0.99 / LA.norm(v2)
    v3 = np.stack((v1, v2))
    M = np.random.rand(emb_dim, 5).astype(np.float32)
    numpy_vers = get_numpy_values(v1, v2, v3, r, M)
    #print(numpy_vers)

    v1 = th.tensor(v1)
    v2 = th.tensor(v2)
    v3 = th.tensor(v3)
    M = th.tensor(M)
    th_vers = get_th_values(v1, v2, v3, r, M)
    #print(th_vers)

    print('Printing Distances!!!!')
    for operation_name, val1 in numpy_vers.items():
        if 'matmul' in operation_name:
            print(val1)
            print(val2)
        val2 = th_vers[operation_name]
        diff = np.sum(np.abs(val1 - val2.numpy()))
        print('Component wise differences for ', operation_name, ' is ', diff)
        




run()