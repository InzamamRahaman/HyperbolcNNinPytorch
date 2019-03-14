from non_tensor_ops import  *
#########################

####################################################################################################
####################################################################################################
####################################### Unit tests #################################################
####################################################################################################
####################################################################################################
def mobius_addition_left_cancelation_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=1)
        b = np.random.uniform(low=-0.01, high=0.01, size=1)

        c = random.random()
        res = mob_add(-a, mob_add(a, b, c=c), c=c)
        diff = np.sum(np.abs(b - res))
        if diff > 1e-10:
            print('Invalid :/')
            print('b: ')
            print(b)
            print('res: ')
            print(res)
            exit()

    print('Test left cancelation passed!')


def mobius_addition_cancel_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=10)

        res = mob_add(-a, a, c=random.random())
        diff = np.sum(np.abs(res))
        if diff > 1e-10:
            print('Invalid :/')
            print('res: ')
            print(res)
            exit()

    print('Test -a + a passed!')


def mobius_addition_2a_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=10)

        res1 = mob_add(a, a, c=1.0)
        res2 = 2.0 / (1.0 + np.dot(a, a)) * a
        diff = np.sum(np.abs(res1 - res2))
        if diff > 1e-10:
            print('Invalid :/')
            print('res1: ')
            print(res1)
            print('res2: ')
            print(res2)
            exit()

    print('Test a+a passed!')


def mobius_addition_poinc_dist_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=0.0, high=0.01, size=10)
        b = np.random.uniform(low=0.0, high=0.01, size=10)

        res1 = poinc_dist_sq(a, b, c=1.0)
        res2 = 2 * np.arctanh(np.linalg.norm(mob_add(-a, b, c=1.0)))
        diff = np.sum(np.abs(res1 - res2**2))
        if diff > 1e-10:
            print('Test 4 FAILED at trial %d :/' % i)
            print('res1: ')
            print(res1)
            print('res2: ')
            print(res2)
            print('2xres2: ')
            print(2 * res2)
            print('2xres2 - res1')
            print(2 * res2 - res1)
            return

    print('Test poinc dist - mobius passed!')


def mobius_addition_zero_b_test():
    for i in range(0, 10000):
        a = np.zeros(10)
        b = np.random.uniform(low=-0.01, high=0.01, size=10)

        res = mob_add(a, b, c=1.0)
        diff = np.sum(np.abs(res - b))
        if diff > 1e-10:
            print('Test 5 FAILED at trial %d :/' % i)
            print('res: ')
            print(res)
            print('b: ')
            print(b)
            exit()

    print('Test 0 + b passed!')


def mobius_addition_negative_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=10)
        b = np.random.uniform(low=-0.01, high=0.01, size=10)

        c = random.random()
        res1 = mob_add(-a, -b, c)
        res2 = -mob_add(a, b, c)
        diff = np.sum(np.abs(res1 - res2))

        if diff > 1e-10:
            print('Test 6 FAILED at trial %d :/' % i)
            print('res1: ')
            print(res1)
            print('res2: ')
            print(res2)
            exit()

    print('Test a+b = -a + -b passed!')


def mobius_addition_infinity_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=10)
        b = np.random.uniform(low=-0.01, high=0.01, size=10)

        a = a / LA.norm(a)

        res = mob_add(a, b, c=1.0)
        diff = LA.norm(a - res)

        if diff > 1e-10:
            print('Test 7 FAILED at trial %d :/' % i)
            print('res: ')
            print(res)
            print('diff: ')
            print(diff)
            exit()

        res = mob_add(b, a, c=1.0)
        diff = np.abs(1 - LA.norm(res))

        if diff > 1e-10:
            print('Test 7 FAILED at trial %d :/' % i)
            print('res: ')
            print(res)
            print('diff: ')
            print(diff)
            exit()


    print('Test mob add at infinity passed!')

def mobius_unit_speed_geo_test():

    emb_dim = 5
    c = 0.76

    x = np_random.uniform(-.5, .5, (emb_dim)).astype(np.float64)
    v = np_random.uniform(-.5, .5, (emb_dim)).astype(np.float64)

    x = x * 0.54321 / LA.norm(x)
    v = v / (LA.norm(v) * n_lambda_x(x, c))

    t = 1e-6
    d = (- unit_speed_geo(x, v, 0, c) + unit_speed_geo(x, v, t, c)) / t

    assert LA.norm(d - v) < 1e-5
    print('Test unit speed geodesic passed!')


def mobius_exp_map_test():
    emb_dim = 5
    c = 0.76

    x = np_random.uniform(-.5, .5, (emb_dim)).astype(np.float64)
    v = np_random.uniform(-.5, .5, (emb_dim)).astype(np.float64)

    x = x * 0.54321 / LA.norm(x)

    assert LA.norm(n_log_map_x(x, n_exp_map_x(x, v, c), c) - v ) < 1e-5

    r = np_random.random() * 10
    assert LA.norm(n_exp_map_x(0, r * n_log_map_x(0, x, c), c) - mob_scalar_mul(r, x, c)) < 1e-8

    print('Test exp map passed!')

def mobius_mat_mul_test():
    M = np.random.rand(5, 8)
    x = np.random.rand(8)
    x = x / LA.norm(x) * 0.789
    c = 1.0
    assert LA.norm(mob_mat_mul(M, x, c) - n_exp_map_x(0, M.dot(n_log_map_x(0, x, c)), c)) < 1e-5

    for i in range(10):
        c = random.random()
        assert LA.norm(mob_mat_mul(M, x, c) - n_exp_map_x(0, M.dot(n_log_map_x(0, x, c)), c)) < 1e-5
        assert LA.norm(mob_mat_mul(M, x, 1e-10) - M.dot(x)) < 1e-5

        M_prime = np.random.rand(7, 5)
        assert LA.norm(mob_mat_mul(M_prime.dot(M), x, c) - mob_mat_mul(M_prime, mob_mat_mul(M,x,c),c)) < 1e-5

        r = random.random() * 10
        assert LA.norm(mob_mat_mul(r * M, x, c) - mob_scalar_mul(r, mob_mat_mul(M,x,c),c)) < 1e-5

        assert LA.norm(mob_mat_mul(M, x, c) / LA.norm(mob_mat_mul(M, x, c)) - M.dot(x) / LA.norm(M.dot(x))) < 1e-5

    print('Mobius mat mul test passed!')


def run_all_unit_tests():
    mobius_unit_speed_geo_test()
    mobius_mat_mul_test()
    mobius_exp_map_test()
    mobius_addition_left_cancelation_test()
    mobius_addition_cancel_test()
    mobius_addition_2a_test()
    mobius_addition_poinc_dist_test()
    mobius_addition_zero_b_test()
    mobius_addition_negative_test()
    mobius_addition_infinity_test()

run_all_unit_tests()


####################################################################################################




