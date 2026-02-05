def hutchinson(Hz_func, z_gen_func, num_iters):
    D = 0
    for _ in range(num_iters):
        z = z_gen_func()
        Hz = Hz_func(z)
        D += z * Hz
    return D / num_iters
