import torch
from diagonal_estimator import hutchinson

class SophiaOptimizer:
    def __init__(self, betas=(0.9, 0.999), eps=1e-15,
                 diagonal_update_interval=10,
                 num_update_iter=1):

        self.betas = betas
        self.eps = eps
        self.diagonal_update_interval = diagonal_update_interval
        self.num_update_iter = num_update_iter
        self.iter = 0
        self.total_D_iter = 0
        self.m = 0
        self.D_smoothed = 0
        self.D_est = 0
        self.diagonal_initialized = False

    def get_update(self, g, JTJv_func, z_gen_func):
        self.iter += 1

        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * g

        if self.iter % self.diagonal_update_interval == 0 or not self.diagonal_initialized:
            self.diagonal_initialized = True
            self.update_diagonal(JTJv_func, z_gen_func)

        v = self.D_est
        m_hat = self.m / (1 - self.betas[0] ** self.iter)
        s = -m_hat / (v + self.eps)


        return s

    def update_diagonal(self, JTJv_func, z_gen_func):
        self.total_D_iter += 1
        beta2 = self.betas[1]

        num_diag_iter = self.num_update_iter

        D_est_t = hutchinson(Hz_func=JTJv_func,
                             z_gen_func=z_gen_func,
                             num_iters=num_diag_iter,
                             )

        if D_est_t.isnan().any():
            print("Warning: D_est_t contains NaN values.")
            import code; code.interact(local=locals(), banner="Debugging NaN in D_est")

        self.D_smoothed = beta2 * self.D_smoothed + (1 - beta2) * D_est_t
        self.D_est = self.D_smoothed.abs() / (1 - beta2 ** self.total_D_iter)


