import torch

class AdamOptimizer:
    def __init__(self, lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = 0
        self.v = 0
        self.t = 0

    def get_update(self, g):
        self.t += 1
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * g
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * (g * g)

        m_hat = self.m / (1 - self.betas[0] ** self.t)
        v_hat = self.v / (1 - self.betas[1] ** self.t)
        update = -self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        return update
