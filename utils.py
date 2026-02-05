import torch
import torch.autograd.forward_ad as fwAD

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))
def inverse_sigmoid(y):
    return torch.log(y / (1 - y))
def softplus(x):
    return torch.log(1 + torch.exp(x))
def inverse_softplus(y):
    return torch.log(torch.exp(y) - 1)

def tensor_to_params(x, activate=True):
    K = x.shape[0]
    stds = softplus(x[:, 0:1]) if activate else x[:, 0:1]
    alphas = sigmoid(x[:, 1:2]) if activate else x[:, 1:2]
    colors = sigmoid(x[:, 2:5]) if activate else x[:, 2:5]
    return stds, alphas, colors

def params_to_tensor(stds, alphas, colors, activate=True):
    K = alphas.shape[0]
    x_stds = inverse_softplus(stds) if activate else stds
    x_alphas = inverse_sigmoid(alphas) if activate else alphas
    x_colors = inverse_sigmoid(colors) if activate else colors
    x = torch.zeros((K, 5))
    x[:, 0:1] = x_stds
    x[:, 1:2] = x_alphas
    x[:, 2:5] = x_colors
    return x

def has_tangent(x):
    return fwAD.unpack_dual(x).tangent is not None

def get_tangent(x):
    if has_tangent(x):
        return fwAD.unpack_dual(x).tangent
    elif isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
