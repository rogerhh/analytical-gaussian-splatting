import torch

from utils import tensor_to_params, params_to_tensor, sigmoid, inverse_sigmoid, softplus, inverse_softplus

def hellinger_clip(params, s, eps):
    std, alpha, color = tensor_to_params(params)
    s_std, s_alpha, s_color = tensor_to_params(s, activate=False)

    std_thresh = torch.sqrt(2 * (std ** 2) * eps)
    s_std.clip_(-std_thresh, std_thresh)

    alpha_thresh = torch.sqrt(4 * alpha * eps)
    s_alpha.clip_(-alpha_thresh, alpha_thresh)

    color_thresh = torch.sqrt(4 * color * eps)
    s_color.clip_(-color_thresh, color_thresh)

    s = params_to_tensor(s_std, s_alpha, s_color, activate=False)

    return s
