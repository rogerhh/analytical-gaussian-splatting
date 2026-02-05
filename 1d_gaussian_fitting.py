import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from matplotlib import pyplot as plt

from utils import tensor_to_params, params_to_tensor, sigmoid, inverse_sigmoid, softplus, inverse_softplus
from utils import has_tangent, get_tangent

from adam_optimizer import AdamOptimizer
from sophia_optimizer import SophiaOptimizer

from hellinger_clip import hellinger_clip

import argparse


def render_gaussians(test_points, stds, alphas, colors, bg, debug=False, with_bg=True):
    K = alphas.shape[0]
    m = test_points.shape[0]
    T = torch.ones((m, 1))
    render = torch.zeros((m, 3))
    evaluated_alphas = alphas.T * torch.exp(-0.5 * (test_points / stds.T) ** 2)
    for k in range(K):
        render = render + T * evaluated_alphas[:, k:k+1] * colors[k:k+1, :]
        T = T * (1 - evaluated_alphas[:, k:k+1])

        if debug:
            print(f"After Gaussian {k}: render = {render.squeeze().detach().numpy()}, T = {T.squeeze().detach().numpy()}")
            print(f"evaluated_alphas[:, {k}] = {evaluated_alphas[:, k].squeeze().detach().numpy()}")
    if with_bg:
        render = render + T * bg

    if debug:
        print(f"Final render after background: {render.squeeze().detach().numpy()}")
    return render

def loss_func(test_points, params_gt, params, bg):
    std_gt, alpha_gt, color_gt = tensor_to_params(params_gt)
    std, alpha, color = tensor_to_params(params)

    render_gt = render_gaussians(test_points, std_gt, alpha_gt, color_gt, bg, debug=False)
    render = render_gaussians(test_points, std, alpha, color, bg, debug=False)
    # loss = (render - render_gt).norm() ** 2
    # loss += 1e-2 * alpha.mean()  # regularization
    # loss = (render - render_gt).abs().mean()
    # loss += 1e-1 * alpha.mean()  # regularization

    pix_vec = ((render - render_gt).abs() + 1e-16).sqrt() * ((1 / render.numel()) ** 0.5)
    reg_vec = (alpha.abs() + 1e-16).sqrt() * ((1 / alpha.numel()) ** 0.5) * (1e-1 ** 0.5)

    loss_vec = torch.cat([pix_vec.flatten(), reg_vec.flatten()], dim=0)

    if has_tangent(loss_vec):
        loss_vec_tangent = get_tangent(loss_vec)
        if loss_vec_tangent.isnan().any():
            print("Warning: loss_vec tangent contains NaN values.")
            import code; code.interact(local=locals(), banner="Debugging NaN in loss_vec tangent")

    return loss_vec

def g_func(test_points, params_gt, params, bg):
    with torch.enable_grad():
        loss_vec = loss_func(test_points, params_gt, params, bg)
        loss = 0.5 * (loss_vec ** 2).sum()

    with torch.no_grad():
        params.grad = None
        loss_vec.backward(loss_vec)
        g = params.grad

    return g, loss
        
def z_gen_func(params):
    return torch.randint(0, 2, params.shape).float() * 2.0 - 1.0
    
def JTJv_func(v, test_points, params_gt, params, bg):
    with torch.enable_grad(), fwAD.dual_level():
        params_dual = fwAD.make_dual(params, v)
        loss_vec_dual = loss_func(test_points, params_gt, params_dual, bg)
        loss_vec_primal, loss_vec_tangent = fwAD.unpack_dual(loss_vec_dual)
        loss_vec_primal.backward(loss_vec_tangent)

    with torch.no_grad():
        JTJv = params.grad

    if JTJv.isnan().any():
        print("Warning: JTJv contains NaN values.")
        import code; code.interact(local=locals(), banner="Debugging NaN in JTJv")

    return JTJv

def main(num_iter, use_adam):
    torch.manual_seed(0)

    K = 5   # Number of Gaussians

    random_bg = True

    # Generate ground truth parameters
    # std_gt = torch.rand(1, 1)
    # alpha_gt = torch.rand(1, 1) - 0.2
    # color_gt = torch.rand(1, 3)
    # params_gt = params_to_tensor(std_gt, alpha_gt, color_gt)
    params_gt = torch.rand(1, 5)
    std_gt, alpha_gt, color_gt = tensor_to_params(params_gt)
    bg = torch.rand(3)

    m = 10  # Number of test points
    test_range = 1.0
    spacing = test_range / m

    test_points = np.linspace(-m * spacing / 2, m * spacing / 2, m)
    test_points = torch.tensor(test_points, dtype=torch.float32)[:, None]

    evaluated_alpha_gt = torch.exp(-0.5 * (test_points / std_gt) ** 2)

    render_gt = render_gaussians(test_points, std_gt, alpha_gt, color_gt, bg)

    # import code; code.interact(local=locals())

    # alpha_param = torch.nn.Parameter(torch.rand(K, 3))
    # params = torch.nn.Parameter(torch.rand(K, 5))
    params = torch.rand(K, 5, requires_grad=True)

    optimizer = torch.optim.Adam([{"params": params, "lr": 0.02}])
    adam_optimizer = AdamOptimizer(lr=0.02)
    sophia_optimizer = SophiaOptimizer(diagonal_update_interval=1, num_update_iter=1, betas=(0.9, 0.999))

    all_losses = []
    all_alphas = [[] for _ in range(K)]

    for it in range(num_iter):

        if random_bg:
            bg = torch.rand(3)

        with torch.no_grad():
            g, loss = g_func(test_points, params_gt, params, bg)
            s_adam = adam_optimizer.get_update(g)
            s_sophia = sophia_optimizer.get_update(g, 
                                                   JTJv_func=lambda v: JTJv_func(v, test_points, params_gt, params, bg), 
                                                   z_gen_func=lambda: z_gen_func(params))
            s_sophia_old = s_sophia
            s_sophia = hellinger_clip(params, s_sophia_old, eps=0.001)

            if use_adam:
                s = s_adam
            else:
                s = s_sophia
            # import code; code.interact(local=locals(), banner="After getting updates")
            params += s
            # s_sophia = sophia_optimizer.get_update(g, JTJv_func, z_gen_func)
            # params += s_sophia

        std_gt, alpha_gt, color_gt = tensor_to_params(params_gt)
        std, alpha, color = tensor_to_params(params)

        zero_point = torch.zeros(1, 1)
        render_zero_gt = render_gaussians(zero_point, std_gt, alpha_gt, color_gt, bg, debug=False, with_bg=True)
        render_zero = render_gaussians(zero_point, std, alpha, color, bg, debug=False, with_bg=True)
        render_zero_gt_no_bg = render_gaussians(zero_point, std_gt, alpha_gt, color_gt, bg, debug=False, with_bg=False)
        render_zero_bg = render_gaussians(zero_point, std, alpha, color, bg, debug=False, with_bg=False)

        if it % 10 == 0:
            print(f"Iter {it}: Loss = {loss.item()}, alpha = {alpha.detach().numpy().flatten()}, alpha_gt = {alpha_gt.detach().numpy()}")
            print(f"color = {color.detach().numpy().flatten()}, color_gt = {color_gt.detach().numpy()}")

            print(f"with bg: render: {render_zero.squeeze().detach().numpy()}, render_gt: {render_zero_gt.squeeze().detach().numpy()}")
            print(f"no bg: render: {render_zero_bg.squeeze().detach().numpy()}, render_gt: {render_zero_gt_no_bg.squeeze().detach().numpy()}")

        all_losses.append(loss.item())
        for k in range(K):
            all_alphas[k].append(alpha[k].item())

        print(f"sophia optimizer ", sophia_optimizer.D_est[:, 1])

    # all_losses = np.array(all_losses)
    # all_alphas = np.array(all_alphas)
     
    figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(all_losses, label='Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend()

    for k in range(K):
        ax2.plot(all_alphas[k])
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Opacity')
    ax2.axhline(y=alpha_gt.item(), color='r', linestyle='--', label='Ground Truth')
    ax2.legend()

    figname = f'figures/1d_gaussian_fitting_loss_alpha_adam.png' if use_adam else f'figures/1d_gaussian_fitting_loss_alpha_sophia.png'
    plt.savefig(figname)

    # import code with both globals and locals
    import code; code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='1D Gaussian Fitting')
    parser.add_argument('--num_iter', type=int, default=100, help='Number of iterations')
    parser.add_argument('--use_adam', action='store_true', help='Use Adam optimizer instead of Sophia')
    args = parser.parse_args()

    main(num_iter=args.num_iter, use_adam=args.use_adam)
