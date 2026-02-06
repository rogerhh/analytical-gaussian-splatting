import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from matplotlib import pyplot as plt

from utils import tensor_to_params, params_to_tensor, sigmoid, inverse_sigmoid, softplus, inverse_softplus
from utils import has_tangent, get_tangent
from utils import render_gaussians, render_gaussians_alpha_grid, loss_func, g_func, z_gen_func, JTJv_func, compute_J, compute_hessian_diagonal

from adam_optimizer import AdamOptimizer
from sophia_optimizer import SophiaOptimizer

from hellinger_clip import hellinger_clip
from diagonal_estimator import hutchinson

import argparse

def main(num_iter, use_adam, diagonal_update_interval=10, use_D_exact=False, dropout_ratio=0.0, dropout_JTJv=True):
    torch.manual_seed(0)

    K = 2   # Number of Gaussians

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
    params = params_gt.tile((K, 1))
    params[:, 1] = torch.rand(K)  # initialize alpha to a different value than gt
    params.requires_grad = True

    optimizer = torch.optim.Adam([{"params": params, "lr": 0.02}])
    adam_optimizer = AdamOptimizer(lr=0.02)
    sophia_optimizer = SophiaOptimizer(diagonal_update_interval=diagonal_update_interval, num_update_iter=1, betas=(0.9, 0.99))

    all_losses = []
    all_alphas = [[] for _ in range(K)]

    for it in range(num_iter):

        # if it > 1500:
        #     use_D_exact = True

        if random_bg:
            bg = torch.rand(3)

        with torch.no_grad():
            g, loss = g_func(test_points, params_gt, params, bg, dropout_ratio=dropout_ratio)
            s_adam = adam_optimizer.get_update(g)
            dropout_JTJv_ratio = dropout_ratio if dropout_JTJv else 0.0
            s_sophia = sophia_optimizer.get_update(g, 
                                                   JTJv_func=lambda v: JTJv_func(v, test_points, params_gt, params, bg, dropout_ratio=dropout_JTJv_ratio), 
                                                   z_gen_func=lambda: z_gen_func(params),
                                                   use_D_exact=use_D_exact,
                                                   D_exact_func=lambda: compute_hessian_diagonal(test_points, params_gt, params, bg),
                                                   )
            s_sophia_old = s_sophia
            s_sophia = hellinger_clip(params, s_sophia_old, eps=0.001)

            if use_adam:
                s = s_adam
            else:
                s = s_sophia
            # import code; code.interact(local=locals(), banner="After getting updates")

            s_copy = s.clone()
            s *= 0.0
            s[:, 1] = s_copy[:, 1]  # only update alpha
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

        if it % 100 == 0:
            print(f"Iter {it}: Loss = {loss.item()}, alpha = {alpha.detach().numpy().flatten()}, alpha_gt = {alpha_gt.detach().numpy()}")
            print(f"color = {color.detach().numpy().flatten()}, color_gt = {color_gt.detach().numpy()}")
            print(f"std = {std.detach().numpy().flatten()}, std_gt = {std_gt.detach().numpy().flatten()}")
            print(f"with bg: render: {render_zero.squeeze().detach().numpy()}, render_gt: {render_zero_gt.squeeze().detach().numpy()}")
            print(f"no bg: render: {render_zero_bg.squeeze().detach().numpy()}, render_gt: {render_zero_gt_no_bg.squeeze().detach().numpy()}")
            print(f"sophia optimizer ", sophia_optimizer.D_est[:, 1])

        all_losses.append(loss.item())
        for k in range(K):
            all_alphas[k].append(alpha[k].item())
     
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

    figname = f'figures/2gaussian_alpha_fitting_adam' if use_adam else f'figures/2gaussian_alpha_fitting_sophia'
    if not use_adam:
        figname += '_with_D_exact' if use_D_exact else '_with_D_est'
    if dropout_ratio > 0.0:
        figname += f'_dropout_{dropout_ratio}'
    if not dropout_JTJv:
        figname += '_no_dropout_JTJv'
    plt.savefig(figname + '.png')

    J = compute_J(test_points, params_gt, params, bg)
    JTJ = J.T @ J
    D = compute_hessian_diagonal(test_points, params_gt, params, bg)
    D_est_1k = hutchinson(lambda v: JTJv_func(v, test_points, params_gt, params, bg), 
                          lambda: z_gen_func(params),
                          num_iters=1000)

    
    with torch.no_grad():
        alpha1 = torch.linspace(0.001, 1.0, 500)
        alpha2 = torch.linspace(0.001, 1.0, 500)
        alpha1, alpha2 = torch.meshgrid(alpha1, alpha2, indexing='ij')
        alpha_grid = torch.stack([alpha1, alpha2], dim=-1)

        render_gt_no_bg = render_gaussians(test_points, std_gt, alpha_gt, color_gt, bg, with_bg=False)
        std, alpha, color = tensor_to_params(params)
        render_grid_no_bg = render_gaussians_alpha_grid(test_points, std, alpha_grid, color, bg, with_bg=False)

        pix_grid = ((render_gt_no_bg[None,None,:,:] - render_grid_no_bg).abs().mean(dim=(-1,-2)))
        reg_grid = 1e-1 * alpha_grid.abs().mean(dim=-1)

        loss_grid = pix_grid + reg_grid

        plt.figure(figsize=(8, 6))
        plt.contourf(alpha1.detach().numpy(), alpha2.detach().numpy(), loss_grid.detach().numpy(), levels=50, cmap='viridis')
        plt.plot(all_alphas[0], all_alphas[1], color='r', label='Optimization Path')
        plt.colorbar(label='Loss')
        plt.savefig(figname + '_alpha_grid_heatmap.png')
        print(f"Saved alpha grid heatmap to {figname + '_alpha_grid_heatmap.png'}")

        import code; code.interact(local=locals(), banner="After optimization, before plotting alpha grid heatmap")
        



    # import code with both globals and locals
    import code; code.interact(local=dict(globals(), **locals()), banner="End of main, interactive session")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='1D Gaussian Fitting')
    parser.add_argument('--num_iter', type=int, default=100, help='Number of iterations')
    parser.add_argument('--use_adam', action='store_true', help='Use Adam optimizer instead of Sophia')
    parser.add_argument('--diagonal_update_interval', type=int, default=10, help='Interval for updating the diagonal in Sophia optimizer')
    parser.add_argument('--use_D_exact', action='store_true', help='Use exact Hessian diagonal for Sophia optimizer')
    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='Dropout ratio for alpha during rendering (between 0 and 1)')
    parser.add_argument('--no_dropout_JTJv', action='store_true', help='Do not apply dropout to the JTJv product in the Sophia optimizer diagonal estimation')
    args = parser.parse_args()

    main(num_iter=args.num_iter, use_adam=args.use_adam, 
         diagonal_update_interval=args.diagonal_update_interval,
         use_D_exact=args.use_D_exact, dropout_ratio=args.dropout_ratio, dropout_JTJv=not args.no_dropout_JTJv)
