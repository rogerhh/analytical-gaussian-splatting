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

def render_gaussians(test_points, stds, alphas, colors, bg, debug=False, with_bg=True, dropout_ratio=0.0):
    K = alphas.shape[0]
    m = test_points.shape[0]
    T = torch.ones((m, 1))
    render = torch.zeros((m, 3))

    if dropout_ratio > 0.0:
        dropout_probability = dropout_ratio * (1 - alphas)
        dropout_mask = torch.bernoulli(dropout_probability).bool()
        alphas = alphas * (~dropout_mask).float()

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

def render_gaussians_alpha_grid(test_points, stds, alpha_grid, colors, bg, debug=False, with_bg=True):
    K = alpha_grid.shape[-1]
    grid_dims = alpha_grid.shape[:-1]
    m = test_points.shape[0]
    render = torch.zeros((*grid_dims, m, 3))
    evaluated_alphas = alpha_grid[:,:,None,:] * torch.exp(-0.5 * (test_points / stds.T) ** 2)[None,None,:,:]
    evaluated_alpha1 = evaluated_alphas[:,:,:,0]
    evaluated_alpha2 = evaluated_alphas[:,:,:,1]
    color1 = colors[0:1, :].view(1, 1, 1, -1)
    color2 = colors[1:2, :].view(1, 1, 1, -1)

    T = 1.0
    render = render + (T * evaluated_alpha1).unsqueeze(-1) * color1
    T = T * (1 - evaluated_alpha1)
    render = render + (T * evaluated_alpha2).unsqueeze(-1) * color2
    T = T * (1 - evaluated_alpha2)

    if with_bg:
        render = render + T * bg

    if debug:
        print(f"Final render after background: {render.squeeze().detach().numpy()}")
    return render

def loss_func(test_points, params_gt, params, bg, dropout_ratio=0.0):
    std_gt, alpha_gt, color_gt = tensor_to_params(params_gt)
    std, alpha, color = tensor_to_params(params)

    render_gt = render_gaussians(test_points, std_gt, alpha_gt, color_gt, bg, debug=False)
    render = render_gaussians(test_points, std, alpha, color, bg, debug=False, dropout_ratio=dropout_ratio)
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

def g_func(test_points, params_gt, params, bg, dropout_ratio=0.0):
    with torch.enable_grad():
        loss_vec = loss_func(test_points, params_gt, params, bg, dropout_ratio=dropout_ratio)
        loss = 0.5 * (loss_vec ** 2).sum()

    with torch.no_grad():
        params.grad = None
        loss_vec.backward(loss_vec)
        g = params.grad

    return g, loss
        
def z_gen_func(params):
    return torch.randint(0, 2, params.shape).float() * 2.0 - 1.0
    
def JTJv_func(v, test_points, params_gt, params, bg, dropout_ratio=0.0):
    params.grad = None
    with torch.enable_grad(), fwAD.dual_level():
        params_dual = fwAD.make_dual(params, v)
        loss_vec_dual = loss_func(test_points, params_gt, params_dual, bg, dropout_ratio=dropout_ratio)
        loss_vec_primal, loss_vec_tangent = fwAD.unpack_dual(loss_vec_dual)
        loss_vec_primal.backward(loss_vec_tangent)

    with torch.no_grad():
        JTJv = params.grad

    if JTJv.isnan().any():
        print("Warning: JTJv contains NaN values.")
        import code; code.interact(local=locals(), banner="Debugging NaN in JTJv")

    return JTJv

def compute_J(test_points, params_gt, params, bg, dropout_ratio=0.0):
    with torch.enable_grad():
        loss_vec = loss_func(test_points, params_gt, params, bg, dropout_ratio=dropout_ratio)
        m = loss_vec.shape[0]
        J = torch.zeros((m, params.numel()))
        for i in range(m):
            params.grad = None
            loss_vec[i].backward(retain_graph=True)
            J[i, :] = params.grad.flatten()
    return J

def compute_hessian_diagonal(test_points, params_gt, params, bg, dropout_ratio=0.0):
    n = params.numel()
    D = torch.zeros(n)
    for i in range(n):
        v = torch.zeros_like(params).flatten()
        v[i] = 1.0
        v = v.view_as(params)
        JTJv = JTJv_func(v, test_points, params_gt, params, bg, dropout_ratio=dropout_ratio)
        D[i] = JTJv.flatten()[i]
    return D.view_as(params)
