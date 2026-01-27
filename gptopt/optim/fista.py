import torch 

from .attn_utils import *
from .linop import *


@torch.no_grad()
def fista_ls_l1_reg(A_linop: TorchLinearOperator,
                    Grad: torch.Tensor, 
                    beta: float,
                    mu: float,
                    lamb_max=None,
                    max_iter: int = 500,
                    eps_abs: float = 1e-8,
                    eps_rel: float = 1e-8,
                    f_star: float | None = None,
                    Y0: torch.Tensor | None = None,
                    stopping: bool = True,
                    pd_residuals: Optional[Callable] = None,
                    min_iter: int = 10
                    ):
    """
    Fista for solving the dual problem:
    maximize -\frac{1}{2\mu}\|\mathcal{A}^*(Y) + G\|_F^2 - \beta \|vec(Y)\|_1
    \mathcal{A}^*(Y) = [A1 Y^T; A2 Y]  
    """  
    if lamb_max is None: 
        lamb_max = A_linop.fro_norm
    
    step_size = mu / lamb_max**2
    n_head = A_linop.n_head
    m, n = linop_mn_from_shape(A_linop, n_head)
    if Y0 is not None: 
        Y_old = Y0.clone()
    else: 
        Y_old = torch.zeros((n_head * n, n), device=Grad.device, dtype=Grad.dtype)
    tilde_Y = Y_old.clone() 
    record = ResidualRecorder(pd_residuals=pd_residuals, 
                              A_linop=A_linop, Grad=Grad,
                              beta=beta, mu=mu, f_star=f_star)
    record.record_without_relax_or_reg(0, Y=Y_old, 
        Z=(1 / mu) * (- Grad - A_linop.rmatvec(Y_old)), dual_val=-np.inf)
    
    for t in range(1, max_iter+1):
        # Gradient step
        grad = (1/mu) * A_linop.matvec(A_linop.rmatvec(tilde_Y) + Grad)
        # Prox step
        Y_new = prox_l1(tilde_Y - step_size * grad, beta * step_size)

        # Extrapolation step
        tilde_Y = Y_new + ((t - 1) / (t + 2)) * (Y_new - Y_old)
         
        # primal recovery
        Z_fista = (1 / mu) * (- Grad - A_linop.rmatvec(Y_new)) 

        Y_old.copy_(Y_new)
         
        dual_val = (-(mu/2) * (Z_fista).pow(2).sum() - beta * torch.sum(torch.abs(Y_new))).item()
            
        r1, r1_rel, r2, r2_rel = record.record_without_relax_or_reg(t, Y=Y_new, Z=Z_fista, dual_val=dual_val)

        if attn_stopping_criteria(r1, r2, r1_rel, r2_rel, eps_abs, eps_rel, min_iter, t) and stopping: 
            break 
    
    return Y_new, Z_fista, record.as_dict()


