import torch 

from .attn_utils import *
from .linop import *



@torch.no_grad()
def nesterov_lmax_moreau(A_linop: TorchLinearOperator,
                    Grad: torch.Tensor, 
                    beta: float,
                    mu: float,
                    lamb_max=None,
                    max_iter: int = 500,
                    eps_abs: float = 1e-8,
                    eps_rel: float = 1e-8,
                    f_star: float | None = None,
                    Z0: torch.Tensor | None = None,
                    stopping: bool = True,
                    pd_residuals: Optional[Callable] = None,
                    min_iter: int = 10
                    ):
    """
    Nesterov for solving the dual problem:
    minimize  \langle G, Z \rangle +  {}^\mu h(\mathcal{A}(Z))
    \mathcal{A}(Z) = Z1^T A1 + A2^T Z2   
    """  
    if lamb_max is None: 
        lamb_max = A_linop.fro_norm 

    n_head = A_linop.n_head
    m, n = linop_mn_from_shape(A_linop, n_head)

    if Z0 is not None: 
        Z_old = Z0.clone()
    else: 
        Z_old = torch.zeros((2 * n_head * m, n), device=A_linop.device, dtype=A_linop.dtype)
    U = Z_old.clone() 

    record = ResidualRecorder(pd_residuals=pd_residuals, beta=beta, 
                              A_linop=A_linop, Grad=Grad,
                              mu_moreau=mu, f_star=f_star)
    AU = A_linop.matvec(Z_old)
    Y = (1/mu) * (AU - torch.clamp(AU, min=-beta, max=beta))
    primal_val = ((Grad * Z_old).sum() + (mu / (2)) * torch.norm(Y, p="fro").square()).item()
    record.record_without_relax_or_reg(0, Z=Z_old, Y=Y, primal_val=primal_val)

    step_size = mu / lamb_max**2
    
    for t in range(1, max_iter+1):
        # Gradient step
        AU = A_linop.matvec(U)
        Y = (1/mu) * (AU - torch.clamp(AU, min=-beta, max=beta))
        grad = Grad + A_linop.rmatvec(Y)
        # grad step
        Z = U - step_size * grad
        # Momentum update
        U = Z + (t - 1) / (t + 2) * (Z - Z_old)
        Z_old = Z
                   
        primal_val = ((Grad * Z).sum() + (mu / (2)) * torch.norm(Y, p="fro").square()).item()
        r1, r1_rel, r2, r2_rel = record.record_without_relax_or_reg(t, Y=Y, Z=Z, primal_val=primal_val)

        if attn_stopping_criteria(r1, r2, r1_rel, r2_rel, eps_abs, eps_rel, min_iter, t) and stopping: 
            break 
    
    return Z, record.as_dict(), Y
