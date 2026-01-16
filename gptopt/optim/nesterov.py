import torch 

from .attn_utils import *



@torch.no_grad()
def nesterov_lmax_moreau(A2: torch.Tensor,
                    A1: torch.Tensor,
                    G1: torch.Tensor,
                    G2: torch.Tensor, 
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
        nA = A1.pow(2).sum().sqrt().item()
        nB = A2.pow(2).sum().sqrt().item()
        lamb_max = (nA * nA + nB * nB) ** 0.5 
        # print(f"{lamb_max=}")

    if Z0 is not None: 
        Z_old = Z0.clone()
    else:
        Z_old = torch.zeros((2 * G1.shape[0], G1.shape[1]), device=G1.device, dtype=G1.dtype)
    U = Z_old.clone()
    G12 = torch.cat([G1, G2], dim=0)

    record = ResidualRecorder(pd_residuals=pd_residuals,
                                    A1=A1, A2=A2, G1=G1, G2=G2,
                                    beta=beta, mu_moreau=mu, f_star=f_star)
    AU = mathcal_A_linop(A1=A1, A2=A2, Z=Z_old)
    Y = (1/mu) * (AU - torch.clamp(AU, min=-beta, max=beta))
    primal_val = ((G12 * Z_old).sum() + (mu / (2)) * Y.pow(2).sum()).item()
    record.record_without_relax_or_reg(0, Z=Z_old, Y=Y, primal_val=primal_val)

    step_size = mu / lamb_max**2
    
    for t in range(1, max_iter+1):
        # Gradient step
        AU = mathcal_A_linop(A1=A1, A2=A2, Z=U)
        Y = (1/mu) * (AU - torch.clamp(AU, min=-beta, max=beta))
        grad = G12 + mathcal_A_adj_linop(A1=A1, A2=A2, Y=Y)
        # grad step
        Z = U - step_size * grad
        # Momentum update
        U = Z + (t - 1) / (t + 2) * (Z - Z_old)
        Z_old = Z
                   
        primal_val = ((G12 * Z).sum() + (mu / (2)) * Y.pow(2).sum()).item()
        r1, r1_rel, r2, r2_rel = record.record_without_relax_or_reg(t, Y=Y, Z=Z, primal_val=primal_val)

        if attn_stopping_criteria(r1, r2, r1_rel, r2_rel, eps_abs, eps_rel, min_iter, t) and stopping: 
            break 
    
    return Z, record.as_dict()

