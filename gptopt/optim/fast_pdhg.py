import torch
from .pdhg import *

def pdhg_initialize_variables(A1, Z0=None, Y0=None):
    m, n = A1.shape
    if Z0 is not None:
        Z = Z0 
    else:
        Z = torch.zeros((2 * m, n), device=A1.device, dtype=A1.dtype)
    Z_bar = Z.clone() 
    if Y0 is not None:
        Y = Y0
    else:
        Y = torch.zeros((n, n), device=A1.device, dtype=A1.dtype)
    return Z, Z_bar, Y


def pdhg_stopping_criteria(r1, r2, r1_rel, r2_rel, eps_abs, eps_rel, min_iter, t):
    if t < min_iter:
        return False 
    if (r1 < eps_abs and r2 < eps_abs) or (r1_rel < eps_rel and r2_rel < eps_rel):
        return True  
    return False


def pdhg_kq_attn_layer(
    prox_h_conj,
    A1: torch.Tensor,
    A2: torch.Tensor,
    G1: torch.Tensor,
    G2: torch.Tensor,
    beta: float,
    mu: float = 0.0,
    lamb_max=None,
    max_iter: int = 100,
    eps_abs: float = 1e-3,
    eps_rel: float = 1e-3,
    stopping: bool = False,
    min_iter: int = 10,
    Y0: torch.Tensor | None = None,
    theta: float = 1.0, 
    Z0: torch.Tensor | None = None, 
    diag_scaling: bool = False,
    h_conj= None,
    f_star: float | None = None,
    pd_residuals: Optional[callable] = None, 
    verbose: bool = False,
    equilibration: bool = False,
    ): 
    """
    PDHG method for solving 
                minimize_Z      tr(G^TZ) + h(mathcal_A(Z)) + (mu/2) ||Z||_F^2
    """
    if lamb_max is None:
        nA = A1.pow(2).sum().sqrt().item()
        nB = A2.pow(2).sum().sqrt().item()
        lamb_max = (nA * nA + nB * nB) ** 0.5 
        if verbose:
            print(f"||A||_op <= {lamb_max:.4e}")

    rho = gamma = max(min(0.99 / (lamb_max + 1e-12), 1e4), 1e-5)
    Z, Z_bar, Y = pdhg_initialize_variables(A1=A1, Z0=Z0, Y0=Y0) 
    Grad = torch.cat([G1, G2], dim=0)
    m, n = A1.shape

    record = PDHGResidualRecorder(pd_residuals=pd_residuals,
                                    A1=A1, A2=A2, G1=G1, G2=G2,
                                    beta=beta, mu=mu, f_star=f_star)

    if diag_scaling:
        R, Gamma1, Gamma2 = pdhg_diagonal_scaling(A=A2, B=A1, eta=0.99)
        Gamma = torch.cat([Gamma1, Gamma2], dim=0)
        rho = gamma = 1
    elif equilibration:
        R, Gamma1, Gamma2 = ruiz_equilibration(A1=A1, A2=A2, num_iters=50, eps=1e-8)
        Gamma = torch.cat([Gamma1, Gamma2], dim=0)
    else:
        R = torch.ones((n, n), device=Y.device, dtype=Y.dtype)
        Gamma = torch.ones((2 * m, 1), device=A1.device, dtype=A1.dtype) 

    dual_val = None 
    record.record(0, Y=Y, Z=Z, dual_val=dual_val)
    
    for t in range(max_iter):
        # PDHG updates
        Y_new = prox_h_conj(Y + rho * R * mathcal_A_linop(A1=A1, A2=A2, Z=Z_bar), rho, R=R)
        Z_new = (1 / (1 + gamma * Gamma * mu)) * (Z - gamma * Gamma * (mathcal_A_adj_linop(A1=A1, A2=A2, Y=Y_new) + Grad))
        
        # if primal is strongly convex -- record dual values
        if mu > 0 and h_conj is not None: 
            dual_val = - h_conj(Y_new) - (1/(2 * mu)) * (A1 @ Y_new.t() + G1).pow(2).sum()
            dual_val = dual_val - (1/(2 * mu)) * (A2 @ Y_new + G2).pow(2).sum() 

        r1, r1_rel, r2, r2_rel = record.record(t, Y=Y_new, Z=Z_new, dual_val=dual_val)
 
        Z_bar = Z_new + theta * (Z_new - Z)
        Z = Z_new
        Y = Y_new

        if pdhg_stopping_criteria(r1, r2, r1_rel, r2_rel, eps_abs, eps_rel, min_iter, t): 
            break 

    return Z, record.as_dict(), (Y_new.pow(2).sum()).sqrt().item()


def ruiz_equilibration(A1: torch.Tensor, A2: torch.Tensor, num_iters=10, eps=1e-8, debug=False):
    """
    Ruiz equilibration for linear operator 
        \mathcal{A}(Z) = Z1^T A1 + A2^T Z2
    returns R, Gamma1, Gamma2 such that the matrix \tilde K of the the equilibrated operator
        \tilde \mathcal{A}(Z) = R * (Z1^T (Gamma2 A2) + (Gamma1 A1)^T Z2)
    is s.t. -1 <= \tilde K_{ij} <= 1
    where \vec{\tilde \mathcal{A}(Z)} = \tilde K [\vec{Z1}; \vec{Z2}]
    """
    device, dtype = A1.device, A1.dtype
    p1, n = A1.shape
    p2, n2 = A2.shape
    assert n == n2

    R = torch.ones((n, n), device=device, dtype=dtype)
    Gamma1 = torch.ones((p1, n), device=device, dtype=dtype)
    Gamma2 = torch.ones((p2, n), device=device, dtype=dtype)

    absA1 = A1.abs()
    absA2 = A2.abs()

    def inv_sqrt_pos(x: torch.Tensor) -> torch.Tensor:
        # 1/sqrt(x) for x>eps else 1 (do nothing if identically zero)
        return torch.where(x > eps, torch.rsqrt(x), torch.ones_like(x))

    for _ in range(num_iters): 
        # ---- Row max for K rows (i,j) ----
        # term1(i,j) = max_ℓ |Gamma1_{ℓ,i}| |A1_{ℓ,j}|
        term1 = (Gamma1[:, :, None] * absA1[:, None, :]).amax(dim=0)          # (n,n)
        # term2(i,j) = max_ℓ |A2_{ℓ,i}| |Gamma2_{ℓ,j}|
        term2 = (absA2[:, :, None] * Gamma2[:, None, :]).amax(dim=0)          # (n,n)
        row_max = R * torch.maximum(term1, term2)                         # (n,n)

        # ---- Column max for Z1 columns (ℓ,i) ----
        # m1(ℓ,i) = max_j |A1_{ℓ,j}| |R_{i,j}|
        m1 = (absA1[:, None, :] * R[None, :, :]).amax(dim=2)              # (p1,n)
        col1_max = Gamma1 * m1
        

        # ---- Column max for Z2 columns (ℓ,j) ----
        # m2(ℓ,j) = max_i |A2_{ℓ,i}| |R_{i,j}|
        m2 = (absA2[:, :, None] * R[None, :, :]).amax(dim=1)              # (p2,n)
        col2_max = Gamma2 * m2

        # ---- Update ----
        Gamma1 = Gamma1 * inv_sqrt_pos(col1_max)
        Gamma2 = Gamma2 * inv_sqrt_pos(col2_max)
        R = R * inv_sqrt_pos(row_max)

    return R, Gamma1, Gamma2


