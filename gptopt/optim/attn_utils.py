import torch
import numpy as np

from typing import Optional, Callable, Literal, Any


def mathcal_A_linop(*, A1, A2, Z): # A operator
    m = A1.shape[0]
    return mathcal_A_linop_base(A1=A1, A2=A2, Z1=Z[:m, :], Z2=Z[m:, :])


def mathcal_A_linop_base(*, A1, A2, Z1, Z2): # A operator
    return Z1.T @ A1 + A2.T @ Z2
       
        
def mathcal_A_adj_linop(*, A1, A2, Y):         # A^* operator  
    return torch.cat([A1 @ Y.T, A2 @ Y], dim=0)


def attn_stopping_criteria(r1, r2, r1_rel, r2_rel, eps_abs, eps_rel, min_iter, t):
    if t < min_iter:
        return False 
    if (r1 <= eps_abs and r2 <= eps_abs) or (r1_rel <= eps_rel and r2_rel <= eps_rel):
        return True  
    return False


def prox_l1(x, beta, R=None):
    # soft-thresholding
    # proximal operator for l1 norm beta * ||x||_1
    if R is not None:
        threshold = beta * R
    else:
        threshold = beta
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)


class ResidualRecorder: 

    def __init__(
        self,
        *,
        pd_residuals: Optional[Callable[..., tuple[float, float, float, float]]] = None,
        mu: float = 0.0,
        f_star: float | None = None,
        normalize: Literal["none", "by_first"] = "none",
        warmup_iters: int = 5,
        eps: float = 1e-8,
        **pd_residuals_kwargs: Any,
    ):
        self.pd_residuals = pd_residuals
        self.kw = dict(pd_residuals_kwargs)
        self.mu = float(self.kw.get("mu", mu))
        self.kw.setdefault("mu", self.mu)
        self.f_star = f_star
        self.normalize = normalize
        self.warmup_iters = int(warmup_iters)
        self.eps = float(eps)

        self.r1: list[float] = []
        self.r2: list[float] = []
        self.r1_rel: list[float] = []
        self.r2_rel: list[float] = []
        self.r_rel: list[float] = []
        self.dual_vals: list[float] = []
        self.rel_gap: list[float] = []
        self._norm1: float | None = None
        self._norm2: float | None = None
        self.z_norm: list[float] = []
        self.y_norm: list[float] = []


    def update(self, t: int, *, r1: float, r2: float, r1_rel=None, r2_rel=None, dual_val=None) -> None:
        self.r1.append(r1); self.r2.append(r2)

        if self.normalize == "by_first" and self._norm1 is None and t >= self.warmup_iters:
            self._norm1 = max(r1, self.eps)
            self._norm2 = max(r2, self.eps)
        if (r1_rel is None or r2_rel is None) and self.normalize == "by_first" and self._norm1 is not None:
            r1_rel, r2_rel = r1 / self._norm1, r2 / self._norm2
        if r1_rel is not None and r2_rel is not None:
            self.r1_rel.append(r1_rel); self.r2_rel.append(r2_rel)

        if dual_val is not None: 
            self.dual_vals.append(dual_val)
            if self.f_star is not None:
                self.rel_gap.append(np.abs(self.f_star - dual_val) / (abs(self.f_star) + 1e-12))
        self.r_rel.append(max(r1_rel, r2_rel))

    def record(self, t: int, *, Y: torch.Tensor, Z: torch.Tensor, dual_val=None):
        r1, r1_rel, r2, r2_rel = self.pd_residuals(Y=Y, Z=Z, **self.kw)
        
        self.z_norm.append(Z.pow(2).sum().pow(0.5).item())
        self.y_norm.append(Y.pow(2).sum().pow(0.5).item())
        self.update(t, r1=r1, r2=r2, r1_rel=r1_rel, r2_rel=r2_rel, dual_val=dual_val)
        return self.r1[-1], self.r1_rel[-1], self.r2[-1], self.r2_rel[-1]

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"r1": self.r1, "r2": self.r2, "r1_rel": self.r1_rel, "r2_rel": self.r2_rel,
                             "z_norm": self.z_norm, "y_norm": self.y_norm,
                             "r_rel": self.r_rel}
        if self.mu > 0:
            d["dual_vals"] = self.dual_vals
        if self.f_star is not None:
            d["rel_gap"] = self.rel_gap
        return d

    def get(self, key: str, default=None):
        return self.as_dict().get(key, default)

    def __getitem__(self, key: str):
        return self.as_dict()[key]



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
        # term1(i,j) = max_l |Gamma1_{l,i}| |A1_{l,j}|
        term1 = (Gamma1[:, :, None] * absA1[:, None, :]).amax(dim=0)          # (n,n)
        # term2(i,j) = max_l |A2_{l,i}| |Gamma2_{l,j}|
        term2 = (absA2[:, :, None] * Gamma2[:, None, :]).amax(dim=0)          # (n,n)
        row_max = R * torch.maximum(term1, term2)                         # (n,n)

        # ---- Column max for Z1 columns (l,i) ----
        # m1(l,i) = max_j |A1_{l,j}| |R_{i,j}|
        m1 = (absA1[:, None, :] * R[None, :, :]).amax(dim=2)              # (p1,n)
        col1_max = Gamma1 * m1
        

        # ---- Column max for Z2 columns (l,j) ----
        # m2(l,j) = max_i |A2_{l,i}| |R_{i,j}|
        m2 = (absA2[:, :, None] * R[None, :, :]).amax(dim=1)              # (p2,n)
        col2_max = Gamma2 * m2

        # ---- Update ----
        Gamma1 = Gamma1 * inv_sqrt_pos(col1_max)
        Gamma2 = Gamma2 * inv_sqrt_pos(col2_max)
        R = R * inv_sqrt_pos(row_max)

    return R, Gamma1, Gamma2


def proj_subgrad_l1(AZ, Y, beta=1, y_zero_tol_abs=1e-7, y_zero_tol_rel=1e-12):
    # \min_S \beta\|AZ/\beta - S\|_F s.t. S \in \partial \|\vec(Y)\|_1 
    S = torch.sign(Y)
    y_tol = y_zero_tol_rel * Y.abs().max().item() + y_zero_tol_abs
    S[Y.abs() <= y_tol] = torch.clamp((AZ[Y.abs() <= y_tol]/beta), -1.0, 1.0)
    r = beta * (AZ / beta - S).pow(2).sum().sqrt().item()
    norm = beta * (S.numel()**0.5)
    return r, r / norm


def pd_residuals_infty_ball(A, B, Y, Z1, Z2, G1, G2, beta, mu=0, abs_tol=1e-4):
    # KKT residuals 
    # h = I_{||.||_\max \leq beta}
    # 0 \in \partial h^*(Y) - \mathcal{A}(Z)   -- primal residual
    # 0 = G + \mu Z + \mathcal{A}^*(Y).        -- dual residual
    AZ = Z1.T @ B + A.T @ Z2 
    r1, r1_rel = proj_subgrad_l1(AZ, Y, beta=beta)
    r2_1 = (G1 + mu * Z1 + B @ Y.t()).pow(2).sum().sqrt().item()
    r2_2 = (G2 + mu * Z2 + A @ Y).pow(2).sum().sqrt().item()
    r2 = (r2_1**2 + r2_2**2)**0.5 
    norm2 = ((G1.pow(2).sum() + G2.pow(2).sum()).sqrt().item())**0.5
    if norm2 < 1e-6: norm2 = 1.0
    r2_rel = r2 / norm2 
    return r1, r1_rel, r2, r2_rel


def pd_residuals_max_ball(A1, A2, Y, Z, G1, G2, beta, mu=0, abs_tol=1e-4):
    m = A1.shape[0]
    Z1, Z2 = Z[:m, :], Z[m:, :] 
    r1, r1_rel, r2, r2_rel = pd_residuals_infty_ball(B=A1, A=A2, Y=Y, Z1=Z1, Z2=Z2, G1=G1, G2=G2, 
                                   beta=beta, mu=mu, abs_tol=abs_tol)
    return r1, r1_rel, r2, r2_rel
