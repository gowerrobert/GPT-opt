
import torch
from torch.optim import Optimizer
import numpy as np
from typing import Optional, Callable, Literal, Any
from .least_squares import Y_dual_feasible, pdhg_diagonal_scaling, attn_least_squares_solve


class PDHGResidualRecorder: 

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
        self.dual_vals: list[float] = []
        self.rel_gap: list[float] = []
        self._norm1: float | None = None
        self._norm2: float | None = None


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

    def record(self, t: int, *, Y: torch.Tensor, Z: torch.Tensor, dual_val=None):
        r1, r1_rel, r2, r2_rel = self.pd_residuals(Y=Y, Z=Z, **self.kw)
        self.update(t, r1=r1, r2=r2, r1_rel=r1_rel, r2_rel=r2_rel, dual_val=dual_val)
        return self.r1[-1], self.r1_rel[-1], self.r2[-1], self.r2_rel[-1]

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"r1": self.r1, "r2": self.r2, "r1_rel": self.r1_rel, "r2_rel": self.r2_rel}
        if self.mu > 0:
            d["dual_vals"] = self.dual_vals
        if self.f_star is not None:
            d["rel_gap"] = self.rel_gap
        return d

    def get(self, key: str, default=None):
        return self.as_dict().get(key, default)

    def __getitem__(self, key: str):
        return self.as_dict()[key]



class AttnPDAdamW(Optimizer):
    """AdamW that sees (name, param) and can treat attention Q/K specially.

    Assumes attn.c_attn.weight is stacked [Q; K; V] along dim 0.
    """

    def __init__(
        self,
        named_params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        qk_lr_scale=1.0,
        max_norm_tr=1,
        pdhg_max_iter=1000,
        momentum=False,
        diag_scaling=False,
        acceleration=False,
        pd_type="pdhg",
        halpern_start: int = np.inf,
        reflected_pdhg: bool = False,
        warm_start: bool = False,
    ):
        params = []
        self.name_by_param = {}
        for name, p in named_params:
            if not p.requires_grad:
                continue
            params.append(p)
            self.name_by_param[p] = name

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            qk_lr_scale=qk_lr_scale,
            max_norm_tr=max_norm_tr,
            pdhg_max_iter=pdhg_max_iter, 
            momentum=momentum,
            diag_scaling=diag_scaling,
            acceleration=acceleration,
            pd_type=pd_type,
            halpern_start=halpern_start,
            reflected_pdhg=reflected_pdhg,
            warm_start=warm_start
        )
        print(
            f"[AttnPDAdamW] lr={lr}, betas={betas}, eps={eps}, wd={weight_decay}, "
            f"qk_lr_scale={qk_lr_scale}, max_norm_tr={max_norm_tr}, pdhg_iters={pdhg_max_iter}, {warm_start=}"
            f"\n    momentum={momentum}, diag_scaling={diag_scaling}, accel={acceleration}, {pd_type=}, {halpern_start=}, {reflected_pdhg=}"
        )
        super().__init__(params, defaults)

    def _is_attn_qkv_weight(self, name: str) -> bool:
        return ("c_attn" in name) and name.endswith("weight")

    @torch.no_grad()
    def step(self, closure=None, ):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        updated = set()
        n_att_layers = 0
        residuals_n_layers = {}
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            qk_lr_scale = group["qk_lr_scale"]
            pdhg_max_iter = group["pdhg_max_iter"]
            momentum = group["momentum"]
            pd_type = group["pd_type"] 

            for p in group["params"]:   
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5

                if momentum:
                    buf1 = state["moment1"]
                    buf2 = state["moment2"]
                    buf1.lerp_(g, 1 - beta1)
                    buf2.lerp_(g.square(), 1 - beta2)
                    
                    g = buf1 / (eps + buf2.sqrt())

                name = self.name_by_param[p]
                updated.add(name) 
                if self._is_attn_qkv_weight(name) and p.ndim == 2:
                    # interpret rows as [Q; K; V]
                    n_embed = p.shape[1] 
                    assert p.shape[0] == 3 * n_embed, print(f"{p.shape}")
                    A1, G2 = p[:n_embed, :],                  g[:n_embed, :]
                    A2, G1 = p[n_embed:2 * n_embed, :],       g[n_embed:2 * n_embed, :]
                    W_v, G_v = p[2 * n_embed: 3 * n_embed, :],  g[2 * n_embed: 3 * n_embed, :]
                    # Y0, dual_feas_res = Y_dual_feasible(A2=A2, A1=A1, G1=G1, G2=G2, maxit=20)
                    # print(f"{dual_feas_res=}, ||Y0||_F={Y0.pow(2).sum().item():.4e}")
                    # Y0 = torch.randn(A1.shape[0], A2.shape[0], device=A1.device, dtype=A1.dtype) * 1e-3
                    # print(f"|Y0||_F={Y0.pow(2).sum().item():.4e}")
                    nA = A2.pow(2).sum().sqrt().item()
                    nB = A1.pow(2).sum().sqrt().item()
                    lamb_max = (nA * nA + nB * nB) ** 0.5
                    # mu_max = (G1.t() @ A1 + A2.t() @ G2).abs().max().item() / group["max_norm_tr"]
                    # mu_reg = max(0.1 * mu_max, 1e-6)
                    mu_reg = 0
                    # print(f"{lamb_max=:.4e}, {mu_reg=:.4e}, {mu_max=:.4e}")
                    # PDHG to update key-query  
                    prox_h_conj = lambda y, rho, R: prox_l1(y, rho * group["max_norm_tr"], R=R)
                    h_conj = lambda y: group["max_norm_tr"] * torch.abs(y).sum()
                    assert A1.shape == G2.shape and A2.shape == G1.shape, \
                        print(f"{A1.shape=}, {G2.shape=}, {A2.shape=}, {G1.shape=}")
                    # Y0 = 0.001 * torch.randn((A2.shape[1], A1.shape[1]), device=A2.device, dtype=A2.dtype)
                    # Y0 = None
                    if pd_type == "pdhg": 
                        # Z1_0=-G1; Z2_0=-G2
                        # Z1_0, Z2_0 = None, None
                        if group["warm_start"]:
                            Y0, _ = Y_dual_feasible(A1=A1, A2=A2, G1=G1, G2=G2, method="lsqr", maxit=100) 
                            (Z1_0, Z2_0), res = attn_least_squares_solve(A1=A1, A2=A2, G1=G1, G2=G2, 
                                                                   X_type="Z", Y0=Y0, beta=group["max_norm_tr"], 
                                                tol=1e-10, maxit=100, diag_scaling=True)
                        else:
                            Y0, Z1_0, Z2_0 = None, None, None
 
                        # Run torch PDHG
                        # A1=W_q, A2=W_k
                        # G1=G_k, G2=G_q
                        if pdhg_max_iter >= 1:
                            Z1_t, Z2_t, residuals, norm_Y = pdhg_method_AB(
                            prox_h_conj,
                            A2=A2, A1=A1, G1=G1, G2=G2,
                            max_iter=pdhg_max_iter, lamb_max=lamb_max, beta=group["max_norm_tr"],
                            mu=mu_reg, eps_abs=1e-6, eps_rel=1e-12, stopping=False, min_iter=500,
                            Z1_0=Z1_0, Z2_0=Z2_0, Y0=Y0,
                            diag_scaling=group["diag_scaling"], acceleration=group["acceleration"], 
                            h_conj=h_conj, pd_residuals=pd_residuals_infty_ball,
                            halpern_start=group["halpern_start"], reflected_pdhg=group["reflected_pdhg"]
                        )
                        else:
                            Z1_t, Z2_t = Z1_0, Z2_0
                            residuals = res
                            r1, r1_rel, r2, r2_rel = pd_residuals_infty_ball(
                                A=A2, B=A1, Y=Y0, Z1=Z1_t, Z2=Z2_t, G1=G1, G2=G2,
                                beta=group["max_norm_tr"], mu=mu_reg)
                            residuals = {'r1': [r1], 'r2': [r2], 'r1_rel': [r1_rel], 'r2_rel': [r2_rel]}
                            norm_Y = Y0.pow(2).sum().sqrt().item()
                    elif pd_type == "fista":
                        Y_fista, Z1_t, Z2_t, residuals = fista_ls_l1_reg(
                            A2=A2, A1=A1, G1=G1, G2=G2,
                            beta=group["max_norm_tr"], mu=mu_reg, 
                            lamb_max=lamb_max, max_iter=pdhg_max_iter,
                            eps_abs=1e-6, eps_rel=1e-12, stopping=False,
                            Y0=Y0, pd_residuals=pd_residuals_infty_ball
                        )
                        norm_Y = Y_fista.pow(2).sum().sqrt().item()
                    
                    A2.data.copy_(Z1_t)
                    A1.data.copy_(Z2_t)
                    residuals["A1_norm"] = A1.norm().item()
                    residuals["A2_norm"] = A2.norm().item()
                    residuals["G1_norm"] = G1.norm().item()
                    residuals["G2_norm"] = G2.norm().item()
                    residuals["Y_norm"] = norm_Y
                    residuals_n_layers[n_att_layers] = residuals
                    n_att_layers += 1
                    
                    # apply AdamW-style update (same rule here, but you could
                    # customize Q/K vs V if desired)
                    W_v.data.mul_(1 - lr * wd)
                    W_v.data.add_(G_v, alpha=-lr / scale)
                else:
                    if not momentum:
                        buf1 = state["moment1"]
                        buf2 = state["moment2"]
                        buf1.lerp_(g, 1 - beta1)
                        buf2.lerp_(g.square(), 1 - beta2)
                        
                        g = buf1 / (eps + buf2.sqrt())
                    p.data.mul_(1 - lr * wd)
                    p.data.add_(g, alpha=-lr / scale)

        # Sanity check: only require updates for params that had gradients
        names_with_grad = {
            self.name_by_param[p]
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        }
        missing = [n for n in names_with_grad if n not in updated]
        assert not missing, f"[AttnPDAdamW] Params with grad but no update: {missing}"

        return loss, residuals_n_layers
        # return loss



def prox_l1(x, beta, R=None):
    # soft-thresholding
    # proximal operator for l1 norm beta * ||x||_1
    if R is not None:
        threshold = beta * R
    else:
        threshold = beta
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)


def mathcal_A_linop(*, A1, A2, Z): # A operator
    m = A1.shape[0]
    return Z[:m, :].T @ A1 + A2.T @ Z[m:, :]
       
        
def mathcal_A_adj_linop(*, A1, A2, Y):         # A^* operator  
    return torch.cat([A1 @ Y.T, A2 @ Y], dim=0)


def pdhg_method_AB(
    prox_h_conj,
    A2: torch.Tensor | None,
    A1: torch.Tensor,
    G1: torch.Tensor,
    G2: torch.Tensor | None,
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
    acceleration: bool = False,
    Z1_0: torch.Tensor | None = None,
    Z2_0: torch.Tensor | None = None,
    diag_scaling: bool = False,
    h_conj= None,
    f_star: float | None = None,
    pd_residuals: Optional[callable] = None,
    norm_first_iter: bool = False,
    halpern_start: int = np.inf,
    reflected_pdhg: bool = False,
    ):
    use_Z2 = not ((A2 is None) or (G2 is None) ) 
    if lamb_max is None:
        nA = A1.pow(2).sum().sqrt().item()
        if use_Z2:
            nB = A2.pow(2).sum().sqrt().item()
            lamb_max = (nA * nA + nB * nB) ** 0.5
        else:
            lamb_max = nA
        print(f"{lamb_max=}")
    rho = gamma = max(min(0.99 / (lamb_max + 1e-12), 1e4), 1e-5)
    m, n = A1.shape
    residuals = {"r1": [], "r2": [], "r1_rel": [], "r2_rel": [], "dual_vals":[], "rel_gap":[]}
    if mu == 0.0:
        residuals.pop("dual_vals")

    if Z1_0 is not None:
        Z1 = Z1_0
    else:
        Z1 = torch.zeros_like(G1)
    Z1_bar = Z1.clone()
    if Y0 is not None:
        Y = Y0
    else:
        Y = torch.zeros((Z1.shape[1], A1.shape[1]), device=G1.device, dtype=G1.dtype)
    if use_Z2:
        if Z2_0 is not None:
            Z2 = Z2_0
        else:
            Z2 = torch.zeros_like(G2) 
        Z2_bar = Z2.clone()
        Z_total_size = Z1.numel() + Z2.numel()
    else:
        Z2 = None 
        Z_total_size = Z1.numel()

    if diag_scaling:
        R, Gamma_1, Gamma_2 = pdhg_diagonal_scaling(A=A2, B=A1, eta=0.99)
    else:
        R = rho * torch.ones((Y.shape[0], Y.shape[1]), device=Y.device, dtype=Y.dtype)
        Gamma_1 = gamma * torch.ones((A1.shape[0], 1), device=A1.device, dtype=A1.dtype) 
        Gamma_2 = gamma * torch.ones((A2.shape[0], 1), device=A2.device, dtype=A2.dtype) if use_Z2 else None

    if pd_residuals is not None:
        r1, r1_rel, r2, r2_rel = pd_residuals(
                A=A2, B=A1, Y=Y, Z1=Z1, Z2=Z2, G1=G1, G2=G2,
                beta=beta, mu=mu)
        residuals["r1_rel"].append(r1_rel); residuals["r2_rel"].append(r2_rel) 
        residuals['r1'].append(r1); residuals['r2'].append(r2)
    
    for t in range(max_iter):
        if use_Z2:
            Y_new = prox_h_conj(Y + R * (Z1_bar.t() @ A1 + A2.t() @ Z2_bar), 1, R=R)
            Z1_new = (1 / (1 + Gamma_1 * mu)) * (Z1 - Gamma_1 * (A1 @ Y_new.t() + G1))
            Z2_new = (1 / (1 + Gamma_2 * mu)) * (Z2 - Gamma_2 * (A2 @ Y_new + G2))
        else: 
            Y_new = prox_h_conj(Y + R * (Z1_bar.t() @ A1), 1, R=R)  
            Z1_new = (1 / (1 + Gamma_1 * mu)) * (Z1 - Gamma_1 * (A1 @ Y_new.t() + G1))
        
        if pd_residuals is None:
            if use_Z2:
                r1 = ((1 / R) * ( Y_new - Y - R * ((Z1_bar - Z1_new).t() @ A1)
                                            - R * (A2.t() @ (Z2_bar - Z2_new)))).pow(2).sum().sqrt().item()
                r2 = (((Z1_new - Z1) / Gamma_1).pow(2).sum() + ((Z2_new - Z2) / Gamma_2).pow(2).sum()).sqrt().item()
            else:
                r1 = ((1 / R) * ( Y_new - Y - R * ((Z1_bar - Z1_new).t() @ A1))).pow(2).sum().sqrt().item()
                r2 = ((Z1_new - Z1) / Gamma_1).pow(2).sum().sqrt().item()
        else:
            r1, r1_rel, r2, r2_rel = pd_residuals(
            A=A2, B=A1, Y=Y_new, Z1=Z1_new, Z2=Z2_new, G1=G1, G2=G2,
            beta=beta, mu=mu)
            # residuals["r1_rel"].append(r1_rel); residuals["r2_rel"].append(r2_rel) 
        residuals['r1'].append(r1); residuals['r2'].append(r2)

        if t >= 1:
            if norm_first_iter:
                # normalize by first iter residual
                if t == 5:
                    norm1 = r1 if r1 > 1e-8 else 1.0
                    norm2 = r2 if r2 > 1e-8 else 1.0
                if t >= 5:
                    r1_rel = r1 / norm1 
                    r2_rel = r2 / norm2 

            if pd_residuals is None and not norm_first_iter:
                # normalize using parts of the residuals
                if use_Z2:
                    norm1 = ((Z1_new.t() @ A1 + A2.t() @ Z2_new).pow(2).sum()).sqrt().item()
                    norm2 = ((A1 @ Y_new.t()).pow(2).sum() + (A2 @ Y_new).pow(2).sum()).sqrt().item()
                else:
                    norm1 = (Z1_new.t() @ A1).pow(2).sum().sqrt().item()
                    norm2 = (A1 @ Y_new.t()).pow(2).sum().sqrt().item()
                if norm1 < 1e-6: norm1 = 1.0
                if norm2 < 1e-6: norm2 = 1.0
                r1_rel = max(1e-8, r1 - eps_abs * (Y.numel() ** 0.5)) / norm1
                r2_rel = max(1e-8, r2 - eps_abs * (Z_total_size ** 0.5) ) / norm2 
            
            if (norm_first_iter and t >= 5) or not norm_first_iter:
                residuals["r1_rel"].append(r1_rel); residuals["r2_rel"].append(r2_rel) 

        if stopping and pd_residuals is None and t > 5:
            if (r1 <= eps_abs * (Y.numel() ** 0.5) + eps_rel * norm1
                and r2 <= eps_abs * (Z_total_size ** 0.5) + eps_rel * norm2 and t >= min_iter):
                Z1 = Z1_new
                if use_Z2:
                    Z2 = Z2_new
                Y = Y_new 
                break
                
        if acceleration and not diag_scaling:
            theta = 1 / (1 + 2 * gamma * mu)**0.5
            gamma = theta * gamma 
            rho = rho / theta
            R = rho * torch.ones((Y.shape[0], Y.shape[1]), device=Y.device, dtype=Y.dtype)
            Gamma_1 = gamma * torch.ones((A2.shape[0], 1), device=A2.device, dtype=A2.dtype) 
            if use_Z2:
                Gamma_2 = gamma * torch.ones((A1.shape[0], 1), device=A1.device, dtype=A1.dtype)  

        if mu > 0 and h_conj is not None: 
            dual_val = - h_conj(Y_new) - (1/(2 * mu)) * (A1 @ Y_new.t() + G1).pow(2).sum()
            if use_Z2:
                dual_val = dual_val - (1/(2 * mu)) * (A2 @ Y_new + G2).pow(2).sum()
            residuals["dual_vals"].append(dual_val.item())
            if f_star is not None:
                residuals["rel_gap"].append( np.abs(f_star - dual_val.item()) / (abs(f_star) + 1e-12) )
        if reflected_pdhg:
            Y_new = 2 * Y_new - Y
            Z1_new = 2 * Z1_new - Z1
            if use_Z2:
                Z2_new = 2 * Z2_new - Z2
        if halpern_start == t:
            anchor_Y = Y_new.clone()
            anchor_Z1 = Z1_new.clone()
            anchor_Z2 = Z2_new.clone() if use_Z2 else None
        if halpern_start < t:
            Y_new = ((t + 1)/ (t + 2)) * Y_new + (1 / (t + 2)) * anchor_Y
            Z1_new = ((t + 1) / (t + 2)) * Z1_new + (1 / (t + 2)) * anchor_Z1
            if use_Z2:
                Z2_new = ((t + 1) / (t + 2)) * Z2_new + (1 / (t + 2)) * anchor_Z2
        Z1_bar = Z1_new + theta * (Z1_new - Z1)
        Z1 = Z1_new
        if use_Z2:  
            Z2_bar = Z2_new + theta * (Z2_new - Z2)
            Z2 = Z2_new
        Y = Y_new
    return Z1, Z2, residuals, (Y_new.pow(2).sum()).sqrt().item()



def fista_ls_l1_reg(A2: torch.Tensor | None,
                    A1: torch.Tensor,
                    G1: torch.Tensor,
                    G2: torch.Tensor | None, 
                    beta: float,
                    mu: float,
                    lamb_max=None,
                    max_iter: int = 500,
                    eps_abs: float = 1e-8,
                    eps_rel: float = 1e-8,
                    f_star: float | None = None,
                    Y0: torch.Tensor | None = None,
                    stopping: bool = True,
                    pd_residuals: Optional[Callable] = None
                    ):
    """
    Fista for solving the dual problem:
    maximize -\frac{1}{2\mu}\|\mathcal{A}^*(Y) + G\|_F^2 - \beta \|vec(Y)\|_1
    \mathcal{A}^*(Y) = [A1 Y^T; A2 Y] or A1 Y if A2 is None
    """ 
    use_Z2 = not ((A2 is None) or (G2 is None) ) 
    if lamb_max is None:
        nA = A1.pow(2).sum().sqrt().item()
        if use_Z2:
            nB = A2.pow(2).sum().sqrt().item()
            lamb_max = (nA * nA + nB * nB) ** 0.5
        else:
            lamb_max = nA
        print(f"{lamb_max=}")
    
    step_size = mu / lamb_max**2
    if Y0 is not None:
        X1 = Y0
    else:
        X1 = torch.zeros((A2.shape[1], A1.shape[1]), device=G1.device, dtype=G1.dtype)
    X0 = torch.zeros((G1.shape[1], A1.shape[1]), device=G1.device, dtype=G1.dtype)
    X1 = X0.clone()
    residuals = {"r1": [],  "r1_rel": [], "r2": [],  "r2_rel": [], "dual_vals":[], "rel_gap":[]}
    if pd_residuals is not None:
        residuals["r2"] = []; residuals["r2_rel"] = []
 
    for t in range(1, max_iter+1):
        # Extrapolation step
        Y = X1 + ((t - 2) / (t + 1)) * (X1 - X0)
        # Gradient step
        grad = (1/mu) * (A1 @ Y.T + G1).T @ A1 
        if use_Z2:
            grad = grad + (1/mu) * (A2.T @ (A2 @ Y + G2)) 
        # Prox step
        Y1 = prox_l1(Y - step_size * grad, beta * step_size) 
        if pd_residuals is None:
            # FPI residual
            normalize = Y.pow(2).sum().sqrt().item() 
            normalize = normalize if normalize > 1e-6 else 1.0
            r1 = (Y1 - Y).pow(2).sum().sqrt().item() / step_size
            r1_rel = max(1e-12, r1 - eps_abs * (Y.numel() ** 0.5)) / normalize 
        else:
            # primal recovery
            Z1_fista = (1 / mu) * (- G1 - A1 @ Y1.T)
            Z2_fista = (1 / mu) * (- G2 - A2 @ Y1)

            r1, r1_rel, r2, r2_rel = pd_residuals(
                A=A2,
                B=A1,
                Y=Y1,
                Z1=Z1_fista,
                Z2=Z2_fista,
                G1=G1,
                G2=G2,
                beta=beta,
                mu=mu,
            ) 
            residuals['r2'].append(r2); residuals['r2_rel'].append(r2_rel) 
        residuals['r1'].append(r1); residuals['r1_rel'].append(r1_rel)

        X0, X1 = X1, Y1
        
        if use_Z2:
            loss = (
                -(1.0/(2*mu)) * (A1 @ X1.T + G1).pow(2).sum()
                -(1.0/(2*mu)) * (A2 @ X1   + G2).pow(2).sum()
                - beta * torch.sum(torch.abs(X1))).item()
        else:
            loss = (
                -(1.0/(2*mu)) * (A1 @ X1.T + G1).pow(2).sum()
                - beta * torch.sum(torch.abs(X1))).item()
        residuals['dual_vals'].append(loss)
        if f_star is not None:
            rel_gap = abs(f_star - loss) / (abs(f_star) + 1e-12)
            residuals['rel_gap'].append(rel_gap)

        if stopping and t >= 2 and pd_residuals is None and \
            r1_rel < eps_abs * (Y.numel() ** 0.5) + eps_rel * Y.pow(2).sum().sqrt().item():
            print(f"Fista converged in {t} iterations.")
            break 
    if pd_residuals is None:
        # primal recovery
        Z1_fista = (1 / mu) * (- G1 - A1 @ Y1.T)
        Z2_fista = (1 / mu) * (- G2 - A2 @ Y1)
    return X1, Z1_fista, Z2_fista, residuals


def proj_subgrad_l1(AZ, Y, beta=1, abs_tol=1e-4):
    # \min_S \beta\|AZ/\beta - S\|_F s.t. S \in \partial \|\vec(Y)\|_1 
    S = torch.sign(Y)
    S[Y == 0] = torch.clamp((AZ[Y == 0]/beta), -1.0, 1.0)
    r = beta * (AZ / beta - S).pow(2).sum().sqrt().item()
    # norm = AZ.pow(2).sum().sqrt().item()
    # if norm < 1e-6: norm = 1.0
    norm = beta * (1 + abs_tol) * (S.numel()**0.5)
    return r, r / norm


def pd_residuals_infty_ball(A, B, Y, Z1, Z2, G1, G2, beta, mu=0, abs_tol=1e-4):
    # KKT residuals 
    # h = I_{||.||_\max \leq beta}
    # 0 \in \partial h^*(Y) - \mathcal{A}(Z)   -- primal residual
    # 0 = G + \mu Z + \mathcal{A}^*(Y).        -- dual residual
    AZ = Z1.T @ B + A.T @ Z2 
    r1, r1_rel = proj_subgrad_l1(AZ, Y, beta=beta, abs_tol=abs_tol)
    r2_1 = (G1 + mu * Z1 + B @ Y.t()).pow(2).sum().sqrt().item()
    r2_2 = (G2 + mu * Z2 + A @ Y).pow(2).sum().sqrt().item()
    r2 = (r2_1**2 + r2_2**2)**0.5 
    norm2 = (G1.pow(2).sum() + G2.pow(2).sum()).sqrt().item() + abs_tol * (2 * G1.numel())**0.5
    if norm2 < 1e-6: norm2 = 1.0
    r2_rel = r2 / norm2 
    return r1, r1_rel, r2, r2_rel


def pd_residuals_max_ball(A1, A2, Y, Z, G1, G2, beta, mu=0, abs_tol=1e-4):
    m = A1.shape[0]
    Z1, Z2 = Z[:m, :], Z[m:, :] 
    return pd_residuals_infty_ball(B=A1, A=A2, Y=Y, Z1=Z1, Z2=Z2, G1=G1, G2=G2, 
                                   beta=beta, mu=mu, abs_tol=abs_tol)



