
import torch
from torch.optim import Optimizer
import numpy as np
from typing import Optional, Callable



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
        pdhg_momentum=False,
        diag_scaling=False,
        acceleration=False,
        pd_type="pdhg",
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
            pdhg_momentum=pdhg_momentum,
            diag_scaling=diag_scaling,
            acceleration=acceleration,
            pd_type=pd_type,
        )
        print(
            f"[AttnPDAdamW] lr={lr}, betas={betas}, eps={eps}, wd={weight_decay}, "
            f"qk_lr_scale={qk_lr_scale}, max_norm_tr={max_norm_tr}, pdhg_iters={pdhg_max_iter}, "
            f"\n    momentum={pdhg_momentum}, diag_scaling={diag_scaling}, accel={acceleration}, {pd_type=}"
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
            momentum = group["pdhg_momentum"]
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
                    W_q, G_q = p[:n_embed, :],                  g[:n_embed, :]
                    W_k, G_k = p[n_embed:2 * n_embed, :],       g[n_embed:2 * n_embed, :]
                    W_v, G_v = p[2 * n_embed: 3 * n_embed, :],  g[2 * n_embed: 3 * n_embed, :]
                    # Y0, dual_res = check_dual_feasible(W_k, W_q, G_k, G_q, maxit=10000)
                    # print(f"{dual_res=}, ||Y0||_F={Y0.pow(2).sum().item():.4e}")
                    # Y0 = torch.randn(W_q.shape[0], W_k.shape[0], device=W_q.device, dtype=W_q.dtype) * 1e-3
                    # print(f"|Y0||_F={Y0.pow(2).sum().item():.4e}")
                    nA = W_k.pow(2).sum().sqrt().item()
                    nB = W_q.pow(2).sum().sqrt().item()
                    lamb_max = (nA * nA + nB * nB) ** 0.5
                    mu_max = (G_k.t() @ W_q + W_k.t() @ G_q).abs().max().item() / group["max_norm_tr"]
                    mu_reg = max(0.1 * mu_max, 1e-6)
                    print(f"{lamb_max=:.4e}, {mu_reg=:.4e}, {mu_max=:.4e}")
                    # PDHG to update key-query  
                    prox_h_conj = lambda y, rho, R: prox_l1(y, rho * group["max_norm_tr"], R=R)
                    h_conj = lambda y: group["max_norm_tr"] * torch.abs(y).sum()
                    assert W_q.shape == G_q.shape and W_k.shape == G_k.shape, \
                        print(f"{W_q.shape=}, {G_q.shape=}, {W_k.shape=}, {G_k.shape=}")
                    Y0 = 0.001 * torch.randn((W_k.shape[1], W_q.shape[1]), device=W_k.device, dtype=W_k.dtype)
                    # Y0 = None
                    if pd_type == "pdhg": 
                        # Z1_0=-G_k; Z2_0=-G_q
                        Z1_0, Z2_0 = None, None
                        # Run torch PDHG
                        Z1_t, Z2_t, residuals, norm_Y = pdhg_method_AB(
                            prox_h_conj,
                            W_k=W_k, W_q=W_q, G_wk=G_k, G_wq=G_q,
                            max_iter=pdhg_max_iter, lamb_max=lamb_max, beta=group["max_norm_tr"],
                            mu=mu_reg, eps_abs=1e-6, eps_rel=1e-12, stopping=False, min_iter=500,
                            Z1_0=Z1_0, Z2_0=Z2_0, Y0=Y0,
                            diag_scaling=group["diag_scaling"], acceleration=group["acceleration"], 
                            h_conj=h_conj, pd_residuals=pd_residuals_infty_ball
                        )
                    elif pd_type == "fista":
                        Y_fista, Z1_t, Z2_t, residuals = fista_ls_l1_reg(
                            W_k=W_k, W_q=W_q, G_wk=G_k, G_wq=G_q,
                            beta=group["max_norm_tr"], mu=mu_reg, 
                            lamb_max=lamb_max, max_iter=pdhg_max_iter,
                            eps_abs=1e-6, eps_rel=1e-12, stopping=False,
                            Y0=Y0, pd_residuals=pd_residuals_infty_ball
                        )
                        norm_Y = Y_fista.pow(2).sum().sqrt().item()
                    
                    W_k.data.copy_(Z1_t)
                    W_q.data.copy_(Z2_t)
                    residuals["W_q_norm"] = W_q.norm().item()
                    residuals["W_k_norm"] = W_k.norm().item()
                    residuals["G_k_norm"] = G_k.norm().item()
                    residuals["G_q_norm"] = G_q.norm().item()
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



def pdhg_method_AB(
    prox_h_conj,
    W_k: torch.Tensor | None,
    W_q: torch.Tensor,
    G_wk: torch.Tensor,
    G_wq: torch.Tensor | None,
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
    pd_residuals: Optional[callable] = None
    ):
    use_Z2 = not ((W_k is None) or (G_wq is None) ) 
    if lamb_max is None:
        nA = W_q.pow(2).sum().sqrt().item()
        if use_Z2:
            nB = W_k.pow(2).sum().sqrt().item()
            lamb_max = (nA * nA + nB * nB) ** 0.5
        else:
            lamb_max = nA
        print(f"{lamb_max=}")
    rho = gamma = max(min(0.99 / (lamb_max + 1e-12), 1e4), 1e-5)
    m, n = W_q.shape
    residuals = {"r1": [], "r2": [], "r1_rel": [], "r2_rel": [], "dual_vals":[], "rel_gap":[]}
    if mu == 0.0:
        residuals.pop("dual_vals")

    # matrix_details(G_wk)
    # if use_Z2: matrix_details(G_wq)
 
    if Z1_0 is not None:
        Z1 = Z1_0
    else:
        Z1 = torch.zeros_like(G_wk)
    Z1_bar = Z1.clone()
    if Y0 is not None:
        Y = Y0
    else:
        Y = torch.zeros((Z1.shape[1], W_q.shape[1]), device=G_wk.device, dtype=G_wk.dtype)
    if use_Z2:
        if Z2_0 is not None:
            Z2 = Z2_0
        else:
            Z2 = torch.zeros_like(G_wq) 
        Z2_bar = Z2.clone()
        Z_total_size = Z1.numel() + Z2.numel()
    else:
        Z2 = None 
        Z_total_size = Z1.numel()

    if diag_scaling:
        R, Gamma_1, Gamma_2 = pdhg_diagonal_scaling(A=W_k, B=W_q, eta=0.99)
    else:
        R = rho * torch.ones((Y.shape[0], Y.shape[1]), device=Y.device, dtype=Y.dtype)
        Gamma_1 = gamma * torch.ones((W_q.shape[0], 1), device=W_q.device, dtype=W_q.dtype) 
        Gamma_2 = gamma * torch.ones((W_k.shape[0], 1), device=W_k.device, dtype=W_k.dtype) if use_Z2 else None

    for t in range(max_iter):
        if use_Z2:
            Y_new = prox_h_conj(Y + R * (Z1_bar.t() @ W_q + W_k.t() @ Z2_bar), 1, R=R)
            Z1_new = (1 / (1 + Gamma_1 * mu)) * (Z1 - Gamma_1 * (W_q @ Y_new.t() + G_wk))
            Z2_new = (1 / (1 + Gamma_2 * mu)) * (Z2 - Gamma_2 * (W_k @ Y_new + G_wq))
        else: 
            Y_new = prox_h_conj(Y + R * (Z1_bar.t() @ W_q), 1, R=R)  
            Z1_new = (1 / (1 + Gamma_1 * mu)) * (Z1 - Gamma_1 * (W_q @ Y_new.t() + G_wk))
        
        if pd_residuals is None:
            if use_Z2:
                r1 = ((1 / R) * ( Y_new - Y - R * ((Z1_bar - Z1_new).t() @ W_q)
                                            - R * (W_k.t() @ (Z2_bar - Z2_new)))).pow(2).sum().sqrt().item()
                r2 = (((Z1_new - Z1) / Gamma_1).pow(2).sum() + ((Z2_new - Z2) / Gamma_2).pow(2).sum()).sqrt().item()
            else:
                r1 = ((1 / R) * ( Y_new - Y - R * ((Z1_bar - Z1_new).t() @ W_q))).pow(2).sum().sqrt().item()
                r2 = ((Z1_new - Z1) / Gamma_1).pow(2).sum().sqrt().item()
        else:
            r1, r1_rel, r2, r2_rel = pd_residuals(
            A=W_k, B=W_q, Y=Y_new, Z1=Z1_new, Z2=Z2_new, G1=G_wk, G2=G_wq,
            beta=beta, mu=mu)
        residuals['r1'].append(r1); residuals['r2'].append(r2)


        if t >= 1:
            if pd_residuals is None:
                if use_Z2:
                    norm1 = ((Z1_new.t() @ W_q + W_k.t() @ Z2_new).pow(2).sum()).sqrt().item()
                    norm2 = ((W_q @ Y_new.t()).pow(2).sum() + (W_k @ Y_new).pow(2).sum()).sqrt().item()
                else:
                    norm1 = (Z1_new.t() @ W_q).pow(2).sum().sqrt().item()
                    norm2 = (W_q @ Y_new.t()).pow(2).sum().sqrt().item()
                if norm1 < 1e-6: norm1 = 1.0
                if norm2 < 1e-6: norm2 = 1.0
                r1_rel = max(1e-12, r1 - eps_abs * (Y.numel() ** 0.5)) / norm1
                r2_rel = max(1e-12, r2 - eps_abs * (Z_total_size ** 0.5) ) / norm2 
            residuals["r1_rel"].append(r1_rel); residuals["r2_rel"].append(r2_rel)
            if stopping and pd_residuals is None:
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
            Gamma_1 = gamma * torch.ones((W_k.shape[0], 1), device=W_k.device, dtype=W_k.dtype) 
            if use_Z2:
                Gamma_2 = gamma * torch.ones((W_q.shape[0], 1), device=W_q.device, dtype=W_q.dtype)  

        if mu > 0 and h_conj is not None: 
            dual_val = - h_conj(Y_new) - (1/(2 * mu)) * (W_q @ Y_new.t() + G_wk).pow(2).sum()
            if use_Z2:
                dual_val = dual_val - (1/(2 * mu)) * (W_k @ Y_new + G_wq).pow(2).sum()
            residuals["dual_vals"].append(dual_val.item())
            if f_star is not None:
                residuals["rel_gap"].append( np.abs(f_star - dual_val.item()) / (abs(f_star) + 1e-12) )

        Z1_bar = Z1_new + theta * (Z1_new - Z1)
        Z1 = Z1_new
        if use_Z2:  
            Z2_bar = Z2_new + theta * (Z2_new - Z2)
            Z2 = Z2_new
        Y = Y_new
    return Z1, Z2, residuals, (Y_new.pow(2).sum()).sqrt().item()


torch.no_grad()
def check_dual_feasible(A, B, Gk, Gq, max_kron=1_000, verbose=True, maxit=10000, tol=1e-10):
    """
    Check dual feasibility: ∃ Y s.t. A Y = Gq and Y B^T = Gk^T.

    We solve (A^T A)Y + Y(B^T B) = A^T Gq + Gk^T B for Y (LS optimum).
    residual = sqrt( ||A Y + Gq||_F^2 + ||Y B^T + Gk^T||_F^2 ).
    """ 
    device = A.device
    dtype  = A.dtype
    m, p = A.shape
    _m2, n = B.shape
    assert _m2 == m, "A and B must have same number of rows"
    assert Gk.shape == A.shape and Gq.shape == B.shape, "G shapes must match A,B"

    AtA = A.T @ A              # [p x p]
    BtB = B.T @ B              # [n x n]
    C   = (A.T @ Gq + Gk.T @ B)  

    # Small/medium case: solve via Kronecker.
    if p * n <= max_kron:
        Ip = torch.eye(p, device=device, dtype=dtype)
        In = torch.eye(n, device=device, dtype=dtype)
        # K = I x AtA + (BtB) x I
        K = torch.kron(In, AtA) + torch.kron(BtB, Ip)  # [(pn) x (pn)]
        y = torch.linalg.solve(K, C.T.reshape(-1))
        Y = y.reshape(n, p).T
    else:
        # Large case: solve via CG
        def M(Y):         # normal operator  M(Y) = AtA Y + Y BtB
            return AtA @ Y + Y @ BtB

        dA = torch.diag(AtA).clamp_min(1e-12).view(p, 1)
        dB = torch.diag(BtB).clamp_min(1e-12).view(1, n)

        def Minv(R):      # elementwise divide by dA + dB
            return R / (dA + dB)

        # Initialization
        Y  = torch.zeros((p, n), device=device, dtype=dtype)
        R  = C - M(Y)              # residual
        Z  = Minv(R)               # preconditioned residual
        P  = Z.clone()
 
        rz_old = (R * Z).sum()
        normC  = C.pow(2).sum().sqrt().item()
        if normC == 0.0: normC = 1.0

        for k in range(1, maxit+1):
            MP = M(P)
            denom = (P * MP).sum()
            # Guard against breakdown
            if abs(denom) < 1e-20:
                if verbose: print(f"PCG breakdown at iter {k}")
                break

            alpha = (rz_old / denom).item()
            Y = Y + alpha * P
            R = R - alpha * MP

            if R.pow(2).sum().sqrt().item() < tol * normC:
                break

            Z = Minv(R)
            rz_new = (R * Z).sum()
            beta = (rz_new / rz_old).item()
            P = Z + beta * P
            rz_old = rz_new

    # Compute residuals of the *original* equations
    RA = A @ Y - Gq              # should be 0 if feasible
    RB = Y @ B.T - Gk.T          # should be 0 if feasible
    res = torch.sqrt(RA.pow(2).sum() + RB.pow(2).sum()).item() / normC
 
    return Y, res



def pdhg_diagonal_scaling(A, B, eta=0.99, eps=1e-8, debug=False):
    device, dtype = A.device, A.dtype
    p2, n = A.shape
    p1, nB = B.shape 

    # |B| row/col sums 
    r_B = B.abs().sum(dim=1)                 # (p1,)
    c_B = B.abs().sum(dim=0)                 # (n,)

    # |A| row/col sums 
    r_A = A.abs().sum(dim=1)                 # (p2,)
    c_A = A.abs().sum(dim=0)                 # (n,)
 
    inv_rB = torch.where(r_B > 0, 1.0 / (r_B + eps), torch.zeros_like(r_B))
    inv_rA = torch.where(r_A > 0, 1.0 / (r_A + eps), torch.zeros_like(r_A))
    inv_c_sum = torch.where(
        (c_B[None, :] + c_A[:, None]) > 0, 1.0 / (c_B[None, :] + c_A[:, None] + eps),
        torch.zeros((n, n), device=device, dtype=dtype))

    # Diagonal entries as broadcastable tensors: 
    R       = eta * inv_c_sum                     # (n, n)   for Y
    Gamma_1 = eta * inv_rB[:, None]               # (p1, 1)  for Z1 (same across its n cols)
    Gamma_2 = eta * inv_rA[:, None]               # (p2, 1)  for Z2 (same across its n cols)

    if debug:
        assert torch.allclose(1/inv_c_sum, torch.ones(n, 1) @ c_B[None, :] + c_A[:, None] @ torch.ones(1, n)), "Column sum mismatch!"
        print("Diagonal PDHG scaling computed.")
        print(f"{R.mean().item():.4e} +- {R.std().item():.4e}, "
            f"{Gamma_1.mean().item():.4e} +- {Gamma_1.std().item():.4e}, "
            f"{Gamma_2.mean().item():.4e} +- {Gamma_2.std().item():.4e}")
        matrix_details(A)
        matrix_details(B)
    return R, Gamma_1, Gamma_2 


def matrix_details(A: torch.Tensor | None):
    if A is None:
        print("Matrix details: A=None (skipped)")
        return
    rank_tol = torch.linalg.matrix_rank(A, tol=1e-6)
    sigma_max = torch.linalg.norm(A, ord=2)
    fro_norm = torch.linalg.norm(A, ord='fro')
    print(f"{A.shape=}, {rank_tol=:.4e}, {sigma_max=:.4e}, {fro_norm=:.4e}")


def fista_ls_l1_reg(W_k: torch.Tensor | None,
                    W_q: torch.Tensor,
                    G_wk: torch.Tensor,
                    G_wq: torch.Tensor | None, 
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
    \mathcal{A}^*(Y) = [W_q Y^T; W_k Y] or W_q Y if W_k is None
    """ 
    use_Z2 = not ((W_k is None) or (G_wq is None) ) 
    if lamb_max is None:
        nA = W_q.pow(2).sum().sqrt().item()
        if use_Z2:
            nB = W_k.pow(2).sum().sqrt().item()
            lamb_max = (nA * nA + nB * nB) ** 0.5
        else:
            lamb_max = nA
        print(f"{lamb_max=}")
    
    step_size = mu / lamb_max**2
    if Y0 is not None:
        X1 = Y0
    else:
        X1 = torch.zeros((W_k.shape[1], W_q.shape[1]), device=G_wk.device, dtype=G_wk.dtype)
    X0 = torch.zeros((G_wk.shape[1], W_q.shape[1]), device=G_wk.device, dtype=G_wk.dtype)
    X1 = X0.clone()
    residuals = {"r1": [],  "r1_rel": [], "r2": [],  "r2_rel": [], "dual_vals":[], "rel_gap":[]}
    if pd_residuals is not None:
        residuals["r2"] = []; residuals["r2_rel"] = []
 
    for t in range(1, max_iter+1):
        # Extrapolation step
        Y = X1 + ((t - 2) / (t + 1)) * (X1 - X0)
        # Gradient step
        grad = (1/mu) * (W_q @ Y.T + G_wk).T @ W_q 
        if use_Z2:
            grad = grad + (1/mu) * (W_k.T @ (W_k @ Y + G_wq)) 
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
            Z1_fista = (1 / mu) * (- G_wk - W_q @ Y1.T)
            Z2_fista = (1 / mu) * (- G_wq - W_k @ Y1)

            r1, r1_rel, r2, r2_rel = pd_residuals(
                A=W_k,
                B=W_q,
                Y=Y1,
                Z1=Z1_fista,
                Z2=Z2_fista,
                G1=G_wk,
                G2=G_wq,
                beta=beta,
                mu=mu,
            ) 
            residuals['r2'].append(r2); residuals['r2_rel'].append(r2_rel) 
        residuals['r1'].append(r1); residuals['r1_rel'].append(r1_rel)

        X0, X1 = X1, Y1
        
        if use_Z2:
            loss = (
                -(1.0/(2*mu)) * (W_q @ X1.T + G_wk).pow(2).sum()
                -(1.0/(2*mu)) * (W_k @ X1   + G_wq).pow(2).sum()
                - beta * torch.sum(torch.abs(X1))).item()
        else:
            loss = (
                -(1.0/(2*mu)) * (W_q @ X1.T + G_wk).pow(2).sum()
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
        Z1_fista = (1 / mu) * (- G_wk - W_q @ Y1.T)
        Z2_fista = (1 / mu) * (- G_wq - W_k @ Y1)
    return X1, Z1_fista, Z2_fista, residuals


def proj_subgrad_l1(AZ, Y):
    # \min_S \|AZ - S\|_F s.t. S \in \partial \|\vec(Y)\|_1 
    S = torch.sign(Y)
    S[Y == 0] = torch.clamp(AZ[Y == 0], -1.0, 1.0)
    r = (AZ - S).pow(2).sum().sqrt().item()
    norm = max(AZ.pow(2).sum().sqrt().item(), S.pow(2).sum().sqrt().item())
    if norm < 1e-6: norm = 1.0
    return r, r / norm


def pd_residuals_infty_ball(A, B, Y, Z1, Z2, G1, G2, beta, mu):
    # KKT residuals 
    # h = I_{||.||_\max \leq beta}
    # 0 \in \partial h^*(Y) - \mathcal{A}(Z)
    # 0 = G + \mu Z + \mathcal{A}^*(Y)
    AZ = Z1.t() @ B + A.t() @ Z2 
    r1, r1_rel = proj_subgrad_l1(AZ / beta, Y)
    r2_1 = (G1 + mu * Z1 + B @ Y.t()).pow(2).sum().sqrt().item()
    r2_2 = (G2 + mu * Z2 + A @ Y).pow(2).sum().sqrt().item()
    r2 = (r2_1**2 + r2_2**2)**0.5 
    norm2 = (G1.pow(2).sum() + G2.pow(2).sum()).sqrt().item() 
    if norm2 < 1e-6: norm2 = 1.0
    r2_rel = r2 / norm2 
    return r1, r1_rel, r2, r2_rel
