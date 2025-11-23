
import torch
from torch.optim import Optimizer



class AttnPDHGAdamW(Optimizer):
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
                    nA = torch.linalg.norm(W_k, ord="fro").item()
                    nB = torch.linalg.norm(W_q, ord="fro").item()
                    lamb_max = (nA * nA + nB * nB) ** 0.5
                    mu_reg = max(1e-6 * lamb_max**2, 1e-6)
                    # PDHG to update key-query 
                    prox_h_conj = lambda y, rho: prox_l1(y, rho * group["max_norm_tr"])
                    assert W_q.shape == G_q.shape and W_k.shape == G_k.shape, \
                        print(f"{W_q.shape=}, {G_q.shape=}, {W_k.shape=}, {G_k.shape=}")

                    # Run torch PDHG
                    Z1_t, Z2_t, residuals, norm_Y = pdhg_method_AB(
                        prox_h_conj,
                        W_k=W_k,
                        W_q=W_q,
                        G_wk=G_k,
                        G_wq=G_q,
                        max_iter=500,
                        lamb_max=lamb_max,
                        mu=mu_reg,
                        eps_abs=1e-6,
                        eps_rel=1e-12,
                        stopping=False,
                        min_iter=500,
                        # Y0=Y0
                    )
                    
                    W_k.data.copy_(Z1_t)
                    W_q.data.copy_(Z2_t)
                    residuals["W_q_norm"] = W_q.norm().item()
                    residuals["W_k_norm"] = W_k.norm().item()
                    residuals["G_k_norm"] = G_k.norm().item()
                    residuals["G_q_norm"] = G_q.norm().item()
                    residuals["Y_norm"] = norm_Y
                    residuals_n_layers[n_att_layers] = residuals
                    n_att_layers += 1
                    
                    assert W_k.shape == Z1_t.shape and W_q.shape == G_q.shape and W_k.shape == G_k.shape, \
                        print(f"{W_q.shape=}, {W_k.shape=}, {G_q.shape=}, {G_k.shape=}, {Z1_t.shape=}, {Z2_t.shape=}")
                    # apply AdamW-style update (same rule here, but you could
                    # customize Q/K vs V if desired)
                    W_v.data.mul_(1 - lr * wd)
                    W_v.data.add_(G_v, alpha=-lr / scale)
                else:
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
        assert not missing, f"[AttnPDHGAdamW] Params with grad but no update: {missing}"

        return loss, residuals_n_layers



def prox_l1(x, rho):
    # proximal operator for l1 norm rho * ||x||_1
    return torch.sign(x) * torch.clamp(torch.abs(x) - rho, min=0.0)



def pdhg_method_AB(
    prox_h_conj,
    W_k: torch.Tensor | None,
    W_q: torch.Tensor,
    G_wk: torch.Tensor,
    G_wq: torch.Tensor | None,
    mu: float = 0.0,
    lamb_max=None,
    max_iter: int = 100,
    eps_abs: float = 1e-3,
    eps_rel: float = 1e-3,
    stopping: bool = False,
    min_iter: int = 10,
    Y0: torch.Tensor | None = None,
):
    use_Z2 = not ((W_k is None) or (G_wq is None) ) 
    if lamb_max is None:
        nA = torch.linalg.norm(W_q, ord="fro").item()
        if use_Z2:
            nB = torch.linalg.norm(W_k, ord="fro").item()
            lamb_max = (nA * nA + nB * nB) ** 0.5
        else:
            lamb_max = nA
        print(f"{lamb_max=}")
    rho = gamma = 0.99 / lamb_max
    m, n = W_q.shape
    residuals = {"r1": [], "r2": [], "r1_rel": [], "r2_rel": []}
 
    Z1 = torch.zeros_like(G_wk)
    Z1_bar = Z1.clone()
    if Y0 is not None:
        Y = Y0
    else:
        Y = torch.zeros((Z1.shape[1], W_q.shape[1]), device=G_wk.device, dtype=G_wk.dtype)
    if use_Z2:
        Z2 = torch.zeros_like(G_wq) 
        Z2_bar = Z2.clone()
        Z_total_size = Z1.numel() + Z2.numel()
    else:
        Z2 = None 
        Z_total_size = Z1.numel()

    for t in range(max_iter):
        if use_Z2:
            Y_new = prox_h_conj(Y + rho * (Z1_bar.t() @ W_q + W_k.t() @ Z2_bar), rho)
            Z1_new = (1 / (1 + gamma * mu)) * (Z1 - gamma * (W_q @ Y_new.t() + G_wk))
            Z2_new = (1 / (1 + gamma * mu)) * (Z2 - gamma * (W_k @ Y_new + G_wq))
        else: 
            Y_new = prox_h_conj(Y + rho * Z1_bar.t() @ W_q, rho)  
            Z1_new = (1 / (1 + gamma * mu)) * (Z1 - gamma * (W_q @ Y_new.t() + G_wk))
        
        if use_Z2:
            r1 = ( Y_new - Y - rho * ((Z1_bar - Z1_new).t() @ W_q)
                    - rho * (W_k.t() @ (Z2_bar - Z2_new))).norm().item() / rho
            r2 = ((Z1_new - Z1).norm().pow(2) + (Z2_new - Z2).norm().pow(2)).sqrt().item() / gamma
        else:
            r1 = ( Y_new - Y - rho * ((Z1_bar - Z1_new).t() @ W_q)).norm().item() / rho
            r2 = (Z1_new - Z1).norm().item() / gamma
        residuals["r1"].append(r1)
        residuals["r2"].append(r2)

        if t >= 1:
            if use_Z2:
                norm1 = ((Z1_new.t() @ W_q + W_k.t() @ Z2_new).pow(2).sum()).sqrt()
                norm2 = ((W_q @ Y_new.t()).pow(2).sum() + (W_k @ Y_new).pow(2).sum()).sqrt()
            else:
                norm1 = (Z1_new.t() @ W_q).pow(2).sum().sqrt()
                norm2 = (W_q @ Y_new.t()).pow(2).sum().sqrt()
            if norm1 < 1e-6: norm1 = 1.0
            if norm2 < 1e-6: norm2 = 1.0
            r1_rel = max(1e-8, r1 - eps_abs * (Y.numel() ** 0.5)) / norm1
            r2_rel = max(1e-8, r2 - eps_abs * (Z_total_size ** 0.5) ) / norm2 
            residuals["r1_rel"].append(r1_rel)
            residuals["r2_rel"].append(r2_rel)
            if stopping:
                if (r1 <= eps_abs * (Y.numel() ** 0.5) + eps_rel * norm1
                    and r2 <= eps_abs * (Z_total_size ** 0.5) + eps_rel * norm2 and t >= min_iter):
                    Z1 = Z1_new
                    if use_Z2:
                        Z2 = Z2_new
                    Y = Y_new 
                    break
                
        Z1_bar = 2 * Z1_new - Z1
        Z1 = Z1_new
        if use_Z2:  
            Z2_bar = 2 * Z2_new - Z2
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
        normC  = C.norm().item()
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

            if R.norm().item() < tol * normC:
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
