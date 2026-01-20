
import torch
from torch.optim import Optimizer
import numpy as np
from typing import Optional, Callable, Literal, Any
from .least_squares import Y_dual_feasible, attn_least_squares_solve

from .fast_pdhg import *
from .fista import *
from .attn_utils import *
from .linop import *



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
        rho_over_lr=10,         # ratio rho / lr
        attn_max_iter=100,
        momentum=False,
        attn_momentum="none",
        diag_scaling=False, 
        pd_type="fista",
        reflected_halpern: bool = False,
        warm_start: bool = False,
        enable_restart: bool = False,
        lsqr_max_iter: int = 500, 
        mu_frac: float = 0.1, # fraction of mu_max, mu = mu_frac * mu_max
        bias_correction: bool = True
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
            rho_over_lr=rho_over_lr,
            attn_max_iter=attn_max_iter, 
            momentum=momentum,
            attn_momentum=attn_momentum,
            diag_scaling=diag_scaling, 
            pd_type=pd_type, 
            warm_start=warm_start,
            reflected_halpern=reflected_halpern,
            enable_restart=enable_restart,
            lsqr_max_iter=lsqr_max_iter,
            mu_frac=mu_frac,
            bias_correction=bias_correction
        )
        print(
            f"[AttnPDAdamW] lr={lr}, {betas=}, {eps=}, wd={weight_decay}, "
            f"{rho_over_lr=}, {attn_max_iter=}, {warm_start=}, {lsqr_max_iter=}"
            f"{attn_momentum=}, {diag_scaling=}, {pd_type=}, {reflected_halpern=}, {enable_restart=}"
        )
        super().__init__(params, defaults)

    def _is_attn_qkv_weight(self, name: str) -> bool:
        return ("c_attn" in name) and name.endswith("weight")

    @torch.no_grad()
    def step(self, closure=None):
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
            bias_correction = group["bias_correction"]
            attn_momentum = group["attn_momentum"]
            
            for p in group["params"]:  
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    if bias_correction:
                        state["moment1"] = torch.zeros_like(g)
                        state["moment2"] = torch.zeros_like(g)
                    else: 
                        state["moment1"] = g.clone()
                        state["moment2"] = g.square()
                state["step"] += 1
                step = state["step"]

                buf1 = state["moment1"]
                buf2 = state["moment2"]
                
                if bias_correction:
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                else:
                    bias_correction1 = 1.0
                    bias_correction2 = 1.0
                scale = bias_correction1 / bias_correction2**0.5
 
                name = self.name_by_param[p]
                updated.add(name) 
                if self._is_attn_qkv_weight(name) and p.ndim == 2:
                    n_embed = p.shape[1] 
                    # choose input to max-grad-update for Q/K only 
                    if attn_momentum == "none":
                        g_for_mgu = g
                    elif attn_momentum == "prior_m":
                        buf1[:2*n_embed].lerp_(g[:2*n_embed], 1 - beta1) 
                        g[:2*n_embed].copy_(buf1[:2*n_embed] / bias_correction1)
                        g_for_mgu = g                  # M
                    elif attn_momentum == "prior_mv":
                        buf1[:2*n_embed].lerp_(g[:2*n_embed], 1 - beta1)
                        buf2[:2*n_embed].lerp_(g[:2*n_embed].square(), 1 - beta2)
                        g[:2*n_embed] = (buf1[:2*n_embed] / (buf2[:2*n_embed].sqrt() + eps)) / scale  # M / sqrt(V)
                        g_for_mgu = g
                    elif attn_momentum == "post":
                        g_for_mgu = g
                    else:
                        raise ValueError(f"Unknown attn_momentum option: {attn_momentum}")

                    Z1_t, Z2_t,residuals = self._update_kq_weights(p, g_for_mgu, group)

                    # Update weights in place
                    A1, A2   = p[:n_embed, :], p[n_embed:2 * n_embed, :]

                    if attn_momentum == "post": 
                        # the order is [Q, K, V]
                        # A1 = W_q, A2 = W_k
                        # Z2 = \Delta W_q, Z1 = \Delta W_k
                        m_qk, v_qk = buf1[:2*n_embed, :], buf2[:2*n_embed, :]                    # QK moments
                        m_qk[:n_embed].lerp_(Z2_t, 1 - beta1)
                        v_qk[:n_embed].lerp_(Z2_t.square(), 1 - beta2) 
                        m_qk[n_embed:2*n_embed].lerp_(Z1_t, 1 - beta1)
                        v_qk[n_embed:2*n_embed].lerp_(Z1_t.square(), 1 - beta2)

                        g[:2*n_embed, :] = m_qk / (eps + v_qk.sqrt()) 
                        A2.data.add_(g[n_embed:2 * n_embed, :], alpha=-lr / scale)
                        A1.data.add_(g[:n_embed, :],            alpha=-lr / scale)
                    else:
                        # W^{k+1} = W^k + \Delta W
                        A2.data.add_(Z1_t)
                        A1.data.add_(Z2_t)

                    del Z1_t, Z2_t
                    residuals["A1_norm"] = A1.norm().item()
                    residuals["A2_norm"] = A2.norm().item()
                    residuals_n_layers[n_att_layers] = residuals
                    n_att_layers += 1
                    
                    # apply AdamW-style update (same rule here, but you could
                    # customize Q/K vs V if desired)
                    n_embed = p.shape[1]   
                    buf1[2*n_embed : 3*n_embed].lerp_(g[2*n_embed : 3*n_embed], 1 - beta1)
                    buf2[2*n_embed : 3*n_embed].lerp_(g[2*n_embed : 3*n_embed].square(), 1 - beta2)
                    G_v = buf1[2*n_embed : 3*n_embed] / (eps + buf2[2*n_embed : 3*n_embed].sqrt())
                    W_v = p[2 * n_embed: 3 * n_embed, :] 
                    W_v.data.mul_(1 - lr * wd)
                    W_v.data.add_(G_v, alpha=-lr / scale)
                else:  
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


    @torch.no_grad()
    def _update_kq_weights(self, p, g, group: dict[str, Any]):
        # interpret rows as [Q; K; V]
        pd_type = group["pd_type"] 
        lsqr_max_iter = group["lsqr_max_iter"] 
        attn_max_iter = group["attn_max_iter"] 
        n_embed = p.shape[1] 
        assert p.shape[0] == 3 * n_embed
        # A1=W_q, A2=W_k, G1=G_k, G2=G_q
        A1, G2   = p[:n_embed, :],                  g[:n_embed, :]
        A2, G1   = p[n_embed:2 * n_embed, :],       g[n_embed:2 * n_embed, :]
        assert A1.shape == G2.shape and A2.shape == G1.shape 

        A_linop = attn_linop_from_matrices(A1, A2)
        
        Grad = torch.cat([G1, G2], dim=0)

        # upper bound on the operator norm of mathcal{A}
        lamb_max = A_linop.fro_norm
        mu_reg = 0 
        beta = group["rho_over_lr"] * group["lr"]

        # Update key-query weights via PDHG or FISTA
        prox_h_conj = lambda y, rho, R: prox_l1(y, rho * beta, R=R)
        h_conj = lambda y: beta * torch.abs(y).sum()
        
        if group["warm_start"] or attn_max_iter == 0:
            Y0, _ = Y_dual_feasible(A1=A1, A2=A2, G1=G1, G2=G2, method="lsqr", maxit=lsqr_max_iter) 
            if pd_type != "fista" or attn_max_iter == 0:
                (Z1_0, Z2_0), res = attn_least_squares_solve(A1=A1, A2=A2, G1=G1, G2=G2, 
                                                    X_type="Z", Y0=Y0, beta=beta, 
                                                    tol=1e-10, maxit=lsqr_max_iter, diag_scaling=True)
        else:
            Y0, Z1_0, Z2_0 = None, None, None

        if attn_max_iter == 0: # use values directly from LSQR 
            Z1_t, Z2_t = Z1_0, Z2_0
            residuals = res
            r1, r1_rel, r2, r2_rel = pd_residuals_max_ball_linop(
                A_linop=A_linop, Y=Y0, Z=Z0, Grad=Grad, beta=beta, mu=mu_reg)
            residuals = {'r1': [r1], 'r2': [r2], 'r1_rel': [r1_rel], 'r2_rel': [r2_rel], 
                            'z_norm': [ (Z1_t.pow(2).sum() + Z2_t.pow(2).sum()).sqrt().item() ], 
                            'y_norm': [ Y0.pow(2).sum().sqrt().item() ] }
            norm_Y = Y0.pow(2).sum().sqrt().item()

        elif pd_type == "pdhg": 
            # Z1_0=-G1; Z2_0=-G2 
            # Run PDHG  
            if Z1_0 is not None: Z0 = torch.cat([Z1_0, Z2_0], dim=0)
            else: Z0 = None 
            Z_t, residuals, norm_Y, _ = pdhg_kq_attn_layer(prox_h_conj,
                                        A_linop=A_linop, Grad=Grad,
                                        max_iter=attn_max_iter, lamb_max=lamb_max, beta=beta,
                                        mu=mu_reg, eps_abs=1e-6, eps_rel=1e-12, stopping=False, 
                                        Z0=Z0, Y0=Y0, diag_scaling=group["diag_scaling"], 
                                        h_conj=h_conj, pd_residuals=pd_residuals_max_ball_linop,
                                        reflected_halpern=group["reflected_halpern"], 
                                        enable_restart=group["enable_restart"])
            Z1_t, Z2_t = Z_t[:n_embed], Z_t[n_embed:]
            
        elif pd_type == "fista":
            # Run FISTA
            # mu < mu_max = \|A(G)\|_\max / \beta
            mu_max = mathcal_A_linop_base(A1=A1, A2=A2, Z1=G1, Z2=G2).abs().max().item() / beta
            mu_reg = max(group["mu_frac"] * mu_max, 1e-6)
            Y_fista, Z_t, residuals = fista_ls_l1_reg(
                A_linop=A_linop, Grad=Grad,
                beta=beta, mu=mu_reg, 
                lamb_max=lamb_max, max_iter=attn_max_iter,
                eps_abs=1e-6, eps_rel=1e-12, stopping=False,
                Y0=Y0, pd_residuals=pd_residuals_max_ball_linop
            )
            Z1_t, Z2_t = Z_t[:n_embed], Z_t[n_embed:]
            norm_Y = Y_fista.pow(2).sum().sqrt().item() 

        residuals["G1_norm"] = G1.norm().item()
        residuals["G2_norm"] = G2.norm().item()
        residuals["Y_norm"] = norm_Y
        
        return Z1_t, Z2_t, residuals



