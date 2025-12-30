

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Tuple, Any, List
import math
import numpy as np
import torch



@dataclass
class StepwiseSchedule:
    """Piecewise-constant step-size scaling."""
    every: int = 0              # if >0, apply every `every` iterations
    gamma_mult: float = 1.0     # multiply primal step(s)
    rho_mult: float = 1.0       # multiply dual step(s)
    start: int = 0              # start applying at iteration >= start


@dataclass
class PrimalWeightUpdate:
    """
    Balances primal/dual residuals by updating a scalar weight w = gamma/rho
    while keeping product gamma*rho approximately constant.
    """
    enabled: bool = False
    every: int = 10
    kappa: float = 10.0        # imbalance threshold
    eta: float = 1.5           # multiplicative change (like Chambolle-Pock)
    w_min: float = 1e-3
    w_max: float = 1e3
    mode: str = "threshold"    # "threshold" or "sqrt_ratio"


@dataclass
class AdaptiveRestart:
    """
    Restarts inertial/extrapolation when a merit (e.g., residual) worsens.
    """
    enabled: bool = False
    check_every: int = 1
    factor: float = 1.05        # restart if merit > factor * best_or_prev
    use_best: bool = True       # compare to best merit so far or previous


@dataclass
class PDHGOptions:
    max_iter: int = 100
    eps_abs: float = 1e-3
    eps_rel: float = 1e-3
    stopping: bool = False
    min_iter: int = 10

    # step sizes
    lamb_max: Optional[float] = None
    step_clip: Tuple[float, float] = (1e-5, 1e4)  # clip for rho/gamma

    # acceleration (strong convexity)
    acceleration: bool = False

    # diagonal scaling hook (optional)
    diag_scaling: bool = False

    # inertial/extrapolation coefficient (θ in your code)
    theta: float = 1.0

    # extras
    halpern_start: int = np.inf
    reflected_pdhg: bool = False

    # new features requested
    stepwise: StepwiseSchedule = StepwiseSchedule()
    pw_update: PrimalWeightUpdate = PrimalWeightUpdate()
    restart: AdaptiveRestart = AdaptiveRestart()

    # residual normalization
    norm_first_iter: bool = False


# -----------------------------
# Helpers
# -----------------------------

def _use_second_primal_block(W_k: Optional[torch.Tensor], G_wq: Optional[torch.Tensor]) -> bool:
    return (W_k is not None) and (G_wq is not None)


@torch.no_grad()
def _estimate_lamb_max_sq(W_q: torch.Tensor,
                          W_k: Optional[torch.Tensor],
                          iters: int = 20) -> float:
    """
    Estimates largest eigenvalue of (W_q^T W_q + W_k^T W_k) via power iteration.
    Returns a Python float.
    """
    device = W_q.device
    dtype = W_q.dtype
    n = W_q.shape[1]
    v = torch.randn(n, 1, device=device, dtype=dtype)
    v = v / (v.norm() + 1e-12)

    for _ in range(iters):
        u = W_q @ v
        w = W_q.t() @ u
        if W_k is not None:
            u2 = W_k @ v
            w = w + (W_k.t() @ u2)
        v = w / (w.norm() + 1e-12)

    # Rayleigh quotient
    u = W_q @ v
    rq = (u.pow(2).sum())
    if W_k is not None:
        u2 = W_k @ v
        rq = rq + (u2.pow(2).sum())
    return float(rq.item())


def _make_steps(Y: torch.Tensor,
                W_q: torch.Tensor,
                W_k: Optional[torch.Tensor],
                use_Z2: bool,
                mu: float,
                lamb_max: Optional[float],
                step_clip: Tuple[float, float]) -> Tuple[float, float, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns (rho, gamma, R, Gamma_1, Gamma_2).
    """
    if lamb_max is None:
        # use operator norm squared estimate
        lamb_max = _estimate_lamb_max_sq(W_q, W_k)

    rho0 = 0.99 / (lamb_max + 1e-12)
    rho0 = float(max(min(rho0, step_clip[1]), step_clip[0]))
    gamma0 = rho0

    R = rho0 * torch.ones_like(Y)
    Gamma_1 = gamma0 * torch.ones((W_q.shape[0], 1), device=W_q.device, dtype=W_q.dtype)
    Gamma_2 = None
    if use_Z2:
        assert W_k is not None
        Gamma_2 = gamma0 * torch.ones((W_k.shape[0], 1), device=W_k.device, dtype=W_k.dtype)

    return rho0, gamma0, R, Gamma_1, Gamma_2


def _default_residuals_AB(
    W_k: Optional[torch.Tensor],
    W_q: torch.Tensor,
    Y_old: torch.Tensor,
    Y_new: torch.Tensor,
    Z1_old: torch.Tensor,
    Z1_new: torch.Tensor,
    Z2_old: Optional[torch.Tensor],
    Z2_new: Optional[torch.Tensor],
    Z1_bar: torch.Tensor,
    Z2_bar: Optional[torch.Tensor],
    R: torch.Tensor,
    Gamma_1: torch.Tensor,
    Gamma_2: Optional[torch.Tensor],
) -> Tuple[float, float, float, float]:
    """
    Returns r1, r1_rel, r2, r2_rel. The "rel" components here are placeholders;
    caller can compute normalization (matches your original behavior).
    """
    if Z2_new is not None:
        assert W_k is not None and Z2_old is not None and Z2_bar is not None and Gamma_2 is not None
        r1 = ((1 / R) * (
            Y_new - Y_old
            - R * ((Z1_bar - Z1_new).t() @ W_q)
            - R * (W_k.t() @ (Z2_bar - Z2_new))
        )).pow(2).sum().sqrt().item()
        r2 = (((Z1_new - Z1_old) / Gamma_1).pow(2).sum() + ((Z2_new - Z2_old) / Gamma_2).pow(2).sum()).sqrt().item()
    else:
        r1 = ((1 / R) * (Y_new - Y_old - R * ((Z1_bar - Z1_new).t() @ W_q))).pow(2).sum().sqrt().item()
        r2 = ((Z1_new - Z1_old) / Gamma_1).pow(2).sum().sqrt().item()

    # rel terms computed outside (need norms)
    return float(r1), 0.0, float(r2), 0.0


def _compute_rel_norms(
    W_k: Optional[torch.Tensor],
    W_q: torch.Tensor,
    Y_new: torch.Tensor,
    Z1_new: torch.Tensor,
    Z2_new: Optional[torch.Tensor],
) -> Tuple[float, float]:
    """
    Matches your original normalization idea: norm1 from K^T z, norm2 from K y.
    """
    if Z2_new is not None:
        assert W_k is not None
        norm1 = (Z1_new.t() @ W_q + W_k.t() @ Z2_new).pow(2).sum().sqrt().item()
        norm2 = ((W_q @ Y_new.t()).pow(2).sum() + (W_k @ Y_new).pow(2).sum()).sqrt().item()
    else:
        norm1 = (Z1_new.t() @ W_q).pow(2).sum().sqrt().item()
        norm2 = (W_q @ Y_new.t()).pow(2).sum().sqrt().item()

    if norm1 < 1e-6:
        norm1 = 1.0
    if norm2 < 1e-6:
        norm2 = 1.0
    return float(norm1), float(norm2)


def _apply_stepwise(schedule: StepwiseSchedule, t: int, rho: float, gamma: float) -> Tuple[float, float]:
    if schedule.every and t >= schedule.start and (t % schedule.every == 0) and t > 0:
        rho *= schedule.rho_mult
        gamma *= schedule.gamma_mult
    return rho, gamma


def _apply_primal_weight_update(
    cfg: PrimalWeightUpdate,
    t: int,
    r1: float,
    r2: float,
    rho: float,
    gamma: float,
) -> Tuple[float, float]:
    if not cfg.enabled or cfg.every <= 0 or (t % cfg.every != 0) or t == 0:
        return rho, gamma

    eps = 1e-12
    if cfg.mode == "sqrt_ratio":
        # continuous update: w <- clip(w * sqrt(r2/r1))
        w = gamma / max(rho, eps)
        w_new = w * math.sqrt(max(r2, eps) / max(r1, eps))
        w_new = float(min(max(w_new, cfg.w_min), cfg.w_max))
        scale = w_new / max(w, eps)
        gamma *= scale
        rho /= scale
        return rho, gamma

    # threshold mode (Chambolle-Pock style)
    if r2 > cfg.kappa * r1:
        # primal residual dominates => increase gamma, decrease rho
        gamma *= cfg.eta
        rho /= cfg.eta
    elif r1 > cfg.kappa * r2:
        # dual residual dominates => decrease gamma, increase rho
        gamma /= cfg.eta
        rho *= cfg.eta

    # clamp weight
    w = gamma / max(rho, eps)
    w = float(min(max(w, cfg.w_min), cfg.w_max))
    # enforce w by rescaling while keeping product constant
    prod = gamma * rho
    gamma = math.sqrt(prod * w)
    rho = math.sqrt(prod / w)
    return rho, gamma


def _maybe_restart(
    cfg: AdaptiveRestart,
    t: int,
    merit: float,
    merit_prev: Optional[float],
    merit_best: Optional[float],
) -> Tuple[bool, float, float]:
    if not cfg.enabled or cfg.check_every <= 0 or (t % cfg.check_every != 0) or t == 0:
        return False, merit, (merit_best if merit_best is not None else merit)

    ref = merit_best if (cfg.use_best and merit_best is not None) else (merit_prev if merit_prev is not None else merit)
    if merit > cfg.factor * ref:
        # trigger restart
        best = min(merit_best, merit) if merit_best is not None else merit
        return True, merit, best

    best = min(merit_best, merit) if merit_best is not None else merit
    return False, merit, best


# -----------------------------------------
# Main modular PDHG (cuPDLPx-like features)
# -----------------------------------------

def pdhg_method_AB_modular(
    prox_h_conj: Callable[..., torch.Tensor],
    W_k: torch.Tensor | None,
    W_q: torch.Tensor,
    G_wk: torch.Tensor,
    G_wq: torch.Tensor | None,
    beta: float,
    mu: float = 0.0,
    opts: PDHGOptions = PDHGOptions(),
    # init
    Y0: torch.Tensor | None = None,
    Z1_0: torch.Tensor | None = None,
    Z2_0: torch.Tensor | None = None,
    # hooks (optional)
    h_conj: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    f_star: Optional[float] = None,
    pd_residuals: Optional[Callable[..., Tuple[float, float, float, float]]] = None,
    diag_scaling_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, List[float]], float]:
    """
    Drop-in replacement for your pdhg_method_AB with:
      - Halpern scheme
      - Reflection (reflected PDHG)
      - Adaptive restart (residual-based)
      - Fixed stepwise schedule
      - Primal-weight update (residual balancing)

    Returns (Z1, Z2, residuals_dict, ||Y||_F).
    """
    use_Z2 = _use_second_primal_block(W_k, G_wq)

    # init primal blocks
    Z1 = Z1_0 if Z1_0 is not None else torch.zeros_like(G_wk)
    Z1_bar = Z1.clone()
    Z2: Optional[torch.Tensor]
    Z2_bar: Optional[torch.Tensor]
    if use_Z2:
        assert W_k is not None and G_wq is not None
        Z2 = Z2_0 if Z2_0 is not None else torch.zeros_like(G_wq)
        Z2_bar = Z2.clone()
        Z_total_size = Z1.numel() + Z2.numel()
    else:
        Z2 = None
        Z2_bar = None
        Z_total_size = Z1.numel()

    # init dual Y
    if Y0 is None:
        Y = torch.zeros((Z1.shape[1], W_q.shape[1]), device=Z1.device, dtype=Z1.dtype)
    else:
        Y = Y0

    # step sizes
    if opts.diag_scaling and use_Z2 and (diag_scaling_fn is not None):
        # user-supplied diagonal scaling (e.g., pdhg_diagonal_scaling(A=W_k, B=W_q, eta=0.99))
        R, Gamma_1, Gamma_2 = diag_scaling_fn(A=W_k, B=W_q, eta=0.99)
        # keep scalar rho/gamma for optional updates; infer approximate
        rho = float(R.mean().item())
        gamma = float(Gamma_1.mean().item())
    else:
        rho, gamma, R, Gamma_1, Gamma_2 = _make_steps(
            Y=Y,
            W_q=W_q,
            W_k=W_k if use_Z2 else None,
            use_Z2=use_Z2,
            mu=mu,
            lamb_max=opts.lamb_max,
            step_clip=opts.step_clip,
        )

    residuals: Dict[str, List[float]] = {"r1": [], "r2": [], "r1_rel": [], "r2_rel": [], "dual_vals": [], "rel_gap": []}
    if mu == 0.0:
        residuals.pop("dual_vals")
        residuals.pop("rel_gap")

    # Halpern anchor
    anchor_Y: Optional[torch.Tensor] = None
    anchor_Z1: Optional[torch.Tensor] = None
    anchor_Z2: Optional[torch.Tensor] = None

    # restart bookkeeping
    merit_prev: Optional[float] = None
    merit_best: Optional[float] = None

    if opts.max_iter <= 0:
        return Z1, Z2, residuals, float(Y.norm().item())

    for t in range(opts.max_iter):
        # -------------------
        # Prox updates (PDHG)
        # -------------------
        if use_Z2:
            assert W_k is not None and Z2 is not None and Z2_bar is not None and Gamma_2 is not None and G_wq is not None
            Y_new = prox_h_conj(Y + R * (Z1_bar.t() @ W_q + W_k.t() @ Z2_bar), 1, R=R)
            Z1_new = (1.0 / (1.0 + Gamma_1 * mu)) * (Z1 - Gamma_1 * (W_q @ Y_new.t() + G_wk))
            Z2_new = (1.0 / (1.0 + Gamma_2 * mu)) * (Z2 - Gamma_2 * (W_k @ Y_new + G_wq))
        else:
            Y_new = prox_h_conj(Y + R * (Z1_bar.t() @ W_q), 1, R=R)
            Z1_new = (1.0 / (1.0 + Gamma_1 * mu)) * (Z1 - Gamma_1 * (W_q @ Y_new.t() + G_wk))
            Z2_new = None

        # -------------------
        # Residuals / metrics
        # -------------------
        if pd_residuals is not None:
            r1, r1_rel, r2, r2_rel = pd_residuals(
                A=W_k, B=W_q, Y=Y_new, Z1=Z1_new, Z2=Z2_new, G1=G_wk, G2=G_wq, beta=beta, mu=mu
            )
        else:
            r1, _, r2, _ = _default_residuals_AB(
                W_k=W_k if use_Z2 else None,
                W_q=W_q,
                Y_old=Y,
                Y_new=Y_new,
                Z1_old=Z1,
                Z1_new=Z1_new,
                Z2_old=Z2,
                Z2_new=Z2_new,
                Z1_bar=Z1_bar,
                Z2_bar=Z2_bar,
                R=R,
                Gamma_1=Gamma_1,
                Gamma_2=Gamma_2,
            )
            # your normalization scheme
            norm1, norm2 = _compute_rel_norms(W_k if use_Z2 else None, W_q, Y_new, Z1_new, Z2_new)
            r1_rel = max(1e-8, r1 - opts.eps_abs * (Y.numel() ** 0.5)) / norm1
            r2_rel = max(1e-8, r2 - opts.eps_abs * (Z_total_size ** 0.5)) / norm2

        residuals["r1"].append(float(r1))
        residuals["r2"].append(float(r2))
        residuals["r1_rel"].append(float(r1_rel))
        residuals["r2_rel"].append(float(r2_rel))

        # merit for restart (simple + robust)
        merit = float(r1 + r2)

        # -------------------
        # Optional: stopping
        # -------------------
        if opts.stopping and (t >= opts.min_iter) and (pd_residuals is None):
            # use same thresholds as your original code
            norm1, norm2 = _compute_rel_norms(W_k if use_Z2 else None, W_q, Y_new, Z1_new, Z2_new)
            if (r1 <= opts.eps_abs * (Y.numel() ** 0.5) + opts.eps_rel * norm1
                and r2 <= opts.eps_abs * (Z_total_size ** 0.5) + opts.eps_rel * norm2):
                Z1, Z2, Y = Z1_new, Z2_new, Y_new
                break

        # -------------------
        # Optional: strong convex acceleration (your logic)
        # -------------------
        theta = opts.theta
        if opts.acceleration and (not opts.diag_scaling):
            theta = 1.0 / math.sqrt(1.0 + 2.0 * gamma * mu)
            gamma = theta * gamma
            rho = rho / theta
            R = rho * torch.ones_like(Y)
            Gamma_1 = gamma * torch.ones((W_q.shape[0], 1), device=W_q.device, dtype=W_q.dtype)
            if use_Z2:
                assert W_k is not None
                Gamma_2 = gamma * torch.ones((W_k.shape[0], 1), device=W_k.device, dtype=W_k.dtype)

        # -------------------
        # Optional: fixed stepwise schedule
        # -------------------
        if (not opts.diag_scaling):  # for diag_scaling, scaling rules are unclear; keep fixed
            rho, gamma = _apply_stepwise(opts.stepwise, t, rho, gamma)
            R = rho * torch.ones_like(Y)
            Gamma_1 = gamma * torch.ones((W_q.shape[0], 1), device=W_q.device, dtype=W_q.dtype)
            if use_Z2:
                assert W_k is not None
                Gamma_2 = gamma * torch.ones((W_k.shape[0], 1), device=W_k.device, dtype=W_k.dtype)

        # -------------------
        # Optional: primal weight update (balances residuals)
        # -------------------
        if (not opts.diag_scaling):
            rho, gamma = _apply_primal_weight_update(opts.pw_update, t, r1, r2, rho, gamma)
            R = rho * torch.ones_like(Y)
            Gamma_1 = gamma * torch.ones((W_q.shape[0], 1), device=W_q.device, dtype=W_q.dtype)
            if use_Z2:
                assert W_k is not None
                Gamma_2 = gamma * torch.ones((W_k.shape[0], 1), device=W_k.device, dtype=W_k.dtype)

        # -------------------
        # Optional: dual diagnostics (your logic)
        # -------------------
        if mu > 0.0 and (h_conj is not None):
            dual_val = -h_conj(Y_new) - (1.0 / (2.0 * mu)) * (W_q @ Y_new.t() + G_wk).pow(2).sum()
            if use_Z2:
                assert W_k is not None and G_wq is not None
                dual_val = dual_val - (1.0 / (2.0 * mu)) * (W_k @ Y_new + G_wq).pow(2).sum()
            residuals["dual_vals"].append(float(dual_val.item()))
            if f_star is not None:
                residuals["rel_gap"].append(float(abs(f_star - dual_val.item()) / (abs(f_star) + 1e-12)))

        # -------------------
        # Optional: reflection
        # -------------------
        if opts.reflected_pdhg:
            Y_new = 2.0 * Y_new - Y
            Z1_new = 2.0 * Z1_new - Z1
            if use_Z2 and (Z2_new is not None) and (Z2 is not None):
                Z2_new = 2.0 * Z2_new - Z2

        # -------------------
        # Optional: Halpern scheme
        # -------------------
        if opts.halpern_start == t:
            anchor_Y = Y_new.clone()
            anchor_Z1 = Z1_new.clone()
            anchor_Z2 = Z2_new.clone() if use_Z2 else None

        if (t > opts.halpern_start) and (anchor_Y is not None) and (anchor_Z1 is not None):
            a = (t + 1.0) / (t + 2.0)
            b = 1.0 / (t + 2.0)
            Y_new = a * Y_new + b * anchor_Y
            Z1_new = a * Z1_new + b * anchor_Z1
            if use_Z2 and (Z2_new is not None) and (anchor_Z2 is not None):
                Z2_new = a * Z2_new + b * anchor_Z2

        # -------------------
        # Adaptive restart (residual-based)
        # -------------------
        do_restart, merit_prev, merit_best = _maybe_restart(opts.restart, t, merit, merit_prev, merit_best)
        if do_restart:
            # reset extrapolation by collapsing bars to the current iterate
            Z1_bar = Z1_new
            if use_Z2 and (Z2_new is not None):
                Z2_bar = Z2_new
            theta = 0.0  # safest: no inertial step this iteration
        else:
            # standard extrapolation
            Z1_bar = Z1_new + theta * (Z1_new - Z1)
            if use_Z2 and (Z2_new is not None) and (Z2 is not None):
                Z2_bar = Z2_new + theta * (Z2_new - Z2)

        # commit iterate
        Z1, Z2, Y = Z1_new, Z2_new, Y_new

    return Z1, Z2, residuals, float(Y.norm().item())
