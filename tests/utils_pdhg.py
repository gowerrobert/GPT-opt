import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import torch
import contextlib, json, time
from glob import glob
import pandas as pd

from gptopt.optim.fast_pdhg import pdhg_kq_attn_layer
from gptopt.optim.nesterov import nesterov_lmax_moreau
from gptopt.optim.attn_kq import prox_l1, fista_ls_l1_reg, AttnPDAdamW, pd_residuals_infty_ball
from gptopt.optim.myadamw import MyAdamW
from gptopt.train import Logging, eval_validation_loss
from gptopt.gpt_model import CausalSelfAttention
from gptopt.utils import get_worker_info, save_checkpoint, load_checkpoint
import torch.distributed as dist 
from gptopt.train import train
from gptopt.optim.utils import get_scheduler, get_optimizer
from gptopt.utils import hash_config, set_seed, get_worker_info
from gptopt.model import load_model
from gptopt.data.data_utils import get_data_dir
from gptopt.dataloader import ShardedDataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import hydra
import os
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from gptopt.optim.attn_kq import *
from gptopt.optim.attn_utils import *
from gptopt.optim.fast_pdhg import *



def cvxpy_ls_l1_reg(W_k, W_q, G_wk, G_wq, beta, mu):
    Y = cp.Variable((G_wk.shape[1], W_q.shape[1]))
    obj = (1 / (2*mu)) * cp.sum_squares(W_q @ Y.T + G_wk) \
        + (1 / (2*mu)) * cp.sum_squares(W_k @ Y + G_wq)\
        + beta * cp.abs(Y).sum()
    objective = cp.Minimize(obj) 
    prob = cp.Problem(objective, [])
    prob.solve(solver=cp.CLARABEL)
    assert prob.status in ["optimal", "optimal_inaccurate"], print(prob.status)
    return Y.value, -obj.value


def cvxpy_lmax_smooth(W_k, W_q, G_wk, G_wq, beta, mu_moreau):
    Z1, Z2 = cp.Variable(W_k.shape), cp.Variable(W_q.shape)
    X = Z1.T @ W_q + W_k.T @ Z2
    obj = cp.trace(G_wk.T @ Z1) + cp.trace(G_wq.T @ Z2) \
           + (1/(2*mu_moreau)) * cp.sum_squares(cp.pos(cp.abs(X) - beta))
    objective = cp.Minimize(obj)  
    prob = cp.Problem(objective, [])
    prob.solve(solver=cp.CLARABEL)
    assert prob.status in ["optimal", "optimal_inaccurate"], print(prob.status)
    return Z1.value, Z2.value, obj.value


def plot_residuals(res, dpi=120):
    has_gap  = "rel_gap"   in res and len(res["rel_gap"])
    has_dual = "dual_vals" in res and len(res["dual_vals"])
    if has_gap and has_dual:
        fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True, dpi=dpi); ax = ax.ravel()
    elif has_gap or has_dual:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharex=True, dpi=dpi)
    else:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True, dpi=dpi)
    ax = np.atleast_1d(ax)

    r1, r2 = res.get("r1", []), res.get("r2", [])
    if len(r1) and len(r2):
        ax[0].plot(r1, label=r"$r_1$"); ax[0].plot(r2, label=r"$r_2$")
        ax[0].set(yscale="log", title="Absolute residuals", xlabel="iteration")
        ax[0].grid(True, which="both", ls="--", alpha=0.4); ax[0].legend()
    else:
        ax[0].axis("off")

    r1r, r2r = res.get("r1_rel", []), res.get("r2_rel", [])
    if len(r1r) and len(r2r):
        ax[1].plot(r1r, label=r"$r_1^{\text{rel}}$"); ax[1].plot(r2r, label=r"$r_2^{\text{rel}}$")
        ax[1].set(yscale="log", title="Relative residuals", xlabel="iteration")
        ax[1].grid(True, which="both", ls="--", alpha=0.4); ax[1].legend()
    else:
        ax[1].axis("off")

    if has_gap:
        i = 2 if (has_gap and not has_dual) else (2 if not has_dual else 2)
        ax[i].plot(res["rel_gap"]); ax[i].set(yscale="log", title="Relative gap", xlabel="iteration")
        ax[i].grid(True, which="both", ls="--", alpha=0.4)
    if has_dual:
        j = 2 if (has_dual and not has_gap) else (3 if has_gap else 2)
        ax[j].plot(res["dual_vals"]); ax[j].set(title="Dual objective", xlabel="iteration")
        ax[j].grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()


def cvxpy_A(g, A, beta, mu=0):
    z = cp.Variable(A.shape[1])
    obj = g.T @ z
    if mu > 0:
        obj += (mu / 2) * cp.sum_squares(z)
    objective = cp.Minimize(obj)
    constraints = [cp.norm(A @ z, "inf") <= beta]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == "optimal", print("CVXPY problem status:", prob.status)
    return z.value, constraints[0].dual_value, obj.value


def cvxpy_AB(G1, G2, A, B, beta, mu=0, verbose=False):
    Z1, Z2, X = cp.Variable(A.shape), cp.Variable(B.shape), cp.Variable((A.shape[1], B.shape[1]))
    obj = cp.trace(G1.T @ Z1) + cp.trace(G2.T @ Z2)
    if mu > 0:
        obj += (mu / 2) * (cp.sum_squares(Z1) + cp.sum_squares(Z2))
    objective = cp.Minimize(obj)
    constraints = [Z1.T @ B + A.T @ Z2 == X, 
                   cp.max(cp.abs(X)) <= beta]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=verbose)
    assert prob.status in ["optimal", "optimal_inaccurate"], print(prob.status)
    return Z1.value, Z2.value, obj.value, constraints[0].dual_value



def compare_methods_fast_pdhg(prox_h_conj, h_conj, A, B, G1, G2, beta, mu_reg, 
                    Z1_0=None, Z2_0=None, Y0=None,
                    f_star=None, max_iter=1000, stopping=True, pd_residuals=pd_residuals_max_ball,
                    eps_abs=1e-8, eps_rel=1e-8, versbose=False, theta=1):

    func_obj = lambda Z1, Z2: (torch.trace(G1.T @ Z1) + torch.trace(G2.T @ Z2) \
                            + (mu_reg / 2) * ((Z1).pow(2).sum() + Z2.pow(2).sum())).item()  
    func_constr_viol = lambda Z1, Z2: max(torch.max(torch.abs(Z1.T @ B + A.T @ Z2)).item() - beta, 0) / beta

    metrics = {} 

    settings = {"pdhg": {"diag_scaling": False, "equilibration": False, "reflected_halpern":False, "enable_restart": False},
             "rehpdhg": {"diag_scaling": False, "equilibration": False, "reflected_halpern":True, "enable_restart": False},
             "pdhg ds": {"diag_scaling": True, "equilibration": False, "reflected_halpern":False, "enable_restart": False},
           # "pdhg eq": {"diag_scaling": False, "equilibration": True, "reflected_halpern":False, "enable_restart": False},
          "rehpdhg ds": {"diag_scaling": True, "equilibration": False, "reflected_halpern":True, "enable_restart": False}
           }
    if mu_reg == 0:
        settings["ada rehpdhg"] = {"diag_scaling": False, "equilibration": False, "reflected_halpern":True, "enable_restart": True}
        settings["ada rehpdhg ds"] = {"diag_scaling": True, "equilibration": False, "reflected_halpern":True, "enable_restart": True}
    residuals = {}
    m = A.shape[0]
    if Z1_0 is None: 
        Z0 = None
    else:
        Z0 = torch.cat([Z1_0, Z2_0], dim=0) 
    if Z0 is not None and Y0 is not None:
        metrics["init"] = {
                    "obj": func_obj(Z0[:m, :], Z0[m:, :]),
                    "viol": func_constr_viol(Z0[:m, :], Z0[m:, :])}
    # Torch prox for h* (uses prox_l1 from pdhg.py)
    prox_h_conj = lambda y, rho, R: prox_l1(y, rho * beta, R=R)
    h_conj = lambda y: beta * torch.abs(y).sum()

    for setting in settings:
        # Run torch PDHG
        Z_t, res, _, _ = pdhg_kq_attn_layer(
            prox_h_conj, A2=A, A1=B, G1=G1, G2=G2,
            max_iter=max_iter, eps_abs=eps_abs, eps_rel=eps_rel,
            stopping=stopping, Y0=Y0, Z0=Z0,
            h_conj=h_conj, beta=beta, pd_residuals=pd_residuals,
            f_star=f_star, mu=mu_reg,
            diag_scaling=settings[setting]["diag_scaling"], 
            equilibration=settings[setting]["equilibration"],
            reflected_halpern=settings[setting]["reflected_halpern"],
            enable_restart=settings[setting]["enable_restart"],
            verbose=versbose, theta=theta
        )
        residuals[setting] = res  
        metrics[setting] = {
                    "obj": func_obj(Z_t[:m, :], Z_t[m:, :]),
                    "viol": func_constr_viol(Z_t[:m, :], Z_t[m:, :]),
    }  

    if mu_reg > 0:
        _, Z_t, res = fista_ls_l1_reg(
                A2=A, A1=B, G1=G1, G2=G2,
                beta=beta, mu=mu_reg, max_iter=max_iter,
                eps_abs=1e-6, eps_rel=1e-12, stopping=False,
                Y0=Y0, pd_residuals=pd_residuals_max_ball
            )
        residuals["fista"] = res  
        metrics["fista"] = { "obj": func_obj(Z_t[:m, :], Z_t[m:, :]), "viol": func_constr_viol(Z_t[:m, :], Z_t[m:, :])}
        
 
    header = f"{'Method':<12}  {'Obj':>12}  {'Viol':>12}"
    print(header)
    print("-" * len(header))
    for method in metrics.keys(): 
        m = metrics[method]
        print(f"{method:<12}  {m['obj']:>12.6e}  {m['viol']:>12.6e}")

    return residuals


def compare_methods_fista_nesterov(A, B, G1, G2, beta, mu_reg, mu_moreau,
                    Z1_0=None, Z2_0=None, Y0=None,
                    max_iter=1000, stopping=True, pd_residuals=pd_residuals_max_ball,
                    eps_abs=1e-8, eps_rel=1e-8):

    func_obj = lambda Z1, Z2: (torch.trace(G1.T @ Z1) + torch.trace(G2.T @ Z2)).item()  
    func_constr_viol = lambda Z1, Z2: max(torch.max(torch.abs(Z1.T @ B + A.T @ Z2)).item() - beta, 0) / beta

    metrics = {} 

    residuals = {}
    m = A.shape[0]
    if Z1_0 is None: 
        Z0 = None
    else:
        Z0 = torch.cat([Z1_0, Z2_0], dim=0) 
    if Z0 is not None and Y0 is not None:
        metrics["init"] = {
                    "obj": func_obj(Z0[:m, :], Z0[m:, :]),
                    "viol": func_constr_viol(Z0[:m, :], Z0[m:, :])}

    _, Z_t, res = fista_ls_l1_reg(
            A2=A, A1=B, G1=G1, G2=G2,
            beta=beta, mu=mu_reg, max_iter=max_iter,
            eps_abs=eps_abs, eps_rel=eps_rel, stopping=stopping,
            Y0=Y0, pd_residuals=pd_residuals
        )
    residuals["fista"] = res  
    metrics["fista"] = { "obj": func_obj(Z_t[:m, :], Z_t[m:, :]), "viol": func_constr_viol(Z_t[:m, :], Z_t[m:, :])}
        
    G12 = torch.cat([G1, G2], dim=0)
    AG_max = mathcal_A_linop(A1=B, A2=B, Z=G12).abs().max().item()
    Z0 = - (beta * 1.2 / AG_max) * G12
    Z_t, res = nesterov_lmax_moreau(
                A2=A, A1=B, G1=G1, G2=G2,
                beta=beta, mu=mu_moreau, 
                max_iter=max_iter, Z0=Z0,
                eps_abs=eps_abs, eps_rel=eps_rel, stopping=stopping,
                pd_residuals=pd_residuals
            )
    residuals["nesterov G init"] = res  
    metrics["nesterov G init"] = { "obj": func_obj(Z_t[:m, :], Z_t[m:, :]), "viol": func_constr_viol(Z_t[:m, :], Z_t[m:, :])}
      
    Z0 = torch.randn(2 * G1.shape[0], G1.shape[1], device=G1.device, dtype=G1.dtype)
    AZ_max = mathcal_A_linop(A1=B, A2=B, Z=Z0).abs().max().item()
    Z0 = (beta * 1.2 / AZ_max) * Z0
    Z_t, res = nesterov_lmax_moreau(
                A2=A, A1=B, G1=G1, G2=G2,
                beta=beta, mu=mu_moreau, 
                max_iter=max_iter, Z0=Z0,
                eps_abs=eps_abs, eps_rel=eps_rel, stopping=stopping,
                pd_residuals=pd_residuals
            )
    residuals["nesterov rand init"] = res  
    metrics["nesterov rand init"] = { "obj": func_obj(Z_t[:m, :], Z_t[m:, :]), "viol": func_constr_viol(Z_t[:m, :], Z_t[m:, :])}
      
    Z0 = None
    Z_t, res = nesterov_lmax_moreau(
                A2=A, A1=B, G1=G1, G2=G2,
                beta=beta, mu=mu_moreau, 
                max_iter=max_iter, Z0=Z0,
                eps_abs=eps_abs, eps_rel=eps_rel, stopping=stopping,
                pd_residuals=pd_residuals
            )
    residuals["nesterov zero init"] = res  
    metrics["nesterov zero init"] = { "obj": func_obj(Z_t[:m, :], Z_t[m:, :]), "viol": func_constr_viol(Z_t[:m, :], Z_t[m:, :])}
      

    header = f"{'Method':<12}  {'Obj':>12}  {'Viol':>12}"
    print(header)
    print("-" * len(header))
    for method in metrics.keys(): 
        m = metrics[method]
        print(f"{method:<12}  {m['obj']:>12.6e}  {m['viol']:>12.6e}")

    return residuals


def compare_methods_fista_nesterov_mu(A, B, G1, G2, beta, mu_range_fista,
                                      mu_range_nesterov,
                    Z1_0=None, Z2_0=None, Y0=None,
                    max_iter=20, stopping=False, pd_residuals=pd_residuals_max_ball,
                    eps_abs=1e-8, eps_rel=1e-8):

    func_obj = lambda Z1, Z2: (torch.trace(G1.T @ Z1) + torch.trace(G2.T @ Z2)).item()
    func_constr_viol = lambda Z1, Z2: max(torch.max(torch.abs(Z1.T @ B + A.T @ Z2)).item() - beta, 0) / beta
    mu_max = (G1.t() @ B + A.t() @ G2).abs().max().item() / beta

    metrics = {}
    m = A.shape[0]

    # rows for residual dataframe
    rows = []

    if Z1_0 is not None and Z2_0 is not None:
        Z0 = torch.cat([Z1_0, Z2_0], dim=0)
    else:
        Z0 = None

    if Z0 is not None and Y0 is not None:
        metrics["init"] = {
            "obj": func_obj(Z0[:m, :], Z0[m:, :]),
            "viol": func_constr_viol(Z0[:m, :], Z0[m:, :])
        }
 
    for mu_scale in mu_range_fista:
        mu = mu_scale * mu_max
        metrics[mu_scale] = {}

        # --- FISTA ---
        _, Z_t, res = fista_ls_l1_reg(
            A2=A, A1=B, G1=G1, G2=G2,
            beta=beta, mu=mu, max_iter=max_iter,
            eps_abs=eps_abs, eps_rel=eps_rel, stopping=stopping,
            Y0=Y0, pd_residuals=pd_residuals
        ) 
        rows.append({"model": "fista", "mu": mu_scale, "r_true_res": res["r_true_res"][-1], "r_res": res["r_rel"][-1]})
        metrics[mu_scale]["fista"] = {"obj": func_obj(Z_t[:m, :], Z_t[m:, :]), "viol": func_constr_viol(Z_t[:m, :], Z_t[m:, :])}

    for mu_scale in mu_range_nesterov:
        mu = mu_scale * mu_max
        if mu_scale not in metrics:
            metrics[mu_scale] = {}
        
        # --- Nesterov (G init) ---
        G12 = torch.cat([G1, G2], dim=0)
        AG_max = mathcal_A_linop(A1=B, A2=B, Z=G12).abs().max().item()
        Z0n = - (beta * 1.2 / AG_max) * G12
        Z_t, res = nesterov_lmax_moreau(
            A2=A, A1=B, G1=G1, G2=G2,
            beta=beta, mu=mu,
            max_iter=max_iter, Z0=Z0n,
            eps_abs=eps_abs, eps_rel=eps_rel, stopping=stopping,
            pd_residuals=pd_residuals
        ) 
        rows.append({"model": "nesterov G init", "mu": mu_scale, "r_true_res": res["r_true_res"][-1], "r_res": res["r_rel"][-1]})
        metrics[mu_scale]["nesterov G init"] = {"obj": func_obj(Z_t[:m, :], Z_t[m:, :]), "viol": func_constr_viol(Z_t[:m, :], Z_t[m:, :])}

        # --- Nesterov (rand init) ---
        Z0n = torch.randn(2 * G1.shape[0], G1.shape[1], device=G1.device, dtype=G1.dtype)
        AZ_max = mathcal_A_linop(A1=B, A2=B, Z=Z0n).abs().max().item()
        Z0n = (beta * 1.2 / AZ_max) * Z0n
        Z_t, res = nesterov_lmax_moreau(
            A2=A, A1=B, G1=G1, G2=G2, beta=beta, mu=mu, max_iter=max_iter, Z0=Z0n, eps_abs=eps_abs, 
            eps_rel=eps_rel, stopping=stopping, pd_residuals=pd_residuals
        )
        rows.append({"model": "nesterov rand init", "mu": mu_scale, "r_true_res": res["r_true_res"][-1], "r_res": res["r_rel"][-1]})
        metrics[mu_scale]["nesterov rand init"] = {"obj": func_obj(Z_t[:m, :], Z_t[m:, :]), "viol": func_constr_viol(Z_t[:m, :], Z_t[m:, :])}

        # --- Nesterov (zero init) ---
        Z_t, res = nesterov_lmax_moreau(
            A2=A, A1=B, G1=G1, G2=G2, beta=beta, mu=mu, max_iter=max_iter, Z0=None,
            eps_abs=eps_abs, eps_rel=eps_rel, stopping=stopping, pd_residuals=pd_residuals
        )
        rows.append({"model": "nesterov zero init", "mu": mu_scale, "r_true_res": res["r_true_res"][-1], "r_res": res["r_rel"][-1]})
        metrics[mu_scale]["nesterov zero init"] = {"obj": func_obj(Z_t[:m, :], Z_t[m:, :]), "viol": func_constr_viol(Z_t[:m, :], Z_t[m:, :])}

    df_residuals = (pd.DataFrame(rows)
                      .sort_values(["model", "mu"])
                      .reset_index(drop=True))
    header = f"{'Method':<12}  {'Obj':>12}  {'Viol':>12} {'Mu':>12}"
    print(header)
    print("-" * len(header))
    for mu in metrics.keys():
        for method in metrics[mu].keys(): 
            m = metrics[mu][method]
            print(f"{method:<20}  {m['obj']:>12.6e}  {m['viol']:>12.6e}  {mu:>.4e}")
    return df_residuals


def compare_methods_fista_pdhg_mu(A, B, G1, G2, beta, mu_range_fista,
                    mu_range_pdhg,
                    Z1_0=None, Z2_0=None, Y0=None,
                    max_iter=20, stopping=False, pd_residuals=pd_residuals_max_ball,
                    eps_abs=1e-8, eps_rel=1e-8):

    func_obj = lambda Z1, Z2: (torch.trace(G1.T @ Z1) + torch.trace(G2.T @ Z2)).item()
    func_constr_viol = lambda Z1, Z2: max(torch.max(torch.abs(Z1.T @ B + A.T @ Z2)).item() - beta, 0) / beta
    mu_max = (G1.t() @ B + A.t() @ G2).abs().max().item() / beta

    metrics = {}
    m = A.shape[0]

    # rows for residual dataframe
    rows = []

    if Z1_0 is not None and Z2_0 is not None:
        Z0 = torch.cat([Z1_0, Z2_0], dim=0)
    else:
        Z0 = None

    if Z0 is not None and Y0 is not None:
        metrics["init"] = {
            "obj": func_obj(Z0[:m, :], Z0[m:, :]),
            "viol": func_constr_viol(Z0[:m, :], Z0[m:, :])
        }

    settings = {"pdhg": {"diag_scaling": False, "equilibration": False, "reflected_halpern":False, "enable_restart": False},
             "rehpdhg": {"diag_scaling": False, "equilibration": False, "reflected_halpern":True, "enable_restart": False}, 
           "ada rehpdhg":{"diag_scaling": False, "equilibration": False, "reflected_halpern":True, "enable_restart": True}}
    
 
    for mu_scale in mu_range_fista:
        mu = mu_scale * mu_max
        metrics[mu_scale] = {}

        # --- FISTA ---
        _, Z_t, res = fista_ls_l1_reg(
            A2=A, A1=B, G1=G1, G2=G2,
            beta=beta, mu=mu, max_iter=max_iter,
            eps_abs=eps_abs, eps_rel=eps_rel, stopping=stopping,
            Y0=Y0, pd_residuals=pd_residuals
        ) 
        rows.append({"model": "fista", "mu": mu_scale, "r_true_res": res["r_true_res"][-1], "r_res": res["r_rel"][-1]})
        metrics[mu_scale]["fista"] = {"obj": func_obj(Z_t[:m, :], Z_t[m:, :]), "viol": func_constr_viol(Z_t[:m, :], Z_t[m:, :])}

    prox_h_conj = lambda y, rho, R: prox_l1(y, rho * beta, R=R)
    h_conj = lambda y: beta * torch.abs(y).sum()

    for mu_scale in mu_range_pdhg:
        mu = mu_scale * mu_max
        if mu_scale not in metrics:
            metrics[mu_scale] = {}
        
        for setting in settings:
            if mu > 0 and "ada" in setting: 
                # only for adaptive restart for mu=0
                continue
            # Run torch PDHG
            Z_t, res, _, _ = pdhg_kq_attn_layer(
                prox_h_conj, A2=A, A1=B, G1=G1, G2=G2,
                max_iter=max_iter, eps_abs=eps_abs, eps_rel=eps_rel,
                stopping=stopping, Y0=Y0, Z0=Z0,
                h_conj=h_conj, beta=beta, pd_residuals=pd_residuals, mu=mu,
                diag_scaling=settings[setting]["diag_scaling"], 
                equilibration=settings[setting]["equilibration"],
                reflected_halpern=settings[setting]["reflected_halpern"],
                enable_restart=settings[setting]["enable_restart"],
                verbose=False
            )
            rows.append({"model": setting, "mu": mu_scale, "r_true_res": res["r_true_res"][-1], "r_res": res["r_rel"][-1]}) 
            metrics[mu_scale][setting] = {
                        "obj": func_obj(Z_t[:m, :], Z_t[m:, :]),
                        "viol": func_constr_viol(Z_t[:m, :], Z_t[m:, :]),
    } 

    df_residuals = (pd.DataFrame(rows)
                      .sort_values(["model", "mu"])
                      .reset_index(drop=True))
    header = f"{'Method':<12}  {'Obj':>12}  {'Viol':>12} {'Mu':>12}"
    print(header)
    print("-" * len(header))
    for mu in metrics.keys():
        for method in metrics[mu].keys(): 
            m = metrics[mu][method]
            print(f"{method:<20}  {m['obj']:>12.6e}  {m['viol']:>12.6e}  {mu:>.4e}")
    return df_residuals




def plot_residuals_grid_by_mu(res_all, dual_scale=False, dpi=120,
                              abs_ylim=None, rel_ylim=None, gap_ylim=None, dual_ylim=None):
    return plot_residuals_grid_by_param(
        res_all, param_name="mu", dual_scale=dual_scale, dpi=dpi,
        abs_ylim=abs_ylim, rel_ylim=rel_ylim, gap_ylim=gap_ylim, dual_ylim=dual_ylim
    )


def plot_residuals_grid_by_param_appprox_vs_true(res_all, dpi=120,
                                 abs_ylim=None, rel_ylim=None):
    plabel = "μ1, μ2"
    pvals = sorted(res_all) if len(res_all) else []
    methods = sorted({m for D in res_all.values() for m in D})
    colors = {m: plt.cm.Set2(i / max(1, len(methods)-1)) for i, m in enumerate(methods)}
    sample = next(iter(res_all.values()), {})
    vals = sample.values() if isinstance(sample, dict) else [] 
    ncols = 2

    fig, axs = plt.subplots(max(1, len(pvals)), ncols, figsize=(5*ncols, 3.2*max(1, len(pvals))), dpi=dpi)
    axs = np.atleast_2d(axs)

    approx_rel_keys = [("r_rel", "-")] 
    true_rel_keys   = [("r_true_res", "-")] 

    specs = [
        (0, true_rel_keys,   abs_ylim, "Original", "log"),
        (1, approx_rel_keys, rel_ylim, "Relaxed", "log"),
    ]

    for r, pv in enumerate(pvals if pvals else [None]):
        D = res_all[pv] if pvals else {}
        for c, keys, ylim, title, yscale in specs:
            any_ = False
            a = axs[r, c]
            for m in methods:
                res = D.get(m, {})
                for k, ls in keys:
                    v = res.get(k, [])
                    if len(v):
                        a.plot(v, color=colors[m], ls=ls); any_ = True
            if any_:
                if yscale == "symlog":
                    a.set_yscale("symlog", linthresh=1e-6)
                elif yscale:
                    a.set_yscale(yscale)
                if ylim is not None:
                    a.set_ylim(ylim)
                label = f"{plabel}={pv} | {title}" if pv is not None else title
                a.set(title=label, xlabel="iter", ylabel="residuals")
                a.grid(True, which="both", ls="--", alpha=0.4)
            else:
                a.axis("off")

    handles = [plt.Line2D([0],[0], color=colors[m], ls='-') for m in methods]
    fig.legend(handles, methods, loc="center left", bbox_to_anchor=(0.985, 0.5), frameon=False, title="Methods")
    plt.tight_layout(rect=[0,0,0.985,1])
    return fig, axs



def plot_residuals_grid_by_param(res_all, param_name="mu", dual_scale=False, dpi=120,
                                 abs_ylim=None, rel_ylim=None, gap_ylim=None, dual_ylim=None, add_res=True):
    plabel = "μ" if param_name in {"mu", "μ"} else ("β" if param_name in {"beta", "β"} else str(param_name))
    pvals = sorted(res_all) if len(res_all) else []
    methods = sorted({m for D in res_all.values() for m in D})
    colors = {m: plt.cm.Set2(i / max(1, len(methods)-1)) for i, m in enumerate(methods)}
    sample = next(iter(res_all.values()), {})
    vals = sample.values() if isinstance(sample, dict) else []
    has_gap  = any("rel_gap"   in r and len(r.get("rel_gap", []))   for r in vals)
    has_dual = any("dual_vals" in r and len(r.get("dual_vals", [])) for r in vals)
    ncols = 4 if (has_gap and has_dual) else (3 if (has_gap or has_dual) else 2)

    fig, axs = plt.subplots(max(1, len(pvals)), ncols, figsize=(5*ncols, 3.2*max(1, len(pvals))), dpi=dpi)
    axs = np.atleast_2d(axs)

    if add_res:
        abs_keys = [("r_sum", "-")]
        rel_keys = [("r_rel_sum", "-")]
    else:
        abs_keys = [("r1", "-"), ("r2", "--")]
        rel_keys = [("r1_rel", "-"), ("r2_rel", "--")]

    specs = [
        (0, abs_keys, abs_ylim, "Abs", "log"),
        (1, rel_keys, rel_ylim, "Rel", "log"),
    ]
    if has_gap:
        specs.append((2, [("rel_gap",":")], gap_ylim, "Gap", "log"))
    if has_dual:
        specs.append(((2 if not has_gap else 3), [("dual_vals","-.")], dual_ylim, "Dual", "symlog" if dual_scale else None))

    for r, pv in enumerate(pvals if pvals else [None]):
        D = res_all[pv] if pvals else {}
        for c, keys, ylim, title, yscale in specs:
            any_ = False
            a = axs[r, c]
            for m in methods:
                res = D.get(m, {})
                for k, ls in keys:
                    if k == "r_sum":
                        a1, a2 = res.get("r1", []), res.get("r2", [])
                        L = min(len(a1), len(a2))
                        v = [max(a1[i], a2[i]) for i in range(L)] if L else []
                    elif k == "r_rel_sum":
                        a1, a2 = res.get("r1_rel", []), res.get("r2_rel", [])
                        L = min(len(a1), len(a2))
                        v = [max(a1[i], a2[i]) for i in range(L)] if L else []
                    else:
                        v = res.get(k, [])
                    if len(v):
                        a.plot(v, color=colors[m], ls=ls); any_ = True
            if any_:
                if yscale == "symlog":
                    a.set_yscale("symlog", linthresh=1e-6)
                elif yscale:
                    a.set_yscale(yscale)
                if ylim is not None:
                    a.set_ylim(ylim)
                label = f"{plabel}={pv:g} | {title}" if pv is not None else title
                a.set(title=label, xlabel="iter")
                a.grid(True, which="both", ls="--", alpha=0.4)
            else:
                a.axis("off")

    handles = [plt.Line2D([0],[0], color=colors[m], ls='-') for m in methods]
    fig.legend(handles, methods, loc="center left", bbox_to_anchor=(0.985, 0.5), frameon=False, title="Methods")
    plt.tight_layout(rect=[0,0,0.985,1])
    return fig, axs



def plot_residuals_cold_warm_grid_by_param(
    residuals_cold_start,
    residuals_warm_start,
    param_name="beta",
    dpi=120,
    rel_ylim=None,
):
    """Cold vs warm start, side-by-side, plotting Rel residual sum (r1_rel + r2_rel).

    Input: residuals_*[param][method][key] -> list
    Y-scale: log. If rel_ylim is None, y-limits are shared per row (cold+warm combined).
    """

    def res_max(res):
        a = res.get("r1_rel", [])
        b = res.get("r2_rel", [])
        L = min(len(a), len(b))
        return [max(a[i], b[i]) for i in range(L)] if L else []

    plabel = "μ" if param_name in {"mu", "μ"} else ("β" if param_name in {"beta", "β"} else str(param_name))
    pvals = sorted(set(residuals_cold_start) | set(residuals_warm_start)) or [None]
    methods = sorted({m for D in (*residuals_cold_start.values(), *residuals_warm_start.values()) for m in D})
    colors = {m: plt.cm.Set2(i / max(1, len(methods) - 1)) for i, m in enumerate(methods)}

    fig, axs = plt.subplots(len(pvals), 2, figsize=(10, 3.2 * len(pvals)), dpi=dpi)
    axs = np.atleast_2d(axs)

    for r, pv in enumerate(pvals):
        Dc = residuals_cold_start.get(pv, {}) if pv is not None else {}
        Dw = residuals_warm_start.get(pv, {}) if pv is not None else {}

        # Compute shared per-row y-limits from both panels.
        if rel_ylim is None:
            ys = [y for D in (Dc, Dw) for m in methods for y in res_max(D.get(m, {})) if y > 0]
            row_ylim = (min(ys)*0.9, max(ys)*1.1) if ys else None
        else:
            row_ylim = rel_ylim

        for c, (label, D) in enumerate((("Cold", Dc), ("Warm", Dw))):
            ax = axs[r, c]
            any_ = False
            for m in methods:
                v = res_max(D.get(m, {}))
                if len(v):
                    ax.plot(v, color=colors[m]); any_ = True
            if not any_:
                ax.axis("off")
                continue
            ax.set_yscale("log")
            if row_ylim is not None:
                ax.set_ylim(row_ylim)
            title = f"{plabel}={pv:g} | Rel ({label})" if pv is not None else f"Rel ({label})"
            ax.set(title=title, xlabel="iter")
            ax.grid(True, which="both", ls="--", alpha=0.4)

    handles = [plt.Line2D([0], [0], color=colors[m], ls='-') for m in methods]
    fig.legend(handles, methods, loc="center left", bbox_to_anchor=(0.985, 0.5), frameon=False, title="Methods")
    plt.tight_layout(rect=[0, 0, 0.985, 1])
    return fig, axs



def plot_residuals_compare(all_res, dpi=120, dual_scale=False,
                           abs_ylim=None, rel_ylim=None, gap_ylim=None, dual_ylim=None):
    methods = sorted(all_res.keys())
    cmap = plt.cm.Set2
    colors = {m: cmap(i / max(1, len(methods)-1)) for i, m in enumerate(methods)}
    has_gap  = any("rel_gap"   in r and len(r["rel_gap"])   for r in all_res.values())
    has_dual = any("dual_vals" in r and len(r["dual_vals"]) for r in all_res.values())
    if has_gap and has_dual:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, dpi=dpi)
        ax = axes.ravel()
    elif has_gap or has_dual:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, dpi=dpi)
        ax = np.atleast_1d(axes)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, dpi=dpi)
        ax = np.atleast_1d(axes)
    def plot_keys(idx, keys, ylim=None, yscale="log"):
        any_ = False
        for m in methods:
            res = all_res.get(m, {})
            for k, ls in keys:
                v = res.get(k, [])
                if len(v):
                    ax[idx].plot(v, color=colors[m], ls=ls); any_ = True
        if any_:
            ax[idx].set(yscale=yscale, xlabel="iteration")
            if ylim is not None: ax[idx].set_ylim(ylim)
            ax[idx].grid(True, which="both", ls="--", alpha=0.4)
        else:
            ax[idx].axis("off")
    plot_keys(0, [("r1","-"),("r2","--")], abs_ylim)
    plot_keys(1, [("r1_rel","-"),("r2_rel","--")], rel_ylim)
    if has_gap:
        gi = 2
        plot_keys(gi, [("rel_gap",":")], gap_ylim)
    if has_dual:
        di = 2 if (has_dual and not has_gap) else 3
        plot_keys(di, [("dual_vals","-.")], dual_ylim, "symlog" if dual_scale else ax[di].get_yscale())
    handles = [plt.Line2D([0],[0], color=colors[m], ls='-') for m in methods]
    ax_fig = ax[0].figure
    ax_fig.legend(handles, methods, loc="center left", bbox_to_anchor=(0.985, 0.5), frameon=False, title="Methods")
    ax_fig.tight_layout(rect=[0,0,0.985,1])



def generate_matrix_rank_normalized_op(m, n, rank, eps=1e-12):
    # generate random matrix mxn with operator norm 1 and given rank 
    A = np.random.randn(m, rank) @ np.random.randn(rank, n) 
    r_A = (np.abs(A).sum(axis=1)**0.5)[None, :]               
    c_A = (np.abs(A).sum(axis=0)**0.5)[None, :]
    inv_rA = np.where(r_A > 0, 1.0 / (r_A + eps), np.zeros_like(r_A))
    inv_cA = np.where(c_A > 0, 1.0 / (c_A + eps), np.zeros_like(c_A))
    return (inv_rA * A) * inv_cA



def gaussian_data(m, n, std1=1, std2=1, G_in_range=False, rank_ratio=1, debug=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A_np = np.random.randn(m, n) * std1
    B_np = np.random.randn(m, n) * std1
    rank = int(min(m, n) * rank_ratio)
    if G_in_range: 
        Y0_np = generate_matrix_rank_normalized_op(n, n, rank) * std2
        G1_np = B_np @ Y0_np.T
        G2_np = A_np @ Y0_np
    else:
        rank = int(min(m, n) * rank_ratio)
        G1_np = generate_matrix_rank_normalized_op(m, n, rank) * std2
        G2_np = generate_matrix_rank_normalized_op(m, n, rank) * std2
        if debug:
            assert np.linalg.norm(G1_np, ord=2) <= std2 + 1e-5 and \
                    np.linalg.norm(G2_np, ord=2) <= std2 + 1e-5

    # Torch tensors
    A = torch.from_numpy(A_np).to(torch.float32).to(device)
    B = torch.from_numpy(B_np).to(torch.float32).to(device)
    G1 = torch.from_numpy(G1_np).to(torch.float32).to(device)
    G2 = torch.from_numpy(G2_np).to(torch.float32).to(device)

    nA = torch.linalg.norm(A, ord="fro").item()
    nB = torch.linalg.norm(B, ord="fro").item()
    lamb_max = (nA * nA + nB * nB) ** 0.5

    return A, B, G1, G2, A_np, B_np, G1_np, G2_np, lamb_max


def make_output_path_hydra(config, output_dir):
    training_params = config["training"]["training_params"]
    opt_config      = config["optimizer"]["optimizer_params"]
    model_config    = config["model"]["config"]

    config_hash = hash_config(
        OmegaConf.to_container(opt_config),
        OmegaConf.to_container(training_params),
        OmegaConf.to_container(model_config),
    )
    file_name = f"{opt_config['name']}-lr-{opt_config['lr']}-{opt_config['lr_schedule']}"
    if "muon_lr" in opt_config:
        file_name += f"-muonlr-{opt_config['muon_lr']}"
    file_name += f"-{config_hash}"
    return os.path.join(output_dir, file_name + ".json")


def plot_residuals_layers(residuals_by_layer, yscale=True, dual_scale=False, 
                          agg_sum=True, plot_res=['abs', 'rel'], dpi=120, include_norms=True):
    layers = sorted(residuals_by_layer.keys())
    want = {str(x).lower() for x in (plot_res or [])}

    ordered = [k for k in ('abs', 'rel') if k in want]
    has_dual = ('dual' in want) and any('dual_vals' in v and len(v.get('dual_vals', [])) for v in residuals_by_layer.values())
    if has_dual:
        ordered.append('dual')

    nrows = max(1, len(layers))
    ncols = max(1, len(ordered))
    if include_norms:
        ncols += 2
        ordered.extend(['z_norm', 'y_norm'])
    fig, ax = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows), sharex=True, dpi=dpi)
    ax = np.asarray(ax).reshape(nrows, ncols)

    def plot1(a, y, label=None):
        if not len(y):
            return False
        kw = {"marker": 'o'} if len(y) == 1 else {}
        if label is not None:
            kw["label"] = label
        a.plot(y, **kw)
        return True

    def res_max(u, v):
        L = min(len(u), len(v))
        return [max(u[i], v[i]) for i in range(L)] if L else []

    for r, layer in enumerate(layers):
        lr = residuals_by_layer[layer]
        for c, key in enumerate(ordered):
            a = ax[r, c]
            any_ = False
            if key in {'abs', 'rel'}:
                k1, k2 = ('r1', 'r2') if key == 'abs' else ('r1_rel', 'r2_rel')
                d1, d2 = lr.get(k1, []), lr.get(k2, [])
                if agg_sum:
                    lbl = r'$\sqrt{r_1^2+r_2^2}$' if key == 'abs' else r'$\max{r_1^{rel}, r_2^{rel}}$'
                    any_ |= plot1(a, res_max(d1, d2), lbl)
                else:
                    lbl1, lbl2 = (r'$r_1$', r'$r_2$') if key == 'abs' else (r'$r_1^{rel}$', r'$r_2^{rel}$')
                    any_ |= plot1(a, d1, lbl1)
                    any_ |= plot1(a, d2, lbl2)
                if any_:
                    if yscale:
                        a.set(yscale='log', title=f"Layer {layer}-{'Abs' if key == 'abs' else 'Rel'}", xlabel='iter')
                    a.grid(True, which='both', ls='--', alpha=0.4)
                    a.legend()
            elif key == 'z_norm':
                any_ |= plot1(a, lr.get('z_norm', []))
                if any_:
                    a.set(title=f"Layer {layer}-||Z||", xlabel='iter')
                    a.grid(True, which='both', ls='--', alpha=0.4)
            elif key == 'y_norm':
                any_ |= plot1(a, lr.get('y_norm', []))
                if any_:
                    a.set(title=f"Layer {layer}-||Y||", xlabel='iter')
                    a.grid(True, which='both', ls='--', alpha=0.4)
            else:  # dual
                any_ |= plot1(a, lr.get('dual_vals', []))
                if any_:
                    if dual_scale:
                        a.set_yscale('symlog', linthresh=1e-6)
                    a.set(title=f"Layer {layer}-dual", xlabel='iter')
                    a.grid(True, which='both', ls='--', alpha=0.4)

            if not any_:
                a.axis('off')

    plt.tight_layout()



def train(train_dataloader, val_dataloader, model, optimizer, training_params, 
          logging_params, scheduler=None, ckpt_dir="", wandb_run=None,
          number_of_batches=np.inf):
    typedict = {"float16":torch.float16, "float32":torch.float32, "bfloat16":torch.bfloat16}

    record_pdhg_info = [0, 10, 100]
    world_size, rank, local_rank, device  = get_worker_info()
    master_process = (rank == 0)
    logger = Logging()
    logger.pdhg_residuals = []
    optimizer_name = optimizer.__class__.__name__
    if 'momo' in optimizer_name.lower() or 'nesgd' in optimizer_name.lower():
        pass_loss = True
    else:
        pass_loss = False
    if master_process: print(f"Set pass_loss to {pass_loss} for optimizer {optimizer_name}")

    autocast_ctxt = contextlib.nullcontext()
    if training_params['autocast']:
        autocast_ctxt = torch.autocast(device_type=device, dtype=typedict[training_params['mixed_precision']])     
    B, T = training_params['batch_size'], training_params['context_length']
    grad_accum_steps = int(training_params['tokens_processed'] / (world_size*B*T))
    val_accum_steps = int(logging_params['val_tokens_processed'] / (world_size*B*T))
    if master_process: print(f"Accumulate gradient for {grad_accum_steps} steps")
    total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
    max_grad_norm = training_params['gradnorm'] if training_params['gradnorm'] != 0. else float('inf')

    load_ckpt_step = logging_params['load_ckpt_step']
    if load_ckpt_step != 0:
        model, optimizer, train_dataloader, scheduler = load_checkpoint(ckpt_dir, load_ckpt_step, model, \
                                                        optimizer, train_dataloader, scheduler=None)
    if ckpt_dir == "":
        print("Will not save checkpoints as no directory is specified")

    # Training loop
    for epoch in range(training_params['num_epochs']):
        if master_process:
            print(f"Epoch {epoch+1} of {training_params['num_epochs']}")

        model.train()
        start_epoch = time.time()
        start_time = time.time() 
        loss_accum = 0.
        step = 1 if load_ckpt_step == 0 else int(load_ckpt_step)  # micro-step counter
        opt_step = 0 if load_ckpt_step == 0 else int(load_ckpt_step) // grad_accum_steps  # optimizer step counter
        optimizer.zero_grad(set_to_none=True)
        if step != 1 and master_process:
            print(train_dataloader.get_state())
            print(f"Resuming from micro_step={step}, opt_step={opt_step}")

        for batch in train_dataloader:
            with autocast_ctxt:
                output = model(input_ids=batch[0], labels=batch[1])
                loss = (output.loss if hasattr(output, "loss") else output[1])
                loss /= grad_accum_steps

            loss_accum += loss.detach()
                
            # Check if accummulated enough gradients to take a step
            if step % grad_accum_steps != 0:
                with (model.no_sync() if world_size > 1 else contextlib.nullcontext()):
                    loss.backward()
            else:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if world_size > 1: dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                if pass_loss:
                    _, pdhg_residuals = optimizer.step(closure=None, loss=loss_accum)
                else:
                    _, pdhg_residuals = optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

                if step % grad_accum_steps in record_pdhg_info:
                    logger.pdhg_residuals.append(pdhg_residuals)
                    
                #bookkeeping
                torch.cuda.synchronize()
                step_time = time.time() - start_time
                # Count an optimizer step
                opt_step += 1
                if master_process and wandb_run is not None:
                    wandb_log_dict = {
                        "train/loss": loss_accum.item(), 
                        "train/grad_norm": norm.item(),
                        "train/step_time": step_time,
                        "train/step": opt_step,
                        "train/micro_step": step
                    }
                    if getattr(model.config, "record_kq_max", False):
                        base_model = getattr(model, "module", model)
                        kq_max = None
                        for m in base_model.modules():
                            if isinstance(m, CausalSelfAttention) and getattr(m, "kq_max", None) is not None:
                                v = m.kq_max
                                if kq_max is None or v > kq_max:
                                    kq_max = v
                        if kq_max is not None:
                            wandb_log_dict["train/kq_max"] = kq_max
                            logger.kq_max.append(kq_max)
                    if hasattr(optimizer, 'step_size_list'):
                        wandb_log_dict["train/step_size_list"] = optimizer.step_size_list
                    for param_group_ix, param_group in enumerate(optimizer.param_groups):
                        wandb_log_dict[f"train/lr_{param_group_ix}"] = param_group['lr']
                    wandb_run.log(wandb_log_dict)
                logger.step_times.append(step_time)  # Are these different across ranks?
                logger.grad_norms.append(norm.item())
                for param_group in optimizer.param_groups:
                    logger.learning_rates.append(param_group['lr'])
                logger.losses.append(loss_accum.item())
                if hasattr(optimizer, 'step_size_list'):  
                    logger.step_size_list = optimizer.step_size_list  
                
                if (opt_step % logging_params['log_step'] == 0) & master_process:
                    tps = training_params["tokens_processed"] / step_time
                    print(f"Step {opt_step} of {total_iterations} (optimizer steps).")
                    print(f"Time taken : {step_time*1000:0.1f}ms | Tokens/s : {tps/1000:0.1f}k | Loss : {loss_accum.item():0.3f} | Accum: {grad_accum_steps} micro-steps/opt-step")
                    
                if (opt_step % logging_params['val_step'] == 0):
                    val_loss = eval_validation_loss(model, val_dataloader, val_accum_steps, autocast_ctxt)
                    if master_process and wandb_run is not None:
                        wandb_run.log({"val/loss": val_loss.item(), "val/step": opt_step, "val/micro_step": step})
                    logger.val_losses.append(val_loss.item())

                if (opt_step % logging_params['save_ckpt_step'] == 0) & (ckpt_dir != ""):
                    save_checkpoint(ckpt_dir, step, model, optimizer, loss_accum.item(),
                                    train_dataloader, scheduler, logging_params['keep_last'])
                    
                    if master_process:
                        with open(ckpt_dir + '/log.json', 'w') as file:
                            json.dump(logger.__dict__, file)
                loss_accum = 0.
                start_time = time.time() 
                if step >= number_of_batches:
                    break 
            step += 1
            
            
        print(f"In rank: {rank}, epoch {epoch+1}, Train Loss: {logger.losses[-1]}")
        print(f"In rank: {rank}, time taken for epoch {epoch+1} : ", time.time() - start_epoch)
        
        # Evaluate on val set, and save final values
        val_dataloader.reset()
        val_loss = eval_validation_loss(model, val_dataloader, 0, autocast_ctxt)
        logger.val_losses.append(val_loss.item())
        print(f"In rank: {rank}, epoch {epoch+1}, Validation Loss: {val_loss.item()}") 
        if getattr(model.config, "record_kq_max", False):
            base_model = getattr(model, "module", model)
            kq_max = None
            for m in base_model.modules():
                if isinstance(m, CausalSelfAttention) and getattr(m, "kq_max", None) is not None:
                    v = m.kq_max
                    if kq_max is None or v > kq_max:
                        kq_max = v  
                        logger.val_kq_max.append(kq_max)     
        if (ckpt_dir != ""):
            save_checkpoint(ckpt_dir, step, model, optimizer, logger.losses[-1],
                        train_dataloader, scheduler, logging_params['keep_last'])        
            if master_process:
                with open(ckpt_dir + '/log.json', 'w') as file:
                    json.dump(logger.__dict__, file)
        if master_process and wandb_run is not None:
            wandb_log_dict = {
                "val/loss": val_loss.item(),
                "val/step": opt_step,
                "val/micro_step": step,
                "train/loss": logger.losses[-1],
                "train/step": opt_step,
                "train/micro_step": step,
            }
            if getattr(model.config, "record_kq_max", False) and kq_max is not None:
                    wandb_log_dict["val/kq_max"] = kq_max
            wandb_run.log(wandb_log_dict)

    if hasattr(optimizer, 'step_size_list'):      # Check if optimizer has a step_size_list attribute
        logger.step_size_list = optimizer.step_size_list  
    return logger


@hydra.main(version_base=None, config_path="/mnt/home/tparshakova/Documents/GPT-opt/hydra_conf", config_name="config")
def main(config: DictConfig):
    set_seed(42)

    # Establish Hydra run directory for saving outputs
    hydra_run_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(hydra_run_dir, exist_ok=True)
    print(f"Hydra run directory: {hydra_run_dir}")

    # First set up DDP
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        dist.init_process_group(backend="nccl")
    world_size, rank, local_rank, device = get_worker_info()
    master_process = rank == 0  # this process will do logging, checkpointing etc.
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    print(f"Using device: {device}")

    # Set the training parameters
    training_params = config["training"]["training_params"]
    opt_config = config["optimizer"]["optimizer_params"]
    logging_config = config["logging"]["logging_params"]
    model_config = config["model"]["config"]
    model_name = config["model"]["name"]
    # Logging
    outputname = HydraConfig.get().job.config_name
    # Save results into Hydra's run directory for this job
    logging_config["results_dir"] = hydra_run_dir
    output_dir = hydra_run_dir

    CKPT_DIR = logging_config["ckpt_dir"]
    ckpt_dir_base = f"{CKPT_DIR}/{outputname}/" if CKPT_DIR != "" else ""
    if master_process:
        print(f"Training on dataset {config['data']['dataset']['name']}")
        os.makedirs(output_dir, exist_ok=True)
        if CKPT_DIR != "":
            os.makedirs(ckpt_dir_base, exist_ok=True)

    # Load model
    model = load_model(model_name, model_config, device)
    torch.set_float32_matmul_precision(training_params["tensorcore_precision"])

    # Load data
    data_dir = get_data_dir(config["data"]["dataset"]["name"])
    dataset_path = data_dir + f"/{config['data']['dataset']['name']}-gpt2/"
    if master_process:
        print(f"Load data from {dataset_path}")
    B, T = training_params["batch_size"], training_params["context_length"]
    assert training_params["tokens_processed"] % (world_size * B * T) == 0
    num_microbatches = int(
        training_params["tokens_processed"] / (world_size * B * T)
    )

    train_dataloader = ShardedDataLoader(dataset_path, B, T, "train", device)
    val_dataloader = ShardedDataLoader(dataset_path, B, T, "val", device)
    total_iterations = int(
        training_params["num_epochs"]
        * len(train_dataloader)
        / training_params["tokens_processed"]
    )
    if master_process:
        print(
            f"Length of train dataset : {len(train_dataloader)/1e6:0.1f} million tokens"
        )
        print(
            f"Length of validation dataset : {len(val_dataloader)/1e6:0.1f} million tokens"
        )
        print(f"Total number of iterations : {total_iterations}")

    print()
    if master_process:
        print(
            f"Training with optimizer {opt_config['name']} and learning rate {opt_config['lr']}"
        )

    # Generate hash for the current optimizer configuration
    config_hash = hash_config(
        OmegaConf.to_container(opt_config),
        OmegaConf.to_container(training_params),
        OmegaConf.to_container(model_config),
    )
    file_name = (
        f"{opt_config['name']}-lr-{opt_config['lr']}-{opt_config['lr_schedule']}"
    )
    if "muon_lr" in opt_config:
        file_name += f"-muonlr-{opt_config['muon_lr']}"
    file_name += f"-{config_hash}"
    output_path = os.path.join(output_dir, file_name + ".json")
    ckpt_dir = (
        os.path.join(ckpt_dir_base, file_name) + "/" if CKPT_DIR != "" else ""
    )

    # copy model to ensure consistency
    model_copy = copy.deepcopy(model).to(device)
    opt_name = opt_config["name"]
    # Setup optimizer: allow using local AttnPDAdamW in this test
    if opt_name in ["attn_pd_adamw_warm_start", "attn_pd_adamw", "attn_pd_adamw_warm_start_only", 
                    "attn_fista_adamw", "attn_rehpdhg_adamw"]:
        lr = float(opt_config.get("lr", 1e-3))
        betas = tuple(opt_config.get("betas", (0.9, 0.999)))
        eps = float(opt_config.get("eps", 1e-8))
        weight_decay = float(opt_config.get("weight_decay", 0.01)) 
        rho_over_lr = float(opt_config.get("rho_over_lr", 1.0))

        optimizer = AttnPDAdamW(
            model_copy.named_parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay, 
            rho_over_lr=rho_over_lr,
            diag_scaling=opt_config.get("diag_scaling", False),
            attn_max_iter=opt_config.get("attn_max_iter", 50), 
            attn_momentum=opt_config.get("attn_momentum", ""),
            pd_type=opt_config.get("pd_type", "pdhg"), 
            reflected_halpern=opt_config.get("reflected_halpern", False),
            warm_start=opt_config.get("warm_start", False),
            enable_restart=opt_config.get("enable_restart", False),
            lsqr_max_iter=opt_config.get("lsqr_max_iter", 1000),
        )
        use_my_adamw = True
    elif opt_name == "my_adamw":
        lr = float(opt_config.get("lr", 1e-3))
        betas = tuple(opt_config.get("betas", (0.9, 0.999)))
        eps = float(opt_config.get("eps", 1e-8))
        weight_decay = float(opt_config.get("weight_decay", 0.01)) 
        rho_over_lr = float(opt_config.get("rho_over_lr", 1.0))
        optimizer = MyAdamW(
            model_copy.named_parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay, 
        )
        use_my_adamw = True
    else:
        optimizer_obj, hyperp = get_optimizer(opt_config, lr=opt_config["lr"])
        use_my_adamw = False

    if training_params["compile"]:
        if master_process:
            print("Compiling model")
        model_copy = torch.compile(model_copy)

    if ddp:
        model_copy = DDP(model_copy, device_ids=[local_rank])

    if not use_my_adamw:
        p = (
            model_copy.named_parameters()
            if ("muon" in opt_name or "scion" in opt_name)
            else model_copy.parameters()
        )
        optimizer = optimizer_obj(p, **hyperp)

    scheduler = get_scheduler(
        opt_config, optimizer, total_iterations=total_iterations
    )

    # No wandb
    wandb_run = None

    # Train
    logger = train(
            train_dataloader,
            val_dataloader,
            model_copy,
            optimizer,
            training_params,
            scheduler=scheduler,
            ckpt_dir=ckpt_dir,
            logging_params=logging_config,
            wandb_run=wandb_run)

    # Save
    if master_process:
        logger.name = opt_config["name"] + "-lr-" + str(opt_config["lr"])
        if "muon_lr" in opt_config:
            logger.name += f"-muonlr-{opt_config['muon_lr']}"
        if "muon_lr_ratio" in opt_config:
            logger.name += f"-muonlr_ratio-{opt_config['muon_lr_ratio']}"
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Overwriting")
        with open(output_path, "w") as file:
            json.dump(logger.__dict__, file)
        print(f"Saved output to {output_path}")

    if ddp:
        dist.destroy_process_group() 


def load_sweep_jsons(base_dir) -> pd.DataFrame:
    """Recursively load all JSON logger files from the PDHG sweep.

    Each record includes:
      - path: full path to the JSON file
      - lr: learning rate parsed from the directory name bs-4-lr-*-wd-*
      - wd: weight decay parsed from the directory name
      - rho_over_lr: parsed from filename pattern ...-maxnorm-<value>-<hash>.json, if present
      - final_train_loss: last value in logger.losses
      - min_val_loss: minimum of logger.val_losses (NaN if empty)
    """
    pattern = os.path.join(base_dir, "**", "*.json")
    files = sorted(glob(pattern, recursive=True))
    records = []

    for path in files:
        try:
            with open(path, "r") as f:
                d = json.load(f)
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")
            continue

        losses = d.get("losses", [])
        val_losses = d.get("val_losses", [])
        if not losses:
            continue

        final_train_loss = losses[-1]
        min_val_loss = min(val_losses) if val_losses else float("nan")

        # Parse lr and wd from directory name: bs-4-lr-0.003-wd-0
        lr = None
        wd = None
        dir_name = os.path.basename(os.path.dirname(path))
        parts = dir_name.split("-")
        for i, p in enumerate(parts):
            if p == "lr" and i + 1 < len(parts):
                try:
                    lr = float(parts[i + 1])
                except ValueError:
                    pass
            if p == "wd" and i + 1 < len(parts):
                try:
                    wd = float(parts[i + 1])
                except ValueError:
                    pass

        # Parse rho_over_lr from filename: ...-maxnorm-0.001-<hash>.json
        rho_over_lr = None
        fname_parts = os.path.basename(path).split("-")
        for i, p in enumerate(fname_parts):
            if p == "maxnorm" and i + 1 < len(fname_parts):
                val_str = fname_parts[i + 1]
                # Strip extension if it somehow ended up attached
                val_str = val_str.replace(".json", "")
                try:
                    rho_over_lr = float(val_str)
                except ValueError:
                    pass

        records.append({
            "path": path,
            "lr": lr,
            "wd": wd,
            "rho_over_lr": rho_over_lr,
            "final_train_loss": final_train_loss,
            "min_val_loss": min_val_loss,
        })

    return pd.DataFrame(records)



def plot_sweep(df_in: pd.DataFrame) -> None:
    """Plot sweep metrics vs lr for different rho_over_lr values, and a 2D heatmap.

    Expects df_in to have columns:
      - lr
      - rho_over_lr
      - final_train_loss
      - min_val_loss
    """
    df_plot = df_in.dropna(subset=["lr", "rho_over_lr"]).copy()
    if df_plot.empty:
        print("No rows with both lr and rho_over_lr.")
        return

    print("Number of runs used for plots:", len(df_plot))

    # 1) Line plots: metric vs lr, colored by rho_over_lr
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    for maxn, sub in df_plot.groupby("rho_over_lr"):
        label = f"maxnorm={maxn:g}"
        sub_sorted = sub.sort_values("lr")
        ax[0].plot(sub_sorted["lr"], sub_sorted["final_train_loss"], marker="o", label=label)
        ax[1].plot(sub_sorted["lr"], sub_sorted["min_val_loss"], marker="o", label=label)

    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    ax[0].set_xlabel("lr")
    ax[1].set_xlabel("lr")
    ax[0].set_ylabel("final train loss")
    ax[1].set_ylabel("min val loss")
    ax[0].set_title("Train loss vs lr")
    ax[1].set_title("Val loss vs lr")
    for a in ax:
        a.grid(True, which="both", ls="--", alpha=0.4)
    ax[1].legend()
    plt.tight_layout()

    # 2) 2D heatmap: min_val_loss over (lr, rho_over_lr)
    if df_plot["lr"].nunique() > 1 and df_plot["rho_over_lr"].nunique() > 1:
        pivot = df_plot.pivot_table(index="rho_over_lr", columns="lr", values="min_val_loss")
        plt.figure(figsize=(6, 5))
        im = plt.imshow(pivot.values, aspect="auto", origin="lower")
        plt.colorbar(im, label="min val loss")
        plt.xticks(range(len(pivot.columns)), [f"{x:.2e}" for x in pivot.columns], rotation=45)
        plt.yticks(range(len(pivot.index)), [f"{y:.2e}" for y in pivot.index])
        plt.xlabel("lr")
        plt.ylabel("rho_over_lr")
        plt.title("Sweep: min val loss over (lr, rho_over_lr)")
        plt.tight_layout()
    else:
        print("Not enough variation in lr/rho_over_lr for a heatmap.")
