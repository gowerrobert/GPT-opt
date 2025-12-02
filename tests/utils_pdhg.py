import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import torch
import contextlib, json, time
from glob import glob
import pandas as pd

from gptopt.optim.pdhg import prox_l1, pdhg_method_AB, fista_ls_l1_reg, AttnPDAdamW
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

from gptopt.optim.pdhg import *


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
    Z1, Z2 = cp.Variable(A.shape), cp.Variable(B.shape)
    obj = cp.trace(G1.T @ Z1) + cp.trace(G2.T @ Z2)
    if mu > 0:
        obj += (mu / 2) * (cp.sum_squares(Z1) + cp.sum_squares(Z2))
    objective = cp.Minimize(obj)
    constraints = [cp.max(cp.abs(Z1.T @ B + A.T @ Z2)) <= beta]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=verbose)
    assert prob.status in ["optimal", "optimal_inaccurate"], print(prob.status)
    return Z1.value, Z2.value, obj.value


def compare_methods(prox_h_conj, h_conj, lamb_max, A, B, G1, G2, beta, mu_reg,  
                    f_star=None, max_iter=1000, stopping=True, pd_residuals=None):

    func_obj = lambda Z1, Z2: (torch.trace(G1.T @ Z1) + torch.trace(G2.T @ Z2) \
                            + (mu_reg / 2) * ((Z1).pow(2).sum() + Z2.pow(2).sum())).item()  
    func_constr_viol = lambda Z1, Z2: max(torch.max(torch.abs(Z1.T @ B + A.T @ Z2)).item() - beta, 0) / beta

    Z1_t_diag_scaling, Z2_diag_scaling, residuals_diag_scaling, _ = pdhg_method_AB(
                prox_h_conj,
                W_k=A,
                W_q=B,
                G_wk=G1,
                G_wq=G2,
                mu=mu_reg,
                beta=beta,
                max_iter=max_iter,
                eps_abs=1e-8,
                eps_rel=1e-8,
                stopping=stopping,
                h_conj=h_conj,
                f_star=f_star,
                diag_scaling=True,
                pd_residuals=pd_residuals
            )

    print("obj (diag scaling): ", func_obj(Z1_t_diag_scaling, Z2_diag_scaling), 
        "\nconstraint viol (diag scaling): ", func_constr_viol(Z1_t_diag_scaling, Z2_diag_scaling))


    Z1_t_vanilla, Z2_vanilla, residuals_vanilla, _ = pdhg_method_AB(
                prox_h_conj,
                W_k=A,
                W_q=B,
                G_wk=G1,
                G_wq=G2,
                mu=mu_reg,
                beta=beta,
                max_iter=max_iter,
                eps_abs=1e-8,
                eps_rel=1e-8,
                stopping=stopping,
                h_conj=h_conj,
                f_star=f_star, 
                pd_residuals=pd_residuals
            )
    print("obj (vanilla): ", func_obj(Z1_t_vanilla, Z2_vanilla), 
        "\nconstraint viol (vanilla): ", func_constr_viol(Z1_t_vanilla, Z2_vanilla))


    Z1_t_acceleration, Z2_acceleration, residuals_acceleration, _ = pdhg_method_AB(
                prox_h_conj,
                W_k=A,
                W_q=B,
                G_wk=G1,
                G_wq=G2,
                mu=mu_reg,
                beta=beta,
                max_iter=max_iter,
                eps_abs=1e-8,
                eps_rel=1e-8,
                stopping=stopping,
                h_conj=h_conj,
                f_star=f_star, 
                acceleration=True,
                pd_residuals=pd_residuals
            )

    print("obj (acceleration): ", func_obj(Z1_t_acceleration, Z2_acceleration), 
        "\nconstraint viol (acceleration): ", func_constr_viol(Z1_t_acceleration, Z2_acceleration))


    Y_fista, Z1_fista, Z2_fista, residuals_fista = fista_ls_l1_reg(W_k=A, W_q=B, G_wk=G1,
                                G_wq=G2, beta=beta, mu=mu_reg, lamb_max=lamb_max, max_iter=max_iter, 
                                eps_abs=1e-8, eps_rel=1e-8, f_star=f_star, stopping=stopping,
                                pd_residuals=pd_residuals)

    print("obj (fista): ", func_obj(Z1_fista, Z2_fista), 
        "\nconstraint viol (fista): ", func_constr_viol(Z1_fista, Z2_fista))


    residuals = {'PDHG': residuals_vanilla,
                "PDHG DS": residuals_diag_scaling,
                "PDHG Acc": residuals_acceleration,
                "FISTA": residuals_fista}

    return residuals


def plot_residuals_grid_by_mu(res_all, dual_scale=False, dpi=120,
                              abs_ylim=None, rel_ylim=None, gap_ylim=None, dual_ylim=None):
    return plot_residuals_grid_by_param(
        res_all, param_name="mu", dual_scale=dual_scale, dpi=dpi,
        abs_ylim=abs_ylim, rel_ylim=rel_ylim, gap_ylim=gap_ylim, dual_ylim=dual_ylim
    )


def plot_residuals_grid_by_param(res_all, param_name="mu", dual_scale=False, dpi=120,
                                 abs_ylim=None, rel_ylim=None, gap_ylim=None, dual_ylim=None):
    plabel = "μ" if param_name in {"mu", "μ"} else ("β" if param_name in {"beta", "β"} else str(param_name))
    pvals = sorted(res_all) if len(res_all) else []
    methods = sorted({m for D in res_all.values() for m in D})
    colors = {m: plt.cm.tab10(i / max(1, len(methods)-1)) for i, m in enumerate(methods)}
    sample = next(iter(res_all.values()), {})
    vals = sample.values() if isinstance(sample, dict) else []
    has_gap  = any("rel_gap"   in r and len(r.get("rel_gap", []))   for r in vals)
    has_dual = any("dual_vals" in r and len(r.get("dual_vals", [])) for r in vals)
    ncols = 4 if (has_gap and has_dual) else (3 if (has_gap or has_dual) else 2)

    fig, axs = plt.subplots(max(1, len(pvals)), ncols, figsize=(5*ncols, 3.2*max(1, len(pvals))), dpi=dpi)
    axs = np.atleast_2d(axs)

    specs = [
        (0, [("r1","-"),("r2","--")], abs_ylim, "Abs", "log"),
        (1, [("r1_rel","-"),("r2_rel","--")], rel_ylim, "Rel", "log"),
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


def plot_residuals_compare(all_res, dpi=120, dual_scale=False,
                           abs_ylim=None, rel_ylim=None, gap_ylim=None, dual_ylim=None):
    methods = sorted(all_res.keys())
    cmap = plt.cm.tab10
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
    if G_in_range:
        Y0_np = np.random.randn(n, n) * std2
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

    matrix_details(A)
    matrix_details(B)
    matrix_details(G1)
    matrix_details(G2)
    
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


def plot_residuals_layers(residuals_by_layer, dual_scale=False):
    layers = sorted(residuals_by_layer.keys())
    has_dual = any('dual_vals' in v for v in residuals_by_layer.values())
    ncols = 3 if has_dual else 2 
    fig, ax = plt.subplots(len(layers), ncols, figsize=(4*ncols, 3*len(layers)), sharex=True)
    if len(layers) == 1: ax = ax.reshape(1, -1)
    specs = [('r1','r2','Abs',r'$r_1$',r'$r_2$'), ('r1_rel','r2_rel','Rel',r'$r_1^{rel}$',r'$r_2^{rel}$')]
    for r, layer in enumerate(layers):
        lr = residuals_by_layer[layer]
        c = 0
        for k1,k2,title,l1,l2 in specs:
            d1,d2 = lr.get(k1,[]), lr.get(k2,[])
            if len(d1) or len(d2):
                if len(d1): ax[r,c].plot(d1,label=l1)
                if len(d2): ax[r,c].plot(d2,label=l2)
                ax[r,c].set(yscale='log',title=f"Layer {layer}-{title}",xlabel='iter'); ax[r,c].grid(True,which='both',ls='--',alpha=0.4); ax[r,c].legend()
                c += 1
        if has_dual and 'dual_vals' in lr:
            ax[r,c].plot(lr['dual_vals']); 
            if dual_scale:
                ax[r,c].set_yscale("symlog", linthresh=1e-6)
            # ax[r,c].set_yscale('symlog',linthresh=1e-6)
            ax[r,c].set(title=f"Layer {layer}-dual",xlabel='iter'); ax[r,c].grid(True,which='both',ls='--',alpha=0.4)
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
    if opt_name in ["attn_pd_adamw", "attn_fista_adamw"]:
        lr = float(opt_config.get("lr", 1e-3))
        betas = tuple(opt_config.get("betas", (0.9, 0.999)))
        eps = float(opt_config.get("eps", 1e-8))
        weight_decay = float(opt_config.get("weight_decay", 0.01))
        qk_lr_scale = float(opt_config.get("qk_lr_scale", 1.0))
        max_norm_tr = float(opt_config.get("max_norm_tr", 1.0))

        optimizer = AttnPDAdamW(
            model_copy.named_parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            qk_lr_scale=qk_lr_scale,
            max_norm_tr=max_norm_tr,
            diag_scaling=opt_config.get("diag_scaling", False),
            pdhg_max_iter=opt_config.get("pdhg_max_iter", 1000),
            momentum=opt_config.get("momentum", False),
            acceleration=opt_config.get("acceleration", False),
            pd_type=opt_config.get("pd_type", "pdhg"),
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
      - max_norm_tr: parsed from filename pattern ...-maxnorm-<value>-<hash>.json, if present
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

        # Parse max_norm_tr from filename: ...-maxnorm-0.001-<hash>.json
        max_norm_tr = None
        fname_parts = os.path.basename(path).split("-")
        for i, p in enumerate(fname_parts):
            if p == "maxnorm" and i + 1 < len(fname_parts):
                val_str = fname_parts[i + 1]
                # Strip extension if it somehow ended up attached
                val_str = val_str.replace(".json", "")
                try:
                    max_norm_tr = float(val_str)
                except ValueError:
                    pass

        records.append({
            "path": path,
            "lr": lr,
            "wd": wd,
            "max_norm_tr": max_norm_tr,
            "final_train_loss": final_train_loss,
            "min_val_loss": min_val_loss,
        })

    return pd.DataFrame(records)



def plot_sweep(df_in: pd.DataFrame) -> None:
    """Plot sweep metrics vs lr for different max_norm_tr values, and a 2D heatmap.

    Expects df_in to have columns:
      - lr
      - max_norm_tr
      - final_train_loss
      - min_val_loss
    """
    df_plot = df_in.dropna(subset=["lr", "max_norm_tr"]).copy()
    if df_plot.empty:
        print("No rows with both lr and max_norm_tr.")
        return

    print("Number of runs used for plots:", len(df_plot))

    # 1) Line plots: metric vs lr, colored by max_norm_tr
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    for maxn, sub in df_plot.groupby("max_norm_tr"):
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

    # 2) 2D heatmap: min_val_loss over (lr, max_norm_tr)
    if df_plot["lr"].nunique() > 1 and df_plot["max_norm_tr"].nunique() > 1:
        pivot = df_plot.pivot_table(index="max_norm_tr", columns="lr", values="min_val_loss")
        plt.figure(figsize=(6, 5))
        im = plt.imshow(pivot.values, aspect="auto", origin="lower")
        plt.colorbar(im, label="min val loss")
        plt.xticks(range(len(pivot.columns)), [f"{x:.2e}" for x in pivot.columns], rotation=45)
        plt.yticks(range(len(pivot.index)), [f"{y:.2e}" for y in pivot.index])
        plt.xlabel("lr")
        plt.ylabel("max_norm_tr")
        plt.title("Sweep: min val loss over (lr, max_norm_tr)")
        plt.tight_layout()
    else:
        print("Not enough variation in lr/max_norm_tr for a heatmap.")
