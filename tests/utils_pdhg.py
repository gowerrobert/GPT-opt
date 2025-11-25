import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import torch

from gptopt.optim.pdhg import prox_l1, pdhg_method_AB




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
                    stopping: bool = True
                    ):
    """
    Fista for solving the dual problem:
    maximize -\frac{1}{2\mu}\|\mathcal{A}^*(Y) + G\|_F^2 - \beta \|vec(Y)\|_1
    \mathcal{A}^*(Y) = [W_q Y^T; W_k Y] or W_q Y if W_k is None
    """ 
    use_Z2 = not ((W_k is None) or (G_wq is None) ) 
    if lamb_max is None:
        nA = torch.linalg.norm(W_q, ord="fro").item()
        if use_Z2:
            nB = torch.linalg.norm(W_k, ord="fro").item()
            lamb_max = (nA * nA + nB * nB) ** 0.5
        else:
            lamb_max = nA
        print(f"{lamb_max=}")
    
    step_size = mu / lamb_max**2
    X0 = torch.zeros((G_wk.shape[1], W_q.shape[1]), device=G_wk.device, dtype=G_wk.dtype)
    X1 = X0.clone()
    residuals = {"r1": [],  "r1_rel": [], "dual_vals":[], "rel_gap":[]}
 
    for t in range(1, max_iter+1):
        # Extrapolation step
        Y = X1 + ((t - 2) / (t + 1)) * (X1 - X0)
        # Gradient step
        grad = (1/mu) * (W_q @ Y.T + G_wk).T @ W_q 
        if use_Z2:
            grad = grad + (1/mu) * (W_k.T @ (W_k @ Y + G_wq)) 
        # Prox step
        Y1 = prox_l1(Y - step_size * grad, beta * step_size)
        # FPI residual
        normalize = torch.linalg.norm(Y).item() 
        normalize = normalize if normalize > 1e-6 else 1.0
        res = (Y1 - Y).pow(2).sum().sqrt().item() 
        r_norm = res / normalize
        X0, X1 = X1, Y1
        # Logging
        residuals['r1'].append(res)
        residuals['r1_rel'].append(r_norm)
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

 
        if stopping and t >= 2 and r_norm < eps_abs + eps_rel * torch.linalg.norm(Y).item():
            print(f"Fista converged in {t} iterations.")
            break 
    return X1, residuals




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


def compare_methods(prox_h_conj, h_conj, lamb_max, A, B, G1, G2, beta, mu_reg, f_star=None, max_iter=1000, stopping=True):

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
                max_iter=max_iter,
                eps_abs=1e-8,
                eps_rel=1e-8,
                stopping=stopping,
                h_conj=h_conj,
                f_star=f_star,
                diag_scaling=True,
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
                max_iter=max_iter,
                eps_abs=1e-8,
                eps_rel=1e-8,
                stopping=stopping,
                h_conj=h_conj,
                f_star=f_star, 
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
                max_iter=max_iter,
                eps_abs=1e-8,
                eps_rel=1e-8,
                stopping=stopping,
                h_conj=h_conj,
                f_star=f_star, 
                acceleration=True,
            )

    print("obj (vanilla): ", func_obj(Z1_t_acceleration, Z2_acceleration), 
        "\nconstraint viol (vanilla): ", func_constr_viol(Z1_t_acceleration, Z2_acceleration))


    Y_fista, residuals_fista = fista_ls_l1_reg(W_k=A, W_q=B, G_wk=G1,
                G_wq=G2, beta=beta, mu=mu_reg, lamb_max=lamb_max, max_iter=max_iter, 
                eps_abs=1e-8, eps_rel=1e-8, f_star=f_star, stopping=stopping)
    
    # primal recovery
    Z1_fista = (1 / mu_reg) * (- G1 - B @ Y_fista.T)
    Z2_fista = (1 / mu_reg) * (- G2 - A @ Y_fista)

    print("obj (fista): ", func_obj(Z1_fista, Z2_fista), 
        "\nconstraint viol (fista): ", func_constr_viol(Z1_fista, Z2_fista))


    residuals = {'PDHG': residuals_vanilla,
                "PDHG DS": residuals_diag_scaling,
                "PDHG Acc": residuals_acceleration,
                "FISTA": residuals_fista}

    return residuals



def plot_residuals_compare(all_res, dpi=120):
    methods = list(all_res.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    ls_map = {"r1": "-", "r2": "--", "r1_rel": "-", "r2_rel": "--", "rel_gap": ":", "dual": "-."}

    has_gap  = any("rel_gap"   in r and len(r["rel_gap"])   for r in all_res.values())
    has_dual = any("dual_vals" in r and len(r["dual_vals"]) for r in all_res.values())
    if has_gap and has_dual:
        fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True, dpi=dpi); ax = ax.ravel()
    elif has_gap or has_dual:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharex=True, dpi=dpi)
    else:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True, dpi=dpi)
    ax = np.atleast_1d(ax)

    # Absolute residuals
    any_abs = False
    for (name, res), c in zip(all_res.items(), colors):
        r1, r2 = res.get("r1", []), res.get("r2", [])
        if len(r1):
            ax[0].plot(r1, color=c, ls=ls_map["r1"], label=f"{name} $r_1$")
            any_abs = True
        if len(r2):
            ax[0].plot(r2, color=c, ls=ls_map["r2"], label=f"{name} $r_2$")
            any_abs = True
    if any_abs:
        ax[0].set(yscale="log", title="Absolute residuals", xlabel="iteration")
        ax[0].grid(True, which="both", ls="--", alpha=0.4); ax[0].legend()
    else:
        ax[0].axis("off")

    # Relative residuals
    any_rel = False
    for (name, res), c in zip(all_res.items(), colors):
        r1r, r2r = res.get("r1_rel", []), res.get("r2_rel", [])
        if len(r1r):
            ax[1].plot(r1r, color=c, ls=ls_map["r1_rel"], label=f"{name} $r_1^{{rel}}$")
            any_rel = True
        if len(r2r):
            ax[1].plot(r2r, color=c, ls=ls_map["r2_rel"], label=f"{name} $r_2^{{rel}}$")
            any_rel = True
    if any_rel:
        ax[1].set(yscale="log", title="Relative residuals", xlabel="iteration")
        ax[1].grid(True, which="both", ls="--", alpha=0.4); ax[1].legend()
    else:
        ax[1].axis("off")

    # Relative gap
    if has_gap:
        i = 2 if (has_gap and not has_dual) else (2 if not has_dual else 2)
        any_gap = False
        for (name, res), c in zip(all_res.items(), colors):
            if "rel_gap" in res and len(res["rel_gap"]):
                ax[i].plot(res["rel_gap"], color=c, ls=ls_map["rel_gap"], label=name)
                any_gap = True
        if any_gap:
            ax[i].set(yscale="log", title="Relative gap", xlabel="iteration")
            ax[i].grid(True, which="both", ls="--", alpha=0.4); ax[i].legend()

    # Dual objective (symmetric log so negatives are ok)
    if has_dual:
        j = 2 if (has_dual and not has_gap) else (3 if has_gap else 2)
        any_dual = False
        for (name, res), c in zip(all_res.items(), colors):
            if "dual_vals" in res and len(res["dual_vals"]):
                ax[j].plot(res["dual_vals"], color=c, ls=ls_map["dual"], label=name)
                any_dual = True
        if any_dual:
            ax[j].set_yscale("symlog", linthresh=1e-6)
            ax[j].set(title="Dual objective", xlabel="iteration")
            ax[j].grid(True, which="both", ls="--", alpha=0.4); ax[j].legend()

    plt.tight_layout()



def gaussian_data(m, n):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A_np = np.random.randn(m, n)
    B_np = np.random.randn(m, n)
    Y0_np = np.random.randn(n, n)
    G1_np = B_np @ Y0_np.T
    G2_np = A_np @ Y0_np

    # Torch tensors
    A = torch.from_numpy(A_np).to(torch.float32).to(device)
    B = torch.from_numpy(B_np).to(torch.float32).to(device)
    G1 = torch.from_numpy(G1_np).to(torch.float32).to(device)
    G2 = torch.from_numpy(G2_np).to(torch.float32).to(device)

    nA = torch.linalg.norm(A, ord="fro").item()
    nB = torch.linalg.norm(B, ord="fro").item()
    lamb_max = (nA * nA + nB * nB) ** 0.5
    

    return A, B, G1, G2, A_np, B_np, G1_np, G2_np, lamb_max
