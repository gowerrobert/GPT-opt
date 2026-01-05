import numpy as np
import scipy.sparse as sp
from cupdlpx import Model, PDLP
import torch


def cupdlpx_AB_noX(G1t, G2t, At, Bt, beta, verbose=False, iter_limit=None,
                   time_limit=None, feasibility_tol=None, optimality_tol=None,
                   eps_feas=None, eps_opt=None,
                   feasibility_polishing=None, eps_feas_polish=None):
    A = np.asarray(At.cpu().detach(), dtype=np.float64)
    B = np.asarray(Bt.cpu().detach(), dtype=np.float64)
    G1 = np.asarray(G1t.cpu().detach(), dtype=np.float64)
    G2 = np.asarray(G2t.cpu().detach(), dtype=np.float64)

    m, n = A.shape 
    n_z = m * n
    n_con = n * n

    c = np.r_[G1.reshape(-1, order="C"), G2.reshape(-1, order="F")]

    I_n = sp.eye(n, format="csr", dtype=np.float64) 
    K  = sp.hstack([sp.kron(B.T, I_n, format="csr"),
                    sp.kron(I_n, A.T, format="csr")], format="csr")

    beta_vec = (np.full(n_con, float(beta), np.float64)
                if np.isscalar(beta) else np.asarray(beta, np.float64).reshape(-1, order="F"))

    inf = np.inf
    mdl = Model(
        objective_vector=c,
        constraint_matrix=K,
        constraint_lower_bound=-beta_vec,
        constraint_upper_bound=+beta_vec,
        variable_lower_bound=np.full(2 * n_z, -inf, np.float64),
        variable_upper_bound=np.full(2 * n_z, +inf, np.float64),
    )
    mdl.ModelSense = PDLP.MINIMIZE
    mdl.setParams(OutputFlag=bool(verbose))

    def _set(a, b, v):
        if v is None: return
        try: mdl.setParam(a, v)
        except Exception: mdl.setParam(b, v)

    _set("TimeLimit",      "time_limit",      None if time_limit      is None else float(time_limit))
    _set("IterationLimit", "iter_limit",      None if iter_limit      is None else int(iter_limit))
    _set("FeasibilityTol", "feasibility_tol", None if feasibility_tol is None else float(feasibility_tol))
    _set("OptimalityTol",  "optimality_tol",  None if optimality_tol  is None else float(optimality_tol))

    if eps_feas is not None: mdl.setParam("eps_feas", float(eps_feas))
    if eps_opt  is not None: mdl.setParam("eps_opt",  float(eps_opt))

    if feasibility_polishing is not None:
        try: mdl.setParam("feasibility_polishing", bool(feasibility_polishing))
        except Exception: mdl.setParam("FeasibilityPolishing", bool(feasibility_polishing))

    if eps_feas_polish is not None:
        try: mdl.setParam("eps_feas_polish", float(eps_feas_polish))
        except Exception: mdl.setParam("eps_feas_polish", float(eps_feas_polish))

    mdl.optimize()
    sol = mdl.X
    if sol is None:
        raise RuntimeError(f"No solution returned; status={mdl.Status} code={mdl.StatusCode}")

    Z1 = torch.from_numpy(sol[:n_z].reshape((m, n), order="C")).to(G1t.dtype).to(G1t.device)
    Z2 = torch.from_numpy(sol[n_z:2*n_z].reshape((m, n), order="F")).to(G1t.dtype).to(G1t.device)
    return Z1, Z2, float(mdl.ObjVal)



def cupdlpx_AB(G1t, G2t, At, Bt, beta, verbose=False, iter_limit=None,
               time_limit=None, feasibility_tol=None, optimality_tol=None,
               eps_feas=None, eps_opt=None, feasibility_polishing=None, 
               eps_feas_polish=None, presolve=False):
    """
    Solve:
        minimize  trace(G1^T Z1) + trace(G2^T Z2)
        s.t.      Z1^T B + A^T Z2 = X
                  -beta <= X <= beta
    using cuPDLPx (cupdlpx) with a vectorized (no for-loops) formulation.

    Returns:
        Z1, Z2, obj_value, dual_matrix_for_equality (same shape as X)
    """
    # --- inputs (float64) ---
    A = np.asarray(At.cpu().detach(), dtype=np.float64)
    B = np.asarray(Bt.cpu().detach(), dtype=np.float64)
    G1 = np.asarray(G1t.cpu().detach(), dtype=np.float64)
    G2 = np.asarray(G2t.cpu().detach(), dtype=np.float64)

    m, n = A.shape 
    assert A.shape == B.shape == G1.shape == G2.shape

    # --- variable vector layout ---
    # z1 := vec_C(Z1)  (row-major)  length m*n1
    # z2 := vec_F(Z2)  (col-major)  length m*n2
    # x  := vec_F(X)   (col-major)  length n1*n2
    n_z = m * n 
    n_x  = n**2 

    # --- objective c^T v ---
    c = np.concatenate([
        G1.reshape(-1, order="C"),   # matches vec_C(Z1)
        G2.reshape(-1, order="F"),   # matches vec_F(Z2)
        np.zeros(n_x, dtype=np.float64)
    ])

    # --- equality constraints: vec_F(Z1^T B) + vec_F(A^T Z2) - vec_F(X) = 0 ---
    I_n = sp.eye(n, format="csr", dtype=np.float64) 

    # vec_F(Z1^T B) = (B^T ⊗ I_n) vec_F(Z1^T) and vec_F(Z1^T) == vec_C(Z1)
    K1 = sp.kron(B.T, I_n, format="csr")          # (n*n) x (m*n)
    # vec_F(A^T Z2) = (I_n ⊗ A^T) vec_F(Z2)
    K2 = sp.kron(I_n, A.T, format="csr")          # (n*n) x (m*n)

    K3 = -sp.eye(n_x, format="csr", dtype=np.float64)

    Aeq = sp.hstack([K1, K2, K3], format="csr")
    constr_lb = np.zeros(n_x, dtype=np.float64)
    constr_ub = np.zeros(n_x, dtype=np.float64)

    # --- variable bounds ---
    if np.isscalar(beta):
        beta_vec = np.full(n_x, float(beta), dtype=np.float64)
    else:
        beta_arr = np.asarray(beta, dtype=np.float64)
        if beta_arr.shape != (n, n):
            raise ValueError(f"beta must be scalar or shape {(n, n)}; got {beta_arr.shape}")
        beta_vec = beta_arr.reshape(-1, order="F")  # matches vec_F(X)

    var_lb = np.concatenate([
        np.full(n_z, -np.inf, dtype=np.float64),   # Z1 free
        np.full(n_z, -np.inf, dtype=np.float64),   # Z2 free
        -beta_vec                                  # X lower
    ])
    var_ub = np.concatenate([
        np.full(n_z, +np.inf, dtype=np.float64),
        np.full(n_z, +np.inf, dtype=np.float64),
        +beta_vec
    ])

    # --- solve ---
    mdl = Model(
        objective_vector=c,
        constraint_matrix=Aeq,
        constraint_lower_bound=constr_lb,
        constraint_upper_bound=constr_ub,
        variable_lower_bound=var_lb,
        variable_upper_bound=var_ub,
    )
    mdl.ModelSense = PDLP.MINIMIZE  

    # basic params
    mdl.setParams(OutputFlag=bool(verbose))
    if time_limit is not None:
        mdl.setParam("TimeLimit", float(time_limit))
    if iter_limit is not None:
        mdl.setParam("IterationLimit", int(iter_limit))
    if feasibility_tol is not None:
        mdl.setParam("FeasibilityTol", float(feasibility_tol))
    if optimality_tol is not None:
        mdl.setParam("OptimalityTol", float(optimality_tol)) 

    # tolerances (relative)
    if eps_feas is not None:
        mdl.setParam("eps_feas", float(eps_feas))   # relative feasibility tolerance :contentReference[oaicite:1]{index=1}
    if eps_opt is not None:
        mdl.setParam("eps_opt", float(eps_opt))     # relative optimality tolerance :contentReference[oaicite:2]{index=2}

    # optional post-solve feasibility polishing (if your build exposes these params)
    if feasibility_polishing is not None:
        mdl.setParam("feasibility_polishing", bool(feasibility_polishing))  # :contentReference[oaicite:3]{index=3}
    if eps_feas_polish is not None:
        mdl.setParam("eps_feas_polish", float(eps_feas_polish))             # :contentReference[oaicite:4]{index=4}
    mdl.setParam("presolve", presolve)

    mdl.optimize()
    if mdl.Status not in ("OPTIMAL", "TIME_LIMIT", "ITERATION_LIMIT"):
        raise RuntimeError(f"cupdlpx status={mdl.Status} (code={mdl.StatusCode})")

    sol = mdl.X
    if sol is None:
        raise RuntimeError("No primal solution returned (mdl.X is None).")

    # --- unpack solution ---
    Z1 = torch.from_numpy(sol[:n_z].reshape((m, n), order="C")).to(G1t.dtype).to(G1t.device)
    Z2 = torch.from_numpy(sol[n_z:2*n_z].reshape((m, n), order="F")).to(G1t.dtype).to(G1t.device)

    # dual multipliers for equality constraints (same order as constr vector)
    Pi = mdl.Pi
    dual_mat = None if Pi is None else Pi.reshape((n, n), order="F")
    Y = -torch.from_numpy(dual_mat).to(G1t.dtype).to(G1t.device) 

    return Z1, Z2, float(mdl.ObjVal), Y
