import torch
import cvxpy as cp





def cg_diag_precond(lin_op, Z, C, normC, diag_precond, maxit=1000, tol=1e-10, lambda_reg=0.0, verbose=True):
    # Initialization
    R  = C - lin_op(Z, lambda_reg=lambda_reg)              # residual
    Y  = diag_precond(R)               # preconditioned residual
    P  = Y.clone()

    rz_old = (R * Y).sum()

    for k in range(1, maxit+1):
        W = lin_op(P, lambda_reg=lambda_reg)
        denom = (P * W).sum()
        # Guard against breakdown
        if abs(denom) < 1e-12:
            if verbose: print(f"PCG breakdown at iter {k}")
            break

        alpha = (rz_old / denom).item()
        Z = Z + alpha * P
        R = R - alpha * W

        if R.pow(2).sum().sqrt().item() < tol * normC:
            print(f"{k=}")
            break

        Y = diag_precond(R)
        rz_new = (R * Y).sum()
        beta_cg = (rz_new / rz_old).item()
        P = Y + beta_cg * P
        rz_old = rz_new 
    return Z, k


def lsqr(A, AT, b, x, func_res=None, max_iter=1000, atol=1e-6, btol=1e-6, verbose=False):
    """
    LSQR min_x ||A x - b||_2 in PyTorch.
    """
    # Helper norms / dots over arbitrary shapes
    def norm(t):
        return torch.linalg.norm(t.reshape(-1))

    r = b.clone() - A(x)

    beta = norm(r)
    if beta == 0:
        return x, {"converged": True, "iter": 0, "res_norm": 0.0}

    u = r / beta
    v = AT(u)
    alpha = norm(v)
    if alpha == 0:
        # A^T b = 0 -> solution is x0 (minimum-norm)
        return x, {"converged": True, "iter": 0, "res_norm": beta.item()}

    v = v / alpha
    w = v.clone()

    # Scalars in LSQR
    phi_bar = beta
    rho_bar = alpha

    # For stopping criteria
    b_norm = beta
    res_norm = beta
    losses = []
    for k in range(1, max_iter + 1):
        # Bi-diagonalization step
        u = A(v) - alpha * u
        beta = norm(u)
        if beta > 0:
            u = u / beta

        v = AT(u) - beta * v
        alpha = norm(v)
        if alpha > 0:
            v = v / alpha

        # Construct and apply next orthogonal transformation
        rho = torch.sqrt(rho_bar**2 + beta**2)
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar

        # Update solution and search direction
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w

        # Residual norm estimate (LSQR standard estimate)
        res_norm = phi_bar
        if func_res is not None:
            rel_res = func_res(x)  
        else:
            rel_res = (res_norm / (b_norm + 1e-30)).item()

        losses += [rel_res]

        # Stopping criterion: ||A x - b|| <= atol + btol * ||b||
        if res_norm <= atol + btol * b_norm:
            return x, {
                "converged": True,
                "iter": k,
                "res_norm": res_norm.item(),
                "rel_res": rel_res,
                "loss": losses,
            }

    # Not converged within max_iter
    return x, {
        "converged": False,
        "iter": max_iter,
        "res_norm": res_norm.item(),
        "rel_res": rel_res,
        "loss": losses,
    }


@torch.no_grad()
def attn_least_squares_solve(*, A1, A2, G1, G2, X_type, beta=None, Y0=None, verbose=True, 
                    diag_scaling=True, maxit=1000, tol=1e-10, debug=False):
    """
    We solve M(X) = B using LSQR, where 
        M=A   if X=Z  
        M=A^* if X=Y
    diag_scaling: use R^{1/2}MG^{1/2} as an operator
    """  
    device = G1.device
    dtype  = G1.dtype
    m, n = A1.shape
    if diag_scaling:
        R, Gamma_1, Gamma_2 = pdhg_diagonal_scaling(A=A2, B=A1, eta=0.99, agg_op="l2_norm_sq")
        R_05, Gamma_1_05, Gamma_2_05 = torch.pow(R, 0.5), torch.pow(Gamma_1, 0.5), torch.pow(Gamma_2, 0.5)
        del R, Gamma_1, Gamma_2
    else:
        R_05 = torch.ones((n, n), device=device, dtype=dtype)
        Gamma_1_05 = torch.ones((m, 1), device=device, dtype=dtype) 
        Gamma_2_05 = torch.ones((m, 1), device=device, dtype=dtype)
    
    if X_type == "Z":   # A^*A
        R_05 = torch.ones((n, n), device=device, dtype=dtype)
    elif X_type == "Y": # AA^*
        Gamma_1_05 = torch.ones((m, 1), device=device, dtype=dtype) 
        Gamma_2_05 = torch.ones((m, 1), device=device, dtype=dtype)

    def A_linop(Z):         # A operator  
        return R_05 * ((Gamma_1_05 * Z[:m, :]).T @ A1 + A2.T @ ( Gamma_2_05 * Z[m:, :]))
    
    def AT_linop(Y):         # A^* operator  
        return torch.cat([Gamma_1_05 * A1 @ (R_05 * Y).T, Gamma_2_05 * A2 @ (R_05 * Y)], dim=0) 

    if debug:
        Y_test = torch.randn((n, n), device=device, dtype=dtype)
        Z_test = torch.randn((2 * m, n), device=device, dtype=dtype)
        lhs = (A_linop(Z_test) * Y_test).sum()
        rhs = (Z_test * AT_linop(Y_test)).sum()
        denom = lhs.abs() + rhs.abs() + 1e-12
        rel_err = ((lhs - rhs).abs() / denom).item()
        assert rel_err < 1e-5, f"Adjointness check failed: rel_err={rel_err:.3e}"

    if X_type == "Z":
        tilde_Z0  = torch.zeros((2 * m, n), device=device, dtype=dtype) # [Z1.T, Z2]
        S =  torch.sign(Y0) #* (Y0.abs() >= rho*beta)
        tilde_B = - R_05 * beta * S
        normB = beta * (S.pow(2).sum()).sqrt().item()
        if normB == 0.0: normB = 1.0
        func_res = lambda tilde_Z: torch.sqrt(((Gamma_1_05 * tilde_Z[:m, :]).T @ A1 + A2.T @ (Gamma_2_05 * tilde_Z[m:, :]) + beta * S).pow(2).sum()).item() / normB

        tilde_Z, dict_info = lsqr(A_linop, AT_linop, tilde_B, tilde_Z0, func_res=func_res, max_iter=maxit, 
                                  atol=1e-6, btol=1e-6, verbose=verbose)
        Z1, Z2 = tilde_Z[:m, :], tilde_Z[m:, :]
        del tilde_Z, tilde_Z0, tilde_B
        Z1, Z2 = Gamma_1_05 * Z1, Gamma_2_05 * Z2    
        X = (Z1, Z2)
        res = {"res" : torch.sqrt((Z1.T @ A1 + A2.T @ Z2 + beta * S).pow(2).sum()).item() / normB,
               "iter": dict_info["iter"],
               "loss": dict_info["loss"]}
        
    elif X_type == "Y":
        tilde_Y0  = torch.zeros((n, n), device=device, dtype=dtype) 
        tilde_B = - torch.cat([Gamma_1_05 * G1, Gamma_2_05 * G2], dim=0)
        normB = (G1.pow(2).sum() + G2.pow(2).sum()).sqrt().item()
        if normB == 0.0: normB = 1.0
        func_res = lambda tilde_Y: torch.sqrt((A1 @ (R_05 * tilde_Y).T + G1).pow(2).sum() + (A2 @ (R_05 * tilde_Y) + G2).pow(2).sum()).item() / normB

        tilde_Y, dict_info = lsqr(AT_linop, A_linop, b=tilde_B, x=tilde_Y0, func_res=func_res, max_iter=maxit, 
                                  atol=1e-6, btol=1e-6, verbose=verbose)
        Y = R_05 * tilde_Y
        del tilde_Y, tilde_Y0, tilde_B
        X = Y
        res = {"res" : torch.sqrt((A1 @ Y.T + G1).pow(2).sum() + (A2 @ Y + G2).pow(2).sum()).item() / normB,
               "iter": dict_info["iter"], 
               "loss": dict_info["loss"]}
    return X, res
    



@torch.no_grad()
def Z_sylvester_solve(*, A1, A2, Y0, beta, Z1_0=None, Z2_0=None, verbose=True, 
                    diag_scaling=True, maxit=1000, tol=1e-10, debug=False, lambda_reg0=0.0, 
                    method="lsqr"):
    """
    We solve A^*A (Z) = -beta * (A^* Y0)
    residual = ||Z1^TA1 + A2^TZ2 + beta * sign(Y0)||_F.
    """ 
    device = A1.device
    dtype  = A1.dtype
    n = A1.shape[1]
    assert Y0.shape == (n, n)
          
    S = torch.sign(Y0) 
    C   = - beta * torch.cat([S @ A1.T, A2 @ S], dim=0) 
    normC  = C.pow(2).sum().sqrt().item()
    lambda_reg = lambda_reg0 * normC
    if normC == 0.0: normC = 1.0
    # Small/medium case: solve via Kronecker.
    if method == "kron": 
        In = torch.eye(n, device=device, dtype=dtype) 
        K = torch.cat([torch.cat([torch.kron((A1 @ A1.T).contiguous(), In), torch.kron(A1, A2.T.contiguous())], dim=1),
                       torch.cat([torch.kron(A1.T.contiguous(), A2), torch.kron(In, (A2 @ A2.T).contiguous())], dim=1)], dim=0)  # [2(n^2) x 2(n^2)]
        rhs = - beta * torch.cat( [(S @ A1.T).T.reshape(-1),    # vec_col(C1)
                        (A2 @ S).T.reshape(-1)],                # vec_col(C2)
                        dim=0 )
        z = torch.linalg.lstsq(K, rhs).solution
        Z1, Z2 = z[:n**2].reshape(n, n), z[n**2:].reshape(n, n).T
        last_iter = 1
    elif method in ["cg", "lsqr"]:
        # Large case: solve via CG or LSQR
        def A_star_A(Z, lambda_reg=0):         # normal operator  
            return torch.cat([     Z[:n, :] @ A1 @ A1.T +      A2.T @ Z[n:, :] @ A1.T,
                              A2 @ Z[:n, :] @ A1        + A2 @ A2.T @ Z[n:, :]], dim=0) + lambda_reg * Z
        
        if diag_scaling:
            D1 = (A1 * A1).sum(dim=1).clamp_min(1e-12).unsqueeze(0)
            D2 = (A2 * A2).sum(dim=1).clamp_min(1e-12).unsqueeze(1)
            if debug: 
                assert torch.allclose(D1, torch.ones(n, 1, device=device, dtype=dtype) @ D1.reshape(1, -1))
                assert torch.allclose(D2, D2.reshape(-1, 1) @ torch.ones(1, n, device=device, dtype=dtype))
        else:
            D1, D2 = torch.ones(1, n, device=device, dtype=dtype), torch.ones(n, 1, device=device, dtype=dtype) 

        def diag_precond(R):      # elementwise divide by D1 + D2
            return torch.cat([R[:n, :] / D1, R[n:, :] / D2]) 
        
        def A_linop(Z):         # A operator  
            return Z[:n, :].T @ A1 + A2.T @ Z[n:, :]
        
        def AT_linop(Y):         # A^* operator  
            return torch.cat([A1 @ Y.T, A2 @ Y], dim=0)
        
        # Initialization
        if Z1_0 is not None and Z2_0 is not None: 
            Z  = torch.cat([Z1_0.T, Z2_0], dim=0)   # [Z1.T, Z2]
        else:       
            Z  = torch.zeros((2 * n, n), device=device, dtype=dtype) # [Z1.T, Z2]
        if method == "cg":
            Z, last_iter = cg_diag_precond(A_star_A, Z, C, normC, diag_precond, maxit=maxit, tol=tol, lambda_reg=lambda_reg, verbose=verbose)
            Z1, Z2 = Z[:n, :].T, Z[n:, :]
        else:
            B = -beta * S
            Z, dict_info = lsqr(A_linop, AT_linop, B, Z, max_iter=maxit, atol=1e-6, btol=1e-6, verbose=verbose)
            last_iter = dict_info["iter"]
            Z1, Z2 = Z[:n, :], Z[n:, :]
    normB = beta * (S.pow(2).sum()).sqrt().item()
    res = {"res_C":torch.sqrt((Z1.T @ A1 + A2.T @ Z2 + beta * S).pow(2).sum()).item() / normC,
           "res":torch.sqrt((Z1.T @ A1 + A2.T @ Z2 + beta * S).pow(2).sum()).item() / normB,
           "iter":last_iter}
    return Z1, Z2, res


def matcal_A_to_kron_Kron(A1, A2):
    A1t = A1.T.contiguous()
    A2t = A2.T.contiguous()
    n = A1.shape[1]
    In = torch.eye(n, device=A1.device, dtype=A1.dtype)
    K = torch.cat([torch.kron(A1t, In), torch.kron(In, A2t)], dim=1)
    return K

def matcal_AAT_to_kron_Kron(A1, A2):
    A1tA1 = A1.T @ A1              
    A2tA2 = A2.T @ A2 
    n = A1.shape[1]
    Ip = torch.eye(n, device=A1.device, dtype=A1.dtype)
    In = torch.eye(n, device=A1.device, dtype=A1.dtype)
    K = torch.kron(In, A2tA2) + torch.kron(A1tA1, Ip)  # [(n^2) x (n^2)]
    return K


@torch.no_grad()
def Y_dual_feasible(*, A1, A2, G1, G2, verbose=True, method="lsqr",
                    diag_scaling=True, maxit=1000, tol=1e-10, debug=False, lambda_reg0=0.0):
    """
    Check dual feasibility: exists Y s.t. A1 Y^T = -G1 and A2 Y = -G2

    We solve (A1^T A1)Y + Y(A1^T A1) = -(A2^T G2 + G1^T A1) for Y
    residual = sqrt( ||A1 Y^T + G1||_F^2 + ||A2 Y + G2||_F^2 ).
    """ 
    device = G1.device
    dtype  = G1.dtype
    m, n = A1.shape
    assert G1.shape == A1.shape and G2.shape == A2.shape and A1.shape == A2.shape
          
    C   = -(A2.T @ G2 + G1.T @ A1) 
    normC  = C.pow(2).sum().sqrt().item()
    lambda_reg = lambda_reg0 * normC
    if normC == 0.0: normC = 1.0
    # Small/medium case: solve via Kronecker.
    if method == "kron": 
        K = matcal_AAT_to_kron_Kron(A1, A2)  # [(n^2) x (n^2)]
        y = torch.linalg.lstsq(K, C.T.reshape(-1)).solution #torch.linalg.solve(K, C.T.reshape(-1))
        Y = y.reshape(n, n).T
        last_iter = 1
    else:
        # Large case: solve via CG
        def A_A_star(Y, lambda_reg=0):         # normal operator  
            return A2.T @ (A2 @ Y) + (Y @ A1.T) @ A1 + lambda_reg * Y

        if diag_scaling:
            D1 = (A1 * A1).sum(dim=0).clamp_min(1e-12).unsqueeze(0)
            D2 = (A2 * A2).sum(dim=0).clamp_min(1e-12).unsqueeze(1)
            if debug: 
                assert torch.allclose(D1 + D2, D2.reshape(-1, 1) @ torch.ones(1, n, device=device, dtype=dtype) \
                              + torch.ones(n, 1, device=device, dtype=dtype) @ D1.reshape(1, -1))
        else:
            D1, D2 = torch.ones(1, n, device=device, dtype=dtype), torch.ones(n, 1, device=device, dtype=dtype) 

        def diag_precond(R):      # elementwise divide by D1 + D2
            return R / (D1 + D2) 
        
        def A_linop(Z):         # A operator  
            return Z[:m, :].T @ A1 + A2.T @ Z[m:, :]
        
        def AT_linop(Y):         # A^* operator  
            return torch.cat([A1 @ Y.T, A2 @ Y], dim=0)
        
        # Initialization
        Y  = torch.zeros((n, n), device=device, dtype=dtype)

        if method == "cg":
            Y, last_iter = cg_diag_precond(A_A_star, Y, C, normC, diag_precond, maxit=maxit, tol=tol, lambda_reg=lambda_reg, verbose=verbose)
        else:
            B = -torch.cat([G1, G2], dim=0)
            Y, dict_info = lsqr(AT_linop, A_linop, b=B, x=Y, max_iter=maxit, atol=1e-6, btol=1e-6, verbose=verbose)
            last_iter = dict_info["iter"]
    normB = (G1.pow(2).sum() + G2.pow(2).sum()).sqrt().item()
    res = {"res_C":torch.sqrt((A1 @ Y.T + G1).pow(2).sum() + (A2 @ Y + G2).pow(2).sum()).item() / normC,
           "res":torch.sqrt((A1 @ Y.T + G1).pow(2).sum() + (A2 @ Y + G2).pow(2).sum()).item() / normB,
           "iter":last_iter}
    return Y, res


def cvxpy_Z_sylvester_solve(*, A1, A2, Y0, beta):
    """
    We solve A^*A (Z) = -beta * (A^* Y0)
    residual = ||Z1^TA1 + A2^TZ2 + beta * sign(Y0)||_F.
    """ 
    n = A1.shape[0]
    Z1, Z2 = cp.Variable((n, n)), cp.Variable((n, n))
    S = torch.sign(Y0).cpu().numpy()
    A1_np = A1.cpu().numpy()
    A2_np = A2.cpu().numpy()
    obj = cp.sum_squares(Z1.T @ A1_np + A2_np.T @ Z2 + beta * S)
    prob = cp.Problem(cp.Minimize(obj), [])
    prob.solve(solver=cp.CLARABEL)
    assert prob.status in ["optimal", "optimal_inaccurate"], print(prob.status)
    S = torch.sign(Y0)
    normC  = beta * (torch.cat([S @ A1.T, A2 @ S], dim=0)).pow(2).sum().sqrt().item() 
    if normC == 0.0: normC = 1.0
    normB = beta * (S.pow(2).sum()).sqrt().item()
    return Z1.value, Z2.value, {"res_C": obj.value**0.5 / normC, "res": obj.value**0.5 / normB, "iter": 1}


def cvxpy_Y_sylvester_solve(*, A1, A2, G1, G2):
    """
    We solve AA^* (Y) = -(A G)
    residual = sqrt(||A1 Y.T - G1||^2_F + ||A2 Y - G2||^2_F).
    """ 
    m, n = A1.shape
    Y = cp.Variable((n, n)) 
    A1_np = A1.cpu().numpy()
    A2_np = A2.cpu().numpy()
    G1_np = G1.cpu().numpy()
    G2_np = G2.cpu().numpy()
    obj = cp.sum_squares(A1_np @ Y.T + G1_np) + cp.sum_squares(A2_np @ Y + G2_np)
    prob = cp.Problem(cp.Minimize(obj), [])
    prob.solve(solver=cp.CLARABEL)
    assert prob.status in ["optimal", "optimal_inaccurate"], print(prob.status)
    normC  = (A2.T @ G2 + G1.T @ A1).pow(2).sum().sqrt().item() 
    if normC == 0.0: normC = 1.0
    normB = (G1.pow(2).sum() + G2.pow(2).sum()).sqrt().item()
    return Y.value, {"res_C": obj.value**0.5 / normC, "res": obj.value**0.5 / normB, "iter": 1}



def pdhg_diagonal_scaling(A, B, eta=0.99, eps=1e-8, debug=False, agg_op="l1_norm"):
    device, dtype = A.device, A.dtype
    p2, n = A.shape
    p1, nB = B.shape 

    if agg_op == "l1_norm":
        # |B| row/col sums 
        r_B = B.abs().sum(dim=1)                 # (p1,)
        c_B = B.abs().sum(dim=0)                 # (n,)

        # |A| row/col sums 
        r_A = A.abs().sum(dim=1)                 # (p2,)
        c_A = A.abs().sum(dim=0)                 # (n,)
    elif agg_op == "l2_norm_sq":
        # |B| row/col sums 
        r_B = B.pow(2).sum(dim=1)                 # (p1,)
        c_B = B.pow(2).sum(dim=0)                 # (n,)

        # |A| row/col sums 
        r_A = A.pow(2).sum(dim=1)                 # (p2,)
        c_A = A.pow(2).sum(dim=0)                 # (n,) 
 
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
