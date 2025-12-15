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


def lsqr(A, AT, b, x, max_iter=1000, atol=1e-6, btol=1e-6, verbose=False):
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

        rel_res = res_norm / (b_norm + 1e-30) 

        # Stopping criterion: ||A x - b|| <= atol + btol * ||b||
        if res_norm <= atol + btol * b_norm:
            return x, {
                "converged": True,
                "iter": k,
                "res_norm": res_norm.item(),
                "rel_res": rel_res.item(),
            }

    # Not converged within max_iter
    return x, {
        "converged": False,
        "iter": max_iter,
        "res_norm": res_norm.item(),
        "rel_res": (res_norm / (b_norm + 1e-30)).item(),
    }


torch.no_grad()
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
        z = torch.linalg.solve(K, rhs)
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

    res = {"res":torch.sqrt((Z1.T @ A1 + A2.T @ Z2 + beta * S).pow(2).sum()).item() / normC,
           "iter":last_iter}
    return Z1, Z2, res



torch.no_grad()
def Y_dual_feasible(*, A1, A2, G1, G2, verbose=True, method="lsqr",
                    diag_scaling=True, maxit=1000, tol=1e-10, debug=False, lambda_reg0=0.0):
    """
    Check dual feasibility: exists Y s.t. A1 Y^T = -G1 and A2 Y = -G2

    We solve (A1^T A1)Y + Y(A1^T A1) = -(A2^T G2 + G1^T A1) for Y
    residual = sqrt( ||A1 Y^T + G1||_F^2 + ||A2 Y + G2||_F^2 ).
    """ 
    device = A1.device
    dtype  = A1.dtype
    m, n = A1.shape
    assert G1.shape == A1.shape and G2.shape == A2.shape and A1.shape == A2.shape
          
    C   = -(A2.T @ G2 + G1.T @ A1) 
    normC  = C.pow(2).sum().sqrt().item()
    lambda_reg = lambda_reg0 * normC
    if normC == 0.0: normC = 1.0
    # Small/medium case: solve via Kronecker.
    if method == "kron":
        A1tA1 = A1.T @ A1              
        A2tA2 = A2.T @ A2 
        Ip = torch.eye(n, device=device, dtype=dtype)
        In = torch.eye(n, device=device, dtype=dtype)
        K = torch.kron(In, A2tA2) + torch.kron(A1tA1, Ip)  # [(n^2) x (n^2)]
        y = torch.linalg.solve(K, C.T.reshape(-1))
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

    res = {"res":torch.sqrt((A1 @ Y.T + G1).pow(2).sum() + (A2 @ Y + G2).pow(2).sum()).item() / normC,
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
    return Z1.value, Z2.value, obj.value**0.5 / normC


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
    return Y.value, {"res": obj.value**0.5 / normC, "iter": 1}

