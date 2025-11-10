import torch

def rel_err(X: torch.Tensor, Y: torch.Tensor):
    """‖X − Y‖ / (‖Y‖ + ε)"""
    err = (X - Y).norm() / (Y.norm() + 1e-16)
    return err.item()

def mp_residuals(A: torch.Tensor, X: torch.Tensor) -> dict[str, float]:
    """
    Compute Moore–Penrose residuals that measure how well X approximates A^+.

    Returns relative Frobenius norms for:
      - AXA ≈ A
      - XAX ≈ X
      - AX idempotency
      - XA idempotency
    """
    err_AXA = rel_err(A @ X @ A, A)
    err_XAX = rel_err(X @ A @ X, X)

    AX = A @ X
    XA = X @ A
    err_proj_AX = rel_err(AX, AX @ AX)
    err_proj_XA = rel_err(XA, XA @ XA)

    metrics = {
        "AXA": err_AXA,
        "XAX": err_XAX,
        "AX_proj": err_proj_AX,
        "XA_proj": err_proj_XA,
        # if A is low rank, this might not converge:
        "AX": rel_err(AX, torch.eye(AX.shape[0], device=AX.device, dtype=AX.dtype)),
    }
    if X.shape[0] == X.shape[1]:
        metrics = metrics | {"sym": rel_err(X, X.T),}
    return metrics


@torch.no_grad()
def accelerated_ns_pinv(
    A: torch.Tensor,
    l,  # should this be 0-dimensional torch.Tensors?
    u,
    max_steps: int,
    psd: bool = True,
    add_eps: float = 0,
    early_stop_eps: float = 0,
    dtype: torch.dtype = None,
    diagnostics: bool = False,
    return_iters: bool = False,
) -> torch.Tensor:

    assert A.ndim == 2, "2-D input only"
    m, n = A.shape
    assert 0 < l < u, "Require 0 < l < u"
    transposed = A.shape[0] > A.shape[1]  # make the working copy fat
    if transposed: A = A.T # shape: (m≤n, n)
    u = float(u); l = float(l); add_eps = float(add_eps); early_stop_eps = float(early_stop_eps)
    scale = u + 1e-10
    A = A / scale
    u = 1; l /= scale; add_eps /= scale; early_stop_eps /= scale
    if dtype: A = A.to(dtype)

    I_m = torch.eye(m, dtype=A.dtype, device=A.device)
    assert add_eps == 0 or m == n, "add_eps only makes sense for symmetric"
    if add_eps > 0: A += add_eps * I_m

    if psd == "cubic":
        lu = l / u
        initialization_denom = 4 + lu*(1+lu)*(12+lu*(1+lu)*(39+4*lu*(1+lu)))
        b = 54 * (1+lu)**2 * (1+lu+lu**2) / initialization_denom
        c = - 54 * (1+lu)**3 / initialization_denom
        Y = (b*I_m + c * A) @ A.T
        print("In low precision, Y may still have negative eigenvalues due to numerical error :( That's could cause blowup")
        Y *= .99  # the previous line may have sent some of the middle eigenvalues to 1+epsilon
        r = (-2 - 3*lu + 3*lu**2 + 2*lu**3)**2 / initialization_denom 
        assert early_stop_eps < -2*b/(3*c), "the initialization is non-monotonic, so early stopping eps must be small er than the non-monotonicity"
        early_stop_potential = (b + c * early_stop_eps) * early_stop_eps**2   
    elif psd == True:  # writing it this way to avoid confusion when psd is a string
        # Y = (2/(u + l)) * I_m
        # r = (u - l) / (u + l)
        inv_kappa = l / u
        denom = 1 + inv_kappa * (6 + inv_kappa)
        Y = (8 / (u * denom)) * ((1+inv_kappa)*I_m - A.T / u)
        r = (1 - inv_kappa)**2 / denom
        early_stop_potential = (8 / (u * denom)) * ((1+inv_kappa)*early_stop_eps - (early_stop_eps**2) / u)
    elif psd == False:
        Y = (2/(u**2 + l**2)) * A.T
        r = (u**2 - l**2) / (u**2 + l**2)
        assert r - 1 < 0
        early_stop_potential = (2/(u**2 + l**2)) * early_stop_eps**2
    else:
        raise ValueError("psd must be True, False, or 'cubic'")

    if diagnostics:
        if add_eps > 0:
            resids = [mp_residuals(A - add_eps*I_m, Y)]
        else:
            resids = [mp_residuals(A, Y)]

    for step in range(max_steps):
        denom = 2 - r**2
        Y = Y @ (2*I_m - A @ Y) * (2 / denom)
        # Y = (2*Y - Y @ A @ Y) * (2 / denom)
        r = r**2 / denom
        early_stop_potential = early_stop_potential * (2 - early_stop_potential) * (2 / denom)

        if diagnostics:
            if add_eps > 0:
                resids.append(mp_residuals(A - add_eps*I_m, Y))  # Y @ (I_m + eps * Y @ (I_m + eps * Y))))  # Y + eps * Y^2 - eps^2 * Y^3
            else:
                resids.append(mp_residuals(A, Y))

        if abs(1 - early_stop_potential) < 1e-6:
            break

    if transposed: Y = Y.T
    Y /= scale

    if diagnostics:
        return Y, {"residuals": resids}
    elif return_iters:
        return Y, {"iterations": step}
    else:
        return Y


@torch.no_grad()
def ns_pinv_v2(
    A: torch.Tensor,
    eps: float,
    max_steps: int = 100,
    use_double: bool = False,
    use_bf16: bool = False,
    return_iters: bool = False,
    diagnostics: bool = False,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Newton–Schulz pseudoinverse with Söderström–Stewart rank-aware stopping.
    Diagnostics track Moore–Penrose residuals instead of reference pinv.
    """
    assert A.ndim == 2, "2-D input only"
    if use_double:
        A = A.to(torch.float64)
    elif use_bf16:
        A = A.to(torch.bfloat16)

    m, n = A.shape
    I_n = torch.eye(n, dtype=A.dtype, device=A.device)

    # Safe start scale α0 = 1 / (||A||_1 ||A||_∞)
    n1 = torch.linalg.norm(A, ord=1).item()  # max(sum(abs(x), dim=0))
    ninf = torch.linalg.norm(A, ord=float("inf")).item()  # max(sum(abs(x), dim=1))
    alpha0 = 1.0 / max(n1 * ninf, 1e-30)

    if verbose:
        print(f"ns_pinv_v2: m={m}, n={n}, alpha0={alpha0:.2e}")

    # Init
    X = alpha0 * A.T
    T = X @ A
    p_prev = torch.trace(T).item()
    next_i = 1  # first integer crossing to monitor

    # Initialize diagnostics if requested
    if diagnostics:
        resids = [mp_residuals(A, X)]
        stopping_values = []
        p_values = [p_prev]
        stopped_at_crossing = False

    # --------- Unscaled Newton–Schulz ----------
    for k in range(max_steps):
        X_next = (2.0 * I_n - T) @ X
        T_next = X_next @ A
        p = torch.trace(T_next).item()

        if verbose:
            print(f"  iter {k+1}: p={p:.6f}, p_prev={p_prev:.6f}")

        # Handle one or more integer crossings
        while p_prev < next_i <= p:
            diff_norm = torch.linalg.norm(X_next - X, ord='fro').item()
            sigma_hat = diff_norm / ((2.0 ** k) * alpha0 + 1e-30)

            if verbose:
                print(f"    crossing {next_i}: sigma_hat={sigma_hat:.2e}")

            if diagnostics:
                stopping_values.append(sigma_hat)

            if sigma_hat <= eps:
                if verbose:
                    print(f"  converged at crossing {next_i} after {k+1} iterations")
                if diagnostics:
                    stopped_at_crossing = True
                    resids.append(mp_residuals(A, X_next))
                    p_values.append(p)
                    return X, {
                        "residuals": resids,
                        "stopping_values": stopping_values,
                        "p_values": p_values,
                        "iterations": k + 1,
                        "stopped_at_crossing": stopped_at_crossing,
                    }
                elif return_iters:
                    return X, {"iterations": k + 1}
                else:
                    return X  # return X_v
            next_i += 1

        if diagnostics:
            resids.append(mp_residuals(A, X_next))
            p_values.append(p)

        X, T, p_prev = X_next, T_next, p

    # Fallback case
    if verbose:
        print(f"  reached max_steps={max_steps} without convergence")
    if diagnostics:
        return X, {
            "residuals": resids,
            "stopping_values": stopping_values,
            "p_values": p_values,
            "iterations": max_steps,
            "stopped_at_crossing": stopped_at_crossing,
        }
    if return_iters:
        return X, {"iterations": max_steps}
    return X

@torch.no_grad()
def ns_pinv(
    A: torch.Tensor,
    max_steps: int = 20,
    diagnostics: bool = False,
    use_double: bool = False,
    use_bf16: bool = False,
    verbose: bool = False,
):
    """
    Moore–Penrose pseudo-inverse via Newton–Schulz iteration (2-D only).

    Parameters
    ----------
    A : (m, n) tensor
        Input matrix.
    max_steps : int
        Iteration count.
    diagnostics : bool
        If True, also return Moore–Penrose residuals per iteration.
    use_double : bool
        Compute in float64 if True.
    use_bf16 : bool
        Compute in bfloat16 if True.
    verbose : bool
        If True, print iteration information.

    Returns
    -------
    pinv : (n, m) tensor
        The pseudo-inverse of `A`.
    diag : dict (only if diagnostics=True)
        - 'residuals': list of MP residuals per iteration
    """
    assert A.ndim == 2, "This version accepts a single 2-D matrix"

    transposed = A.shape[0] > A.shape[1]   # make the working copy fat
    M = A.T if transposed else A           # shape: (m≤n, n)

    if use_double:
        M = M.double()
    elif use_bf16:
        M = M.bfloat16()

    scale = M.norm() + 1e-16
    M = M / scale
    Y = M.T                                # initial guess (n, m)

    if verbose:
        print(f"ns_pinv: shape={A.shape}, transposed={transposed}, scale={scale:.2e}")

    if diagnostics:
        resids = [mp_residuals(M, Y)]

    for step in range(max_steps):
        Y_new = 2 * Y - Y @ M @ Y
        if verbose:
            print(f"  iter {step+1}/{max_steps}")
        if diagnostics:
            resids.append(mp_residuals(M, Y_new))
        Y = Y_new

    pinv = Y / scale
    if transposed:
        pinv = pinv.T

    if verbose:
        print(f"  completed {max_steps} iterations")

    if diagnostics:
        return pinv, {"residuals": resids}
    return pinv


@torch.no_grad()
def power_method(
    A: torch.Tensor,
    max_iters: int = 100,
    tol: float = 1e-6,
    psd: bool = False,
    use_bf16: bool = False,
    verbose: bool = False,
    return_iters: bool = False,
) -> torch.Tensor:
    """
    Estimate the spectral norm (largest singular value) of A by power iteration.

    If psd=True, A is assumed symmetric positive semidefinite and the method
    performs power iteration directly on A to estimate its largest eigenvalue
    (which equals the spectral norm). Otherwise, it performs power iteration on
    A^T A and returns sqrt of the dominant eigenvalue.

    Parameters
    ----------
    A : torch.Tensor
        2-D matrix.
    max_iters : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on consecutive iterate difference.
    psd : bool
        If True, assume A is symmetric PSD and iterate on A.
    use_bf16 : bool
        Compute in bfloat16 if True.
    verbose : bool
        If True, print iteration information.

    Returns
    -------
    torch.Tensor
        Estimated spectral norm as a scalar tensor on A's device/dtype.
    """
    assert A.ndim == 2, "power_method expects a 2-D matrix"
    m, n = A.shape
    if psd:
        assert m == n, "psd=True requires a square matrix"

    device = A.device
    dtype = A.dtype

    # Convert to bfloat16 if requested
    if use_bf16:
        A = A.to(torch.bfloat16)
        dtype = torch.bfloat16

    if verbose:
        print(f"power_method: shape={A.shape}, psd={psd}, tol={tol}")

    # Initialize with a random nonzero vector
    v = torch.randn(n, device=device, dtype=dtype)
    v = v / (v.norm() + 1e-16)

    for iter_num in range(max_iters):
        v_prev = v
        if psd:
            w = A @ v
        else:
            # TODO: depending on the number of iters, we might want to assemble A.T @ A once
            w = A.T @ (A @ v)
        w_norm = w.norm()
        if w_norm == 0:
            if verbose:
                print(f"  iter {iter_num+1}: zero norm, stopping")
            zero_val = torch.zeros((), device=device, dtype=dtype)
            if return_iters:
                return zero_val, (iter_num + 1)
            return zero_val
        v = w / w_norm
        
        diff_norm = (v - v_prev).norm()
        if verbose:
            print(f"  iter {iter_num+1}: diff_norm={diff_norm:.2e}")
        
        if diff_norm < tol:
            if verbose:
                print(f"  converged after {iter_num+1} iterations")
            break

    if psd:
        # Rayleigh quotient equals largest eigenvalue for unit vector v
        eig_est = torch.dot(v, A @ v)
        # Numerical safety: ensure non-negative for PSD
        result = torch.clamp(eig_est, min=0)
        if verbose:
            print(f"  final eigenvalue estimate: {result:.6f}")
        if return_iters:
            return result, (iter_num + 1)
        return result
    else:
        # Dominant eigenvalue of A^T A equals squared spectral norm
        Av = A @ v
        mu = torch.dot(Av, Av)
        result = torch.sqrt(torch.clamp(mu, min=0))
        if verbose:
            print(f"  final spectral norm estimate: {result:.6f}")
        if return_iters:
            return result, (iter_num + 1)
        return result