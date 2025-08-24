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

    return {
        "AXA": err_AXA,
        "XAX": err_XAX,
        "AX_proj": err_proj_AX,
        "XA_proj": err_proj_XA,
    }


@torch.no_grad()
def ns_pinv_v2(
    A: torch.Tensor,
    eps: float,
    max_steps: int = 100,
    use_double: bool = False,
    diagnostics: bool = False,
) -> torch.Tensor:
    """
    Newton–Schulz pseudoinverse with Söderström–Stewart rank-aware stopping.
    Diagnostics track Moore–Penrose residuals instead of reference pinv.
    """
    assert A.ndim == 2, "2-D input only"
    if use_double:
        A = A.to(torch.float64)

    m, n = A.shape
    I_n = torch.eye(n, dtype=A.dtype, device=A.device)

    # Safe start scale α0 = 1 / (||A||_1 ||A||_∞)
    n1 = torch.linalg.norm(A, ord=1).item()
    ninf = torch.linalg.norm(A, ord=float("inf")).item()
    alpha0 = 1.0 / max(n1 * ninf, 1e-30)

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

        # Handle one or more integer crossings
        while p_prev < next_i <= p:
            diff_norm = torch.linalg.norm(X_next - X).item()
            sigma_hat = diff_norm / ((2.0 ** k) * alpha0 + 1e-30)

            if diagnostics:
                stopping_values.append(sigma_hat)

            if sigma_hat <= eps:
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
                return X  # return X_v
            next_i += 1

        if diagnostics:
            resids.append(mp_residuals(A, X_next))
            p_values.append(p)

        X, T, p_prev = X_next, T_next, p

    # Fallback case
    if diagnostics:
        return X, {
            "residuals": resids,
            "stopping_values": stopping_values,
            "p_values": p_values,
            "iterations": max_steps,
            "stopped_at_crossing": stopped_at_crossing,
        }
    return X

@torch.no_grad()
def ns_pinv(
    A: torch.Tensor,
    max_steps: int = 20,
    diagnostics: bool = False,
    use_double: bool = False,
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

    scale = M.norm() + 1e-16
    M = M / scale
    Y = M.T                                # initial guess (n, m)

    if diagnostics:
        resids = [mp_residuals(M, Y)]

    for _ in range(max_steps):
        Y_new = 2 * Y - Y @ M @ Y
        if diagnostics:
            resids.append(mp_residuals(M, Y_new))
        Y = Y_new

    pinv = Y / scale
    if transposed:
        pinv = pinv.T

    if diagnostics:
        return pinv, {"residuals": resids}
    return pinv