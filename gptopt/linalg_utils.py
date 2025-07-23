import torch

def rel_err(X: torch.Tensor, Y: torch.Tensor):
    """‖X − Y‖ / (‖Y‖ + ε)"""
    err = (X - Y).norm() / (Y.norm() + 1e-16)
    return err.item()

def ns_pinv(A: torch.Tensor, max_steps: int = 20, diagnostics: bool = False, use_double: bool = False):
    """
    Moore–Penrose pseudo-inverse via Newton–Schulz iteration (2-D only).

    Parameters
    ----------
    A : (m, n) tensor
        Input matrix (real or complex).
    steps : int
        Iteration count.
    diagnostics : bool
        If True, also return a list of per-step relative errors.

    Returns
    -------
    pinv : (n, m) tensor
        The pseudo-inverse of `A`.
    errs : list[float]           (only if diagnostics=True)
        Relative error after each iteration.
    """
    assert A.ndim == 2, "This simplified version accepts a single 2-D matrix"

    transposed = A.shape[0] > A.shape[1]   # make the working copy fat
    M = A.T if transposed else A           # shape: (m≤n, n)

    if use_double:  
        M = M.double()

    scale = M.norm() + 1e-16                # stabilising scale factor
    M = M / scale
    Y = M.T                                # initial guess (n, m)

    if diagnostics:
        pinv_ref = torch.linalg.pinv(M)      # reference (scaled) solution
        errs = [rel_err(Y, pinv_ref)]

    for _ in range(max_steps):
        Y_new = 2 * Y - Y @ M @ Y
        if diagnostics:
            err = rel_err(Y_new, pinv_ref)
            errs.append(err)
        Y = Y_new

    pinv = Y / scale                             # undo scaling
    if transposed:
        pinv = pinv.T                            # restore original orientation

    return (pinv, errs) if diagnostics else pinv