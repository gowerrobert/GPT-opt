# test_power_method.py
import argparse
import numpy as np
import torch
from gptopt.linalg_utils import power_method


def assert_close(x, y, *, atol=1e-8, rtol=1e-6, name="value"):
    """
    Raise with a helpful message if x and y are not close.
    Works for scalars or tensors.
    """
    xt = torch.as_tensor(x)
    yt = torch.as_tensor(y, dtype=xt.dtype, device=xt.device)
    if not torch.allclose(xt, yt, atol=atol, rtol=rtol):
        diff = torch.linalg.norm(xt - yt).item()
        maxdiff = torch.max(torch.abs(xt - yt)).item()
        raise AssertionError(
            f"{name} mismatch: ‖Δ‖={diff:.2e}, max|Δ|={maxdiff:.2e} "
            f"(rtol={rtol}, atol={atol})"
        )


def run_tests(*, use_double: bool = False) -> None:
    # Reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    # Dtype + tolerances
    dtype = torch.float64 if use_double else torch.float32
    # Keep atol similar to the original pytest fixture (1e-4) for float32,
    # and tighten for float64.
    atol = 1e-10 if use_double else 1e-4
    # rtol used when comparing to the exact spectral norm
    rtol_spec = 1e-6 if use_double else 1e-3

    print(f"\nRunning power_method tests with use_double={use_double}")

    # --- Test: PSD identity -> largest eigenvalue is 1
    A = torch.eye(5, dtype=dtype)
    est = power_method(A, psd=True)
    assert_close(est, torch.tensor(1.0, dtype=dtype), atol=atol, name="PSD identity λ_max")

    # --- Test: rectangular matrix -> should match spectral norm ||A||_2
    A = torch.randn(6, 10, dtype=dtype)
    est = power_method(A, psd=False)
    true = torch.linalg.norm(A, ord=2)
    assert_close(est, true, atol=atol, rtol=rtol_spec, name="rectangular spectral norm")

    # --- Test: zero matrix -> norm is 0
    A = torch.zeros(4, 7, dtype=dtype)
    est = power_method(A)  # psd defaults to False; 0 either way
    assert_close(est, torch.tensor(0.0, dtype=dtype), atol=atol, name="zero matrix norm")

    print("✅ All tests passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--double", action="store_true", help="Use float64 for the tests")
    args = parser.parse_args()
    run_tests(use_double=args.double)
