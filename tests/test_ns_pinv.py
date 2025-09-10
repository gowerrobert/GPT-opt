# test_ns.py
import torch
import argparse
import numpy as np
from gptopt.linalg_utils import ns_pinv_v2 as ns_pinv  # adjust path if needed


def assert_close(X, Y, atol=1e-8, rtol=1e-6, name="matrix"):
    if not torch.allclose(X, Y, atol=atol, rtol=rtol):
        diff = torch.linalg.norm(X - Y).item()
        maxdiff = torch.max(torch.abs(X - Y)).item()
        raise AssertionError(
            f"{name} mismatch: ‖Δ‖={diff:.2e}, max|Δ|={maxdiff:.2e}"
        )


def run_tests(use_double=False):
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    dtype = torch.float64 if use_double else torch.float32
    tol = 1e-10 if use_double else 1e-5

    print(f"\nRunning tests with use_double={use_double}")

    # identity
    A = torch.eye(5, dtype=dtype)
    X = ns_pinv(A, eps=1e-12, use_double=use_double)
    assert_close(X, A, atol=tol, name="identity")

    # square random
    A = torch.randn(6, 6, dtype=dtype)
    X = ns_pinv(A, eps=1e-8, use_double=use_double)
    ref = torch.linalg.pinv(A)
    assert_close(X, ref, atol=tol, name="square")

    # rectangular tall
    A = torch.randn(8, 5, dtype=dtype) / 10
    X = ns_pinv(A, eps=1e-8, use_double=use_double)
    ref = torch.linalg.pinv(A)
    assert_close(X, ref, atol=tol, name="rectangular tall")

    # rectangular wide
    A = torch.randn(5, 8, dtype=dtype) / 10
    X = ns_pinv(A, eps=1e-8, use_double=use_double)
    ref = torch.linalg.pinv(A)
    assert_close(X, ref, atol=tol, name="rectangular wide")

    print("✅ All tests passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--double", action="store_true", help="Use float64")
    args = parser.parse_args()

    run_tests(use_double=args.double)

