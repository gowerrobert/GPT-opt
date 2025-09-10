# test_sampling.py
import math
import random
import pytest
import torch

# Import your sampler(s)
from gptopt.optim.sampling import SystematicRowSampler



def _global_indices_from_partitions(N, fraction, partitions):
    """
    Simulate a single 'step' with N rows split into `partitions` microbatches.
    Returns the sorted tensor of *global* selected indices in [0, N).
    """
    sampler = SystematicRowSampler(fraction=fraction)
    sampler.reset()

    offset = 0
    picks = []
    device = torch.device("cpu")
    for n in partitions:
        assert n > 0
        idx = sampler.select_indices(n, device)
        if idx is None:
            local = torch.arange(n, device=device, dtype=torch.long)
        else:
            local = idx.to(torch.long)
        if local.numel() > 0:
            picks.append(local + offset)
        offset += n

    if len(picks) == 0:
        return torch.empty(0, dtype=torch.long)
    out = torch.cat(picks, dim=0)
    # selection order is already increasing within each microbatch; concatenation preserves order
    return out


def _one_batch_indices(N, fraction):
    """Convenience: same as _global_indices_from_partitions but a single chunk."""
    return _global_indices_from_partitions(N, fraction, [N])


def _random_partitions(N, k=None, seed=0):
    """
    Create k random positive integers summing to N.
    If k is None, choose a small random k (2..min(8,N)).
    """
    rng = random.Random(seed)
    if k is None:
        k = min(8, N) if N > 1 else 1
        k = rng.randint(2, max(2, k))

    # start with k ones, then distribute remaining (N-k)
    parts = [1] * k
    remaining = max(0, N - k)
    for _ in range(remaining):
        parts[rng.randrange(k)] += 1
    rng.shuffle(parts)
    assert sum(parts) == N and all(p > 0 for p in parts)
    return parts


@pytest.mark.parametrize("N", [1, 7, 37, 1000, 10000, 50000])
@pytest.mark.parametrize("f", [0.05, 0.1, 0.25, 0.5, 0.95])
def test_density_within_one(N, f):
    """Selected count over a whole step is ~f*N with discrepancy < 1."""
    idx = _one_batch_indices(N, f)
    k = int(idx.numel())
    # Theoretical target is floor(f*N) when phase=0; allow ±1 in case of float roundoff
    target_floor = math.floor(f * N)
    assert abs(k - f * N) < 1.0 + 1e-6
    assert k in {target_floor, target_floor + 1}


@pytest.mark.parametrize("N", [17, 123, 999, 10000])
@pytest.mark.parametrize("f", [0.07, 0.2, 0.33, 0.5, 0.8])
def test_microbatch_invariance_exact_indices(N, f):
    """
    For the same global row order, the *exact set of global indices* chosen
    must be identical regardless of how we partition into microbatches.
    """
    idx_one = _one_batch_indices(N, f)

    parts_a = _random_partitions(N, k=3, seed=42)
    parts_b = _random_partitions(N, k=7, seed=123)

    idx_a = _global_indices_from_partitions(N, f, parts_a)
    idx_b = _global_indices_from_partitions(N, f, parts_b)

    # exact equality of ordered indices
    assert torch.equal(idx_one, idx_a)
    assert torch.equal(idx_one, idx_b)


@pytest.mark.parametrize("N", [1, 10, 1000, 12345])
@pytest.mark.parametrize("f", [None, 1.0])
def test_full_fraction_none_and_one(N, f):
    """
    fraction=None and fraction=1.0 should behave as 'keep all rows'.
    """
    idx = _one_batch_indices(N, f)
    assert idx.numel() == N
    if N > 0:
        # must be [0,1,2,...,N-1]
        assert torch.equal(idx, torch.arange(N, dtype=torch.long))


@pytest.mark.parametrize("N", [10, 50, 200])
@pytest.mark.parametrize("f", [0.01, 0.02, 0.05])
def test_tiny_fraction_edge_cases(N, f):
    """
    For very small fractions and small N, some calls may select 0 rows,
    but total per-step count must be floor(f*N) or ceil(f*N).
    """
    idx = _one_batch_indices(N, f)
    k = int(idx.numel())
    assert k in {math.floor(f * N), math.ceil(f * N)}
    assert abs(k - f * N) < 1.0 + 1e-6


def test_reset_reproducibility():
    """
    After reset(), the sampler should reproduce the same indices
    for the same N and partitions.
    """
    N, f = 1000, 0.37
    parts = _random_partitions(N, k=5, seed=7)

    idx1 = _global_indices_from_partitions(N, f, parts)
    idx2 = _global_indices_from_partitions(N, f, parts)  # new sampler/reset internally

    assert torch.equal(idx1, idx2)
