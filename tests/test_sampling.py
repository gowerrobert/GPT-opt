
# test_sampling.py (updated for tensor-only sampler API)
import torch
import random
import pytest

from gptopt.optim.sampling import SystematicRowSampler

def _random_partitions(N, k=5, seed=0):
    rng = random.Random(seed)
    if N == 0:
        return []
    cuts = sorted(rng.sample(range(1, N), max(0, min(k-1, max(0, N-1)))))
    parts, prev = [], 0 
    for c in cuts + [N]:
        parts.append(c - prev)
        prev = c
    assert sum(parts) == N
    return parts

def _global_indices_from_partitions(N, fraction, partitions):
    device = torch.device("cpu")
    s = SystematicRowSampler(fraction)
    s.reset()
    out = []
    offset = 0
    for m in partitions:
        idx_local = s.select_indices(m, device)
        if idx_local.numel() > 0:
            out.append(idx_local + offset)
        offset += m
    if len(out) == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.sort(torch.cat(out))[0]

def _one_shot_indices(N, fraction):
    device = torch.device("cpu")
    s = SystematicRowSampler(fraction)
    s.reset()
    return s.select_indices(N, device).sort()[0]

@pytest.mark.parametrize("N,f", [(1000, 0.25), (1000, 0.37), (257, 0.5), (1, 0.9), (2, 0.1)])
def test_partition_invariance(N, f):
    parts = _random_partitions(N, k=min(5, max(1, N)), seed=123)
    idx_parts = _global_indices_from_partitions(N, f, parts)
    idx_one   = _one_shot_indices(N, f)
    assert torch.equal(idx_parts, idx_one)

@pytest.mark.parametrize("N,f", [(1000, 0.25), (777, 0.37), (100, 0.5), (8, 0.9)])
def test_expected_fraction(N, f):
    idx = _one_shot_indices(N, f)
    k = idx.numel()
    assert abs(k - f*N) <= 1.0 + 1e-6

def test_reset_reproducibility():
    N, f = 1000, 0.37
    parts = _random_partitions(N, k=5, seed=7)
    idx1 = _global_indices_from_partitions(N, f, parts)
    idx2 = _global_indices_from_partitions(N, f, parts)
    assert torch.equal(idx1, idx2)

@pytest.mark.parametrize("N", [0, 1, 2, 17, 256])
def test_keep_all_none(N):
    idx = _one_shot_indices(N, None)
    assert torch.equal(idx, torch.arange(N, dtype=torch.long))

@pytest.mark.parametrize("N", [0, 1, 2, 17, 256])
def test_keep_all_one(N):
    idx = _one_shot_indices(N, 1.0)
    assert torch.equal(idx, torch.arange(N, dtype=torch.long))

def test_invalid_fraction_raises():
    with pytest.raises(ValueError):
        SystematicRowSampler(0.0)
    with pytest.raises(ValueError):
        SystematicRowSampler(-0.1)
    with pytest.raises(ValueError):
        SystematicRowSampler(1.1)
