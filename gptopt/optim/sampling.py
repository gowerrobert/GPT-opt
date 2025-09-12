from typing import Optional
import torch
import torch.nn as nn

class SystematicRowSampler(nn.Module):
    """
    Streaming, microbatch-invariant systematic sampler selecting ~f of rows per step,
    implemented with integer fixed-point arithmetic (Q0.32).
    Always returns a LongTensor of indices (possibly empty). No None returns.
    State is in tensor buffers so it's safe to call from compiled regions.
    """
    __slots__ = ()
    _SHIFT = 32
    _SCALE = 1 << _SHIFT
    _MASK  = _SCALE - 1

    def __init__(self, fraction: Optional[float]) -> None:
        super().__init__()
        # Interpret fraction: None or >=1.0 -> keep all rows (static boolean, compile-safe)
        if fraction is None or float(fraction) == 1.0:
            f = None
        elif float(fraction) > 1.0:
            raise ValueError("fraction must be None, 1.0, or in (0, 1).")
        else:
            f = float(fraction)
            if not (0.0 < f < 1.0):
                raise ValueError("fraction must be None or in (0, 1).")

        # Static flag: no Tensor.item() in compiled regions
        self._keep_all: bool = (f is None)

        # Buffers
        self.register_buffer("phase", torch.zeros((), dtype=torch.int64), persistent=False)
        num = 0 if f is None else int(round(f * self._SCALE))  # Q0.32 numerator
        self.register_buffer("num", torch.tensor(num, dtype=torch.int64), persistent=False)
        self.register_buffer("mask", torch.tensor(self._MASK, dtype=torch.int64), persistent=False)

    @torch.no_grad()
    def reset(self) -> None:
        # Reset per-step accumulator (call once per optimizer step)
        self.phase.zero_()

    def select_indices(self, N: int, device: torch.device) -> torch.Tensor:
        """
        Always returns a LongTensor of indices in [0, N) (possibly empty).
        No Python branching on runtime tensors; safe to call in compiled forward.
        """
        # Keep-all fast path – uses a Python bool constant set at __init__ (compile-safe)
        if self._keep_all:
            return torch.arange(N, device=device, dtype=torch.long)

        # 1..N increments
        i1   = torch.arange(1, N + 1, device=device, dtype=torch.int64)
        step = self.num  # int64 tensor (Q0.32 numerator)

        # Fixed-point systematic sampling
        t1   = self.phase + step * i1
        hi1  = t1 >> self._SHIFT
        hi0  = (t1 - step) >> self._SHIFT

        # Hit detection and mapping to 0..N-1 (branch-free; works for empty too)
        idx  = torch.nonzero((hi1 - hi0) == 1, as_tuple=False).squeeze(-1) - 1

        # Update phase: phase = (phase + step * N) & MASK
        self.phase.copy_((self.phase + step * N) & self.mask)
        return idx
