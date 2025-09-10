# sampling.py
from typing import Optional
import math
import torch

class SystematicRowSampler:
    """
    Streaming, microbatch-invariant systematic sampler selecting ~f of rows per step,
    implemented with integer fixed-point arithmetic (Q0.32) for exact, partition-
    invariant decisions.

    API:
      s = SystematicRowSampler(fraction=0.25)  # None or 1.0 -> keep all rows
      s.reset()                                  # call once per optimizer step
      idx = s.select_indices(N, device)          # LongTensor indices in [0, N), or None for "keep all"
    """
    __slots__ = ("num", "phase")

    _SHIFT = 32
    _SCALE = 1 << _SHIFT
    _MASK  = _SCALE - 1

    def __init__(self, fraction: Optional[float]):
        if fraction is None:
            # None = keep all rows
            self.num = None
        else:
            f = float(fraction)
            if not (0.0 < f <= 1.0):
                raise ValueError("fraction must be None or in (0, 1].")
            # Use 64-bit math for exact step; ceil to avoid undercount when f*N is integer
            step = int(math.ceil(math.ldexp(f, self._SHIFT)))  # ceil(f * 2**SHIFT)
            # step == SCALE means f == 1.0 → treated as "keep all" in select_indices
            self.num = step
        self.phase = 0  # Q0.32 integer in [0, SCALE)

    def reset(self) -> None:
        self.phase = 0

    @torch.no_grad()
    def select_indices(self, N: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Return LongTensor indices (0..N-1) to keep for this call, or None to keep all.
        Updates internal phase so the next call continues the stream.
        """
        if N <= 0:
            return torch.empty(0, dtype=torch.long, device=device)

        # Keep all rows when not subsampling (fraction None or 1.0)
        if self.num is None or self.num >= self._SCALE:
            return None

        # Vectorized Bresenham-like "carry" test over increments i = 1..N
        i1 = torch.arange(1, N + 1, device=device, dtype=torch.int64)  # 1..N
        step = torch.tensor(self.num, dtype=torch.int64, device=device)

        # Did we cross an integer boundary when adding 'step' this increment?
        t1 = self.phase + step * i1
        hi1 = t1 >> self._SHIFT
        hi0 = (t1 - step) >> self._SHIFT
        hits = (hi1 - hi0) == 1  # bool mask

        # Map from increment index (1..N) to row index (0..N-1)
        idx = torch.nonzero(hits, as_tuple=False).squeeze(-1)
        if idx.numel() > 0:
            idx = idx - 1  # select row i-1

        # Advance phase for the next call (mod SCALE)
        self.phase = int((self.phase + int(self.num) * int(N)) & self._MASK)

        return idx
