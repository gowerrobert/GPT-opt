import torch
import time
from typing import Optional

class SimpleTimer:
    """
    Context manager that always reports timing into a shared sink:
      • On CUDA: appends ('cuda', label, ev_start, ev_end, device) to `pending`.
      • On CPU:  appends ('cpu',  label, elapsed_seconds) to `pending`.
    Also exposes .seconds for direct reads (remains 0.0 for CUDA).
    """
    def __init__(self, label: str, device: Optional[torch.device], pending: list, enabled: bool = True):
        self.label = label
        self.enabled = bool(enabled)
        # If disabled, force device to None so we do no work
        self.device = (device if (device is not None and getattr(device, "type", None) == "cuda") else None) if self.enabled else None
        self.pending = pending
        self.elapsed = None
        self._t0 = None
        self._ev_start = None
        self._ev_end = None

    @torch._dynamo.disable()
    def __enter__(self):
        if not self.enabled:
            return self
        if self.device is not None:
            with torch.cuda.device(self.device):
                self._ev_start = torch.cuda.Event(enable_timing=True)
                self._ev_start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    @torch._dynamo.disable()
    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        if self.device is not None:
            with torch.cuda.device(self.device):
                self._ev_end = torch.cuda.Event(enable_timing=True)
                self._ev_end.record()
            if self.pending is not None:
                # ('cuda', label, start_event, end_event, device)
                self.pending.append(("cuda", self.label, self._ev_start, self._ev_end, self.device))
        else:
            self.elapsed = time.perf_counter() - self._t0
            if self.pending is not None:
                # ('cpu', label, seconds)
                self.pending.append(("cpu", self.label, float(self.elapsed)))
        return False

    @property
    def seconds(self) -> float:
        # Returns CPU elapsed seconds; 0.0 for CUDA or when disabled
        return float(self.elapsed) if (self.enabled and self.elapsed is not None) else 0.0