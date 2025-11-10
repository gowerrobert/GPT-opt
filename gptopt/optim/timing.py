# timing.py
import time
from typing import Optional, Sequence, List, Tuple, Union, Any

import torch
from torch import nn

# Record formats:
#   ("cpu",  label, seconds)
#   ("cuda", label, start_event, end_event, device)
TimerCPURecord = Tuple[str, str, float]
TimerCUDARecord = Tuple[str, str, torch.cuda.Event, torch.cuda.Event, torch.device]
TimerRecord = Union[TimerCPURecord, TimerCUDARecord]


class SimpleTimer:
    """
    Minimal context manager for timing a code region.

    - If a CUDA device is provided, records CUDA Events on the *current stream*
      of that device at __enter__/__exit__ so we bracket real work, including
      torch.compile() private streams.
    - Otherwise, uses time.perf_counter() on CPU.
    - If `pending` is provided, appends a record that can be drained later.
    - If `enabled` is False, it becomes a no-op and appends nothing.

    Signature matches existing call sites:
        SimpleTimer(label, device=None, pending=None, enabled=True)
    """
    def __init__(
        self,
        label: str,
        device: Optional[torch.device] = None,
        pending: Optional[List[TimerRecord]] = None,
        enabled: bool = True,
    ) -> None:
        self.label = label
        # Accept ints like 0 as cuda device indexes (light convenience).
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        self.device = device
        self.pending = pending
        self.enabled = enabled

        self._t0: Optional[float] = None
        self._ev_start: Optional[torch.cuda.Event] = None
        self._ev_end: Optional[torch.cuda.Event] = None
        self.elapsed: Optional[float] = None  # CPU-only convenience

    @torch._dynamo.disable
    def __enter__(self) -> "SimpleTimer":
        if not self.enabled:
            return self
        if self.device is not None and torch.cuda.is_available():
            stream = torch.cuda.current_stream(self.device)
            self._ev_start = torch.cuda.Event(enable_timing=True)
            self._ev_start.record(stream=stream)
        else:
            self._t0 = time.perf_counter()
        return self

    @torch._dynamo.disable
    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.enabled:
            return False

        if self.device is not None and torch.cuda.is_available():
            stream = torch.cuda.current_stream(self.device)
            self._ev_end = torch.cuda.Event(enable_timing=True)
            self._ev_end.record(stream=stream)
            if self.pending is not None and self._ev_start is not None:
                # ('cuda', label, start_event, end_event, device)
                self.pending.append(("cuda", self.label, self._ev_start, self._ev_end, self.device))
        else:
            if self._t0 is not None:
                self.elapsed = time.perf_counter() - self._t0
                if self.pending is not None:
                    # ('cpu', label, seconds)
                    self.pending.append(("cpu", self.label, float(self.elapsed)))
        # Do not suppress exceptions
        return False


def install_forward_cuda_timers(
    modules: Sequence[nn.Module],
    pending: List[Tuple],
    label: str = "xtx",
) -> None:
    """
    Attach CUDA timing hooks on selected modules.

    - Hooks are **Dynamo-disabled** (won’t be traced/compiled).
    - Only active when `module.training` and `torch.is_grad_enabled()` (train, not eval).
    - Events are recorded on the **current stream** of the correct device.
    """
    if not torch.cuda.is_available():
        return

    class _TrainOnlyCudaTimerHook:
        def __init__(self, sink: List[Tuple], tag: str) -> None:
            self.sink = sink
            self.tag = tag
            self._start: Optional[torch.cuda.Event] = None
            self._dev: Optional[torch.device] = None

        @torch._dynamo.disable
        def pre(self, m: nn.Module, args: Tuple[Any, ...]) -> None:
            # Skip during eval/validation
            if not m.training or not torch.is_grad_enabled():
                self._start = None
                self._dev = None
                return

            # Pick the device from inputs; fallback to module params
            dev: Optional[torch.device] = None
            if args and isinstance(args[0], torch.Tensor) and args[0].is_cuda:
                dev = args[0].device
            else:
                p = next(m.parameters(), None)
                if p is not None and p.is_cuda:
                    dev = p.device
            if dev is None:
                self._start = None
                return

            stream = torch.cuda.current_stream(dev)
            ev = torch.cuda.Event(enable_timing=True)
            ev.record(stream=stream)

            self._start = ev
            self._dev = dev

        @torch._dynamo.disable
        def post(self, _m: nn.Module, _args: Tuple[Any, ...], _out: Any) -> None:
            if self._start is None or self._dev is None:
                return
            stream = torch.cuda.current_stream(self._dev)
            end_ev = torch.cuda.Event(enable_timing=True)
            end_ev.record(stream=stream)

            self.sink.append(("cuda", self.tag, self._start, end_ev, self._dev))

            # clear for next call
            self._start = None
            self._dev = None

    # One hook instance per module (avoid cross-module clobbering)
    for m in modules:
        h = _TrainOnlyCudaTimerHook(pending, label)
        m.register_forward_pre_hook(h.pre, with_kwargs=False)
        m.register_forward_hook(h.post, with_kwargs=False)
