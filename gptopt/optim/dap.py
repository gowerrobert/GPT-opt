import torch
from torch import nn
from torch.optim import Optimizer
from gptopt.linalg_utils import ns_pinv, ns_pinv_v2, accelerated_ns_pinv, power_method
from gptopt.optim.timing import SimpleTimer, install_forward_cuda_timers
from gptopt.optim.sampling import SystematicRowSampler

from typing import Optional
import torch.nn.functional as F

import os
import sys
import time


class LinearWithXtX(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, *, track_xtx=False):
        super().__init__(in_features, out_features, bias=bias)
        # We default to track_xtx=False; the optimizer will enable it only for DAP layers.
        self.track_xtx = bool(track_xtx)

        # fp32 accumulators (same device as module)
        self.register_buffer(
            "C_accum_sum", torch.zeros(in_features, in_features, dtype=torch.float32)
        )
        self.register_buffer("C_accum_count", torch.zeros((), dtype=torch.int64))

        # wired by the optimizer
        self._dap_timing_enabled: bool = False
        self._dap_timing_sink: Optional[list] = None
        self._dap_accum_enabled: bool = (
            False  # optimizer toggles this ON for DAP layers only
        )
        self._dap_xtx_override: bool = False
        self._xtx_sampler: Optional[SystematicRowSampler] = None

    @torch.no_grad()
    def _accum_xtx(self, x: torch.Tensor) -> None:
        # Completely detach from autograd; don’t let this be traced/compiled.
        X = x.detach().reshape(-1, x.shape[-1])  # [N, d]
        if self._xtx_sampler is not None:
            idx = self._xtx_sampler.select_indices(X.shape[0], X.device)
            if idx is not None:
                X = X.index_select(0, idx)
        C = X.transpose(0, 1) @ X  # [d, d] in activation dtype
        self.C_accum_sum.add_(C.to(self.C_accum_sum.dtype))
        self.C_accum_count.add_(X.shape[0])  # avoid tiny alloc of a scalar tensor

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.training and self._dap_accum_enabled:  # only for DAP layers
            self._accum_xtx(x)
        return y


def swap_linears_for_xtx(mod: torch.nn.Module):
    for name, child in list(mod.named_children()):
        if isinstance(child, LinearWithXtX):
            swap_linears_for_xtx(child)
            continue

        if isinstance(child, nn.Linear):
            repl = LinearWithXtX(
                child.in_features,
                child.out_features,
                bias=(child.bias is not None),
                track_xtx=False,  # <- OFF by default; optimizer will enable per DAP layer
            ).to(device=child.weight.device)
            repl.train(child.training)

            with torch.no_grad():
                repl.weight.data = child.weight.detach().clone()
                if child.bias is not None:
                    repl.bias.data = child.bias.detach().clone()

            setattr(mod, name, repl)
            swap_linears_for_xtx(repl)
        else:
            swap_linears_for_xtx(child)


class DAP(Optimizer):
    def __init__(
        self,
        model,
        named_params,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=True,
        damping=0.0,
        ema_beta=0.0,
        dap_beta1: float = 0.0,
        dap_beta2: float = 0.0,
        dap_eps: float = 1e-8,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        disable_preconditioning: bool = False,
        scalar: bool = False,
        include_output: bool = False,
        include_embed: bool = False,
        use_ns_pinv: bool = False,
        ns_pinv_steps: int = 30,
        rcond: float = 1e-3,
        debug_timing: bool = True,
        use_bf16: bool = False,
        use_fp64: bool = False,
        moments_on_precond: bool = False,
        adagradnorm: bool = False,
        refresh_precond_iters=None,
        accelerated: bool = False,
        spectral_norm_estimator: str = "power_method",  # "power_method", "frobenius"
        xtx_subsample: Optional[float] = None,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            damping=damping,
            ema_beta=ema_beta,
            dap_beta1=dap_beta1,
            dap_beta2=dap_beta2,
            dap_eps=dap_eps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            moments_on_precond=moments_on_precond,
            adagradnorm=adagradnorm,
        )

        if not disable_preconditioning:
            needs_swap = any(
                isinstance(m, nn.Linear) and not isinstance(m, LinearWithXtX)
                for m in model.modules()
            )
            if needs_swap:
                swap_linears_for_xtx(model)

        dap_params, dap_params_names = [], []
        adamw_params, adamw_params_names = [], []

        excluded_names = []
        if not include_output:
            excluded_names.extend(["lm_head", "weight_proj"])

        if not include_embed:
            excluded_names.extend(["embeddings", "embed_tokens", "wte", "wpe"])

        self.param_to_name = {}

        for name, p in named_params:
            self.param_to_name[p] = name

            if p.ndim >= 2 and not any(excluded in name for excluded in excluded_names):
                dap_params.append(p)
                dap_params_names.append(name)
            else:
                adamw_params.append(p)
                adamw_params_names.append(name)

        # Print informative summary of parameter classification
        print(f"\n=== DAP Optimizer Parameter Classification ===")
        print(f"DAP Parameters ({len(dap_params)}):")
        for name in dap_params_names:
            print(f"  - {name}")
        print(f"\nAdamW Parameters ({len(adamw_params)}):")
        for name in adamw_params_names:
            print(f"  - {name}")
        print(f"==============================================\n")

        # One-time params debug print at initialization
        self._num_dap_params_total = len(dap_params)
        if debug_timing:
            print(f"[DAP] params={self._num_dap_params_total}")

        params = list(dap_params)
        params.extend(adamw_params)
        super().__init__(params, defaults)

        # Sort parameters into those for which we will use DAP, and those for which we will not
        # Use DAP for every parameter in dap_params which is >= 2D and doesn't look like an embedding or head layer
        for p in dap_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_dap"] = True

        for p in adamw_params:
            # Do not use DAP for parameters in adamw_params
            self.state[p]["use_dap"] = False

        self.ema_beta = ema_beta
        # Whether to bypass the expensive preconditioner and use a plain SGD update.
        self.disable_preconditioning = disable_preconditioning
        self.scalar = scalar
        self.include_output = include_output
        self.include_embed = include_embed
        self.use_ns_pinv = use_ns_pinv
        self.ns_pinv_steps = ns_pinv_steps
        self.rcond = rcond
        # Debug timing controls
        self.debug_timing = debug_timing
        self._debug_step_idx = 0
        self.xtx_subsample = xtx_subsample

        # Accumulator for average relative Frobenius change of C_ema per step
        self._C_ema_rel_change_accum = 0.0

        self._C_ema_update_time_accum = (
            0.0  # total wall-clock seconds spent updating C_ema since last step()
        )
        self._C_ema_update_count = 0  # how many EMA updates occurred since last step()
        self._pending_timing_events = (
            []
        )  # list of (label, start_event, end_event, device)

        # Accumulators for XtX timing across micro-batches between step() calls
        self._XtX_update_time_accum = 0.0
        self._XtX_update_count = 0

        # Precision control
        self.rcond = float(rcond)
        self.bf16 = bool(use_bf16)
        self.fp64 = bool(use_fp64)

        self.accelerated = accelerated
        if spectral_norm_estimator not in ["power_method", "frobenius"]:
            raise ValueError(
                f"Invalid spectral_norm_estimator: {spectral_norm_estimator}"
            )
        self.spectral_norm_estimator = spectral_norm_estimator

        if self.bf16 and self.fp64:
            raise ValueError("use_bf16 and use_fp64 are mutually exclusive")

        # dtype used for power_method / pinv / ns_pinv operations
        if self.bf16:
            self.op_dtype = torch.bfloat16
        elif self.fp64:
            self.op_dtype = torch.float64
        else:
            self.op_dtype = torch.float32

        assert not (
            self.scalar and self.disable_preconditioning
        ), "choose either scalar or disable_preconditioning"

        # Control how often to refresh preconditioned vector; None disables caching
        if refresh_precond_iters is not None:
            refresh_precond_iters = int(refresh_precond_iters)
            if refresh_precond_iters <= 0:
                refresh_precond_iters = None
        self.refresh_precond_iters = refresh_precond_iters

        if not self.disable_preconditioning:
            self.param_to_module = self._wire_up_xtx_sources(model, dap_params)

    def _wire_up_xtx_sources(self, model: nn.Module, dap_params):
        dap_modules = []
        dap_param_ids = {id(p) for p in dap_params}
        param_to_module = {}

        for mod in model.modules():
            if isinstance(mod, nn.Linear) and id(mod.weight) in dap_param_ids:
                if not isinstance(mod, LinearWithXtX):
                    raise TypeError(
                        "Buffer mode requires LinearWithXtX. Call swap_linears_for_xtx(model) first."
                    )
                # timing + accumulation only for DAP layers
                mod._dap_timing_enabled = bool(self.debug_timing)
                mod._dap_timing_sink = self._pending_timing_events
                mod._dap_accum_enabled = True  # <- enable XtX accumulation
                mod._xtx_sampler = SystematicRowSampler(self.xtx_subsample)
                param_to_module[mod.weight] = mod
                dap_modules.append(mod)

        if len(param_to_module) != len(dap_params):
            missing = [p for p in dap_params if p not in param_to_module]
            raise ValueError(
                f"Some DAP params not owned by any LinearWithXtX: {missing}"
            )

        if self.debug_timing and torch.cuda.is_available() and dap_modules:
            install_forward_cuda_timers(
                modules=dap_modules,
                pending=self._pending_timing_events,
                label="xtx",     # matches your existing aggregator keys
            )

        return param_to_module

    def _estimate_spectral_norm(self, C: torch.Tensor, psd=True, return_iters=True):
        if self.spectral_norm_estimator == "power_method":
            return power_method(
                C, max_iters=self.ns_pinv_steps, psd=psd, return_iters=return_iters
            )
        elif self.spectral_norm_estimator == "frobenius":
            if return_iters:
                return torch.linalg.norm(C, ord="fro"), 0
            else:
                return torch.linalg.norm(C, ord="fro")
        else:
            raise ValueError(
                f"Invalid spectral_norm_estimator: {self.spectral_norm_estimator}"
            )

    def _precondition(
        self, tensor: torch.Tensor, C: torch.Tensor, p: torch.Tensor, step: int
    ):
        """Precondition a gradient-like tensor using covariance C.
        Returns (preconditioned_tensor, pm_iters, ns_iters).
        """
        pm_iters = 0
        ns_iters = 0

        C_op = C.to(self.op_dtype)
        v_op = tensor.to(self.op_dtype)

        if self.use_ns_pinv:
            device = p.device if p.is_cuda else None
            with SimpleTimer(
                "power", device, self._pending_timing_events, enabled=self.debug_timing
            ):
                sig_max, pm_iters = self._estimate_spectral_norm(
                    C_op, psd=True, return_iters=True
                )
            sig_max = sig_max.item()
            eps = sig_max * self.rcond

            with SimpleTimer(
                "pinv", device, self._pending_timing_events, enabled=self.debug_timing
            ):
                if self.accelerated == "add_eps_1e-3":
                    Cinv, info = accelerated_ns_pinv(
                        C_op,
                        l=eps,
                        u=1.2 * sig_max,
                        max_steps=self.ns_pinv_steps,
                        psd=True,
                        add_eps=1e-3,  # note this is absolute, not relative to sig_max, unlike the others
                        early_stop_eps=eps,
                        dtype=torch.float32,
                        diagnostics=False,
                        return_iters=True,
                    )
                elif self.accelerated:
                    Cinv, info = accelerated_ns_pinv(
                        C_op,
                        l=eps,
                        u=1.2 * sig_max,
                        max_steps=self.ns_pinv_steps,
                        psd=False,  # for now
                        add_eps=0,
                        early_stop_eps=eps,
                        dtype=torch.float32,
                        diagnostics=False,
                        return_iters=True,
                    )
                else:
                    Cinv, info = ns_pinv_v2(
                        C_op,
                        eps=eps,
                        max_steps=self.ns_pinv_steps,
                        return_iters=True,
                        diagnostics=False,
                    )

            ns_iters = int(info.get("iterations", 0))
            if torch.isnan(Cinv).any() or torch.isinf(Cinv).any():
                debug_dir = "debug_matrices"
                os.makedirs(debug_dir, exist_ok=True)
                matrix_path = os.path.join(
                    debug_dir, f"C_{self.param_to_name[p]}_step_{step}.pt"
                )
                grad_path = os.path.join(
                    debug_dir, f"grad_{self.param_to_name[p]}_step_{step}.pt"
                )
                torch.save(C_op, matrix_path)
                torch.save(v_op, grad_path)
                print(
                    f"ERROR: NaN/inf detected in pseudo-inverse calculation for parameter {self.param_to_name[p]} at step {step}"
                )
                print(
                    f"Matrix saved to debug_matrices/C_{self.param_to_name[p]}_step_{step}.pt"
                )
                sys.exit(1)

            u_local = v_op @ Cinv
        else:
            if self.scalar:
                device = p.device if p.is_cuda else None
                with SimpleTimer(
                    "power",
                    device,
                    self._pending_timing_events,
                    enabled=self.debug_timing,
                ):
                    sig_max, pm_iters = self._estimate_spectral_norm(
                        C_op, psd=True, return_iters=True
                    )
                u_local = v_op / sig_max
            else:
                device = p.device if p.is_cuda else None
                with SimpleTimer(
                    "pinv",
                    device,
                    self._pending_timing_events,
                    enabled=self.debug_timing,
                ):
                    pinv_mat = torch.linalg.pinv(C_op, rtol=self.rcond)
                u_local = v_op @ pinv_mat

        return u_local.to(p.dtype), pm_iters, ns_iters

    def _precondition_cached(
        self,
        tensor: torch.Tensor,
        C: torch.Tensor,
        p: torch.Tensor,
        step: int,
        cache_key: str,
    ):
        """Wrapper over _precondition that optionally caches the result for a number of steps.
        If self.refresh_precond_iters is None, always recomputes.
        When caching, stores results under state[p][f"precond_cache_{cache_key}"] and associated step.
        Returns (u, pm_iters, ns_iters), where timing/iters are zero if reused from cache.
        """
        refresh = self.refresh_precond_iters
        if refresh is None:
            return self._precondition(tensor, C, p, step)

        state = self.state[p]
        cache_val_key = f"precond_cache_{cache_key}"
        cache_step_key = f"precond_cache_step_{cache_key}"
        last_step = state.get(cache_step_key, None)
        should_recompute = (last_step is None) or ((step - last_step) >= refresh)

        if should_recompute:
            u, pm_iters, ns_iters = self._precondition(tensor, C, p, step)
            # store a cloned tensor to avoid unexpected in-place mutations
            state[cache_val_key] = u.clone()
            state[cache_step_key] = step
            return u, pm_iters, ns_iters
        else:
            u = state[cache_val_key]
            return u, 0, 0

    def _apply_first_moment(
        self,
        tensor: torch.Tensor,
        state: dict,
        momentum: float,
        nesterov: bool,
        beta1: float,
        prefix: str = "",
    ):
        """Apply Adam beta1 EMA or classical momentum on the given tensor.
        Prefix controls state keys suffix (e.g., "", "_pre"). Returns the updated tensor.
        """
        if beta1 > 0.0:
            key = f"dap_moment1{prefix}"
            if key not in state:
                state[key] = torch.zeros_like(tensor)
            m = state[key]
            m.lerp_(tensor, 1 - beta1)
            return m
        else:
            key = f"momentum_buffer{prefix}"
            if key not in state:
                state[key] = torch.zeros_like(tensor)
            buf = state[key]
            buf.mul_(momentum).add_(tensor)
            if nesterov:
                return tensor.add(buf, alpha=momentum)
            else:
                return buf

    def _adam_numerator(
        self, m_t: torch.Tensor, beta1: float, step: int
    ) -> torch.Tensor:
        # m̂_t = m_t / (1 - β₁^t); if β₁==0, denominator=1 → returns m_t
        return m_t / (1.0 - (beta1**step))

    def _nadam_numerator(
        self, m_t: torch.Tensor, g_t: torch.Tensor, beta1: float, step: int
    ) -> torch.Tensor:
        # PyTorch-style Nadam (constant β₁): m̂_t = (β₁ m_t)/(1-β₁^{t+1}) + ((1-β₁) g_t)/(1-β₁^t)
        return (beta1 * m_t) / (1.0 - (beta1 ** (step + 1))) + ((1.0 - beta1) * g_t) / (
            1.0 - (beta1**step)
        )

    def _adam_denominator(
        self, v_t: torch.Tensor, beta2: float, step: int, eps: float
    ) -> torch.Tensor:
        # v̂_t = v_t / (1 - β₂^t)  → denom = sqrt(v̂_t) + eps
        # For β₂==0 we’ll skip this path entirely (see step()) to avoid unnecessary work.
        bc2 = 1.0 - (beta2**step)
        return v_t.sqrt() / (bc2**0.5) + eps

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.
        Args:
        closure (Callable, optional): A closure that reevaluates the model
            and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.debug_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            optimizer_step_t0 = time.perf_counter()
            total_power_method_iters = 0
            total_ns_iters = 0
            num_dap_params = 0

        for group in self.param_groups:
            ############################
            #           DAP           #
            ############################
            lr = group["lr"]
            wd = group["wd"]

            momentum = group["momentum"]
            damping = group["damping"]
            dap_beta1 = group["dap_beta1"]
            dap_beta2 = group["dap_beta2"]
            dap_eps = group["dap_eps"]
            moments_on_precond = group["moments_on_precond"]
            adagradnorm = group["adagradnorm"]

            params = [p for p in group["params"] if self.state[p]["use_dap"]]

            # Before applying updates, finalize accumulated covariance per parameter if present
            if not self.disable_preconditioning:
                for p, mod in self.param_to_module.items():
                    count = mod.C_accum_count.item()
                    if not count:
                        continue

                    C_sum = mod.C_accum_sum
                    C_mean = C_sum / count

                    state = self.state[p]
                    C_prev = state.get("C_ema", None)

                    if self.debug_timing and C_prev is not None:
                        diff = C_mean - C_prev
                        rel = (1.0 - self.ema_beta) * (
                            diff.norm("fro") / C_prev.norm("fro").clamp_min(1e-12)
                        )
                        self._C_ema_rel_change_accum += rel.item()
                        self._C_ema_update_count += 1

                    if C_prev is None or self.ema_beta == 0.0:
                        state["C_ema"] = C_mean.clone()
                    else:
                        C_prev.mul_(self.ema_beta).add_(
                            C_mean, alpha=(1.0 - self.ema_beta)
                        )

                    mod.C_accum_sum.zero_()
                    mod.C_accum_count.zero_()
                    mod._xtx_sampler.reset()

            # apply weight updates
            for i, p in enumerate(params):
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                # calc momentum / first moment and preconditioning based on mode
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1
                step = state["step"]

                use_nadam = group["nesterov"] and dap_beta1 > 0.0

                # Determine if we can/should precondition this parameter this step
                C_eff = None
                if not self.disable_preconditioning:
                    C_cur = state.get("C_ema", None)
                    if C_cur is not None:
                        if damping:
                            trace_C = torch.trace(C_cur)
                            damp_val = damping * trace_C / C_cur.shape[0]
                            I = torch.eye(
                                C_cur.shape[0], dtype=C_cur.dtype, device=C_cur.device
                            )
                            C_eff = C_cur + I * damp_val
                        else:
                            C_eff = C_cur

                if moments_on_precond:
                    # Precondition raw gradient if available; otherwise identity
                    if C_eff is not None:
                        u_pre, pm_iters, ns_iters = self._precondition_cached(
                            g, C_eff, p, step, cache_key="grad"
                        )
                        if self.debug_timing:
                            total_power_method_iters += pm_iters
                            total_ns_iters += ns_iters
                    else:
                        u_pre, pm_iters, ns_iters = g, 0, 0

                    m_t = self._apply_first_moment(
                        u_pre,
                        state,
                        momentum,
                        group["nesterov"],
                        dap_beta1,
                        prefix="_pre",
                    )
                    # Numerator in preconditioned space (bias-corrected)
                    num = (
                        self._nadam_numerator(m_t, u_pre, dap_beta1, step)
                        if use_nadam
                        else self._adam_numerator(m_t, dap_beta1, step)
                    )

                    u = num
                    second_moment_src = u_pre
                else:
                    # First compute momentum/EMA on raw gradient, then (optionally) precondition
                    m_t = self._apply_first_moment(
                        g,
                        state,
                        momentum,
                        group["nesterov"],
                        dap_beta1,
                        prefix="",
                    )

                    if use_nadam:
                        num_grad = self._nadam_numerator(m_t, g, dap_beta1, step)
                    elif dap_beta1 > 0.0:
                        num_grad = self._adam_numerator(m_t, dap_beta1, step)
                    else:
                        num_grad = m_t

                    if C_eff is not None:
                        u, pm_iters, ns_iters = self._precondition_cached(
                            num_grad, C_eff, p, step, cache_key="numgrad"
                        )
                        if self.debug_timing:
                            total_power_method_iters += pm_iters
                            total_ns_iters += ns_iters
                    else:
                        u, pm_iters, ns_iters = num_grad, 0, 0

                    second_moment_src = u

                if self.debug_timing:
                    num_dap_params += 1

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                if dap_beta2 > 0.0:
                    if adagradnorm:
                        # Maintain a scalar EMA of the squared RAW gradient norm.
                        # Uses the same beta (dap_beta2) and bias correction as Adam's second moment.
                        key = "grad_sqnorm_ema"
                        if key not in state:
                            state[key] = torch.tensor(
                                0.0, device=p.device, dtype=p.dtype
                            )
                        # squared L2 norm of the raw gradient
                        sqnorm = g.square().sum()
                        state[key].lerp_(sqnorm, 1.0 - dap_beta2)

                        # Bias-corrected denominator: sqrt(v_hat) + eps
                        bc2 = 1.0 - (dap_beta2**step)
                        denom = (state[key] / bc2).sqrt() + dap_eps
                        p.data.add_(u / denom, alpha=-lr)
                    else:
                        # Adam-style per-parameter normalization and bias correction (epsilon in denominator)
                        if "dap_moment2" not in state:
                            state["dap_moment2"] = torch.zeros_like(u)
                        v = state["dap_moment2"]
                        v.lerp_(second_moment_src.square(), 1 - dap_beta2)
                        denom = self._adam_denominator(v, dap_beta2, step, dap_eps)
                        p.data.add_(u / denom, alpha=-lr)
                else:
                    # No second-moment normalization.
                    p.data.add_(u, alpha=-lr)

            ############################
            #       AdamW backup       #
            ############################

            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]
            params = [p for p in group["params"] if not self.state[p]["use_dap"]]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        if self.debug_timing:
            # If we recorded any CUDA events, sync the relevant devices
            if self._pending_timing_events:
                devices_to_sync = {
                    rec[4]
                    for rec in self._pending_timing_events
                    if rec[0] == "cuda" and rec[4] is not None
                }
                for dev in devices_to_sync:
                    torch.cuda.synchronize(dev)

            power_time_add = 0.0
            pinv_time_add = 0.0
            xtx_time_add = 0.0
            xtx_updates = 0

            # Drain all timing records (both CUDA-event and CPU-wall-clock)
            for rec in self._pending_timing_events:
                kind = rec[0]
                if kind == "cuda":
                    _, label, ev_start, ev_end, _dev = rec
                    try:
                        ms = ev_start.elapsed_time(ev_end)
                    except Exception:
                        ms = 0.0
                    sec = ms / 1000.0
                else:  # "cpu": ('cpu', label, seconds)
                    _, label, sec = rec

                if label == "power":
                    power_time_add += sec
                elif label == "pinv":
                    pinv_time_add += sec
                elif label == "xtx":
                    xtx_time_add += sec
                    xtx_updates += 1

            self._pending_timing_events.clear()

            # Roll into step accumulators used by the printout
            self._XtX_update_time_accum += xtx_time_add
            self._XtX_update_count += xtx_updates

            total_power_method_time = power_time_add
            total_pseudo_inverse_time = pinv_time_add

            optimizer_step_elapsed = time.perf_counter() - optimizer_step_t0

            avg_C_ema_rel_change = self._C_ema_rel_change_accum / max(
                self._num_dap_params_total, 1
            )
            dap_total = (
                self._XtX_update_time_accum
                + total_power_method_time
                + total_pseudo_inverse_time
            )

            print(
                f"[DAP] step={self._debug_step_idx} "
                f"XtX={self._XtX_update_time_accum:.4f}s/{self._XtX_update_count}upds "
                f"power={total_power_method_time:.4f}s/{total_power_method_iters}it "
                f"pinv={total_pseudo_inverse_time:.4f}s/{total_ns_iters}it "
                f"Crel={avg_C_ema_rel_change:.6f} "
                f"opt_step={optimizer_step_elapsed:.4f}s "
                f"dap_total={dap_total:.4f}s"
            )

            self._debug_step_idx += 1

            # reset per-step accumulators
            self._C_ema_rel_change_accum = 0.0
            self._XtX_update_time_accum = 0.0
            self._XtX_update_count = 0

        return loss
