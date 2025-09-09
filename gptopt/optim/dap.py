import torch
from torch import nn
from torch.optim import Optimizer
from gptopt.linalg_utils import ns_pinv, ns_pinv_v2, accelerated_ns_pinv, power_method
from typing import Optional

import os
import sys
import time

class SimpleTimer:
    """
    Context manager:
      - On CUDA: records start/end Events and appends to `pending` for later resolution.
      - On CPU: measures perf_counter() and exposes `elapsed` immediately.
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
                self.pending.append((self.label, self._ev_start, self._ev_end, self.device))
        else:
            self.elapsed = time.perf_counter() - self._t0
        return False

    @property
    def seconds(self) -> float:
        # Returns CPU elapsed seconds; 0.0 for CUDA or when disabled
        return float(self.elapsed) if (self.enabled and self.elapsed is not None) else 0.0



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
        sgd_update: bool = False,
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
        self.sgd_update = sgd_update
        self.scalar = scalar
        self.include_output = include_output
        self.include_embed = include_embed
        self.use_ns_pinv = use_ns_pinv
        self.ns_pinv_steps = ns_pinv_steps
        self.rcond = rcond
        # Debug timing controls
        self.debug_timing = debug_timing
        self._debug_step_idx = 0
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
            raise ValueError(f"Invalid spectral_norm_estimator: {spectral_norm_estimator}")
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
            self.scalar and self.sgd_update
        ), "choose either scalar or sgd update"

        # Control how often to refresh preconditioned vector; None disables caching
        if refresh_precond_iters is not None:
            refresh_precond_iters = int(refresh_precond_iters)
            if refresh_precond_iters <= 0:
                refresh_precond_iters = None
        self.refresh_precond_iters = refresh_precond_iters

        # Register hooks only if we actually intend to use the covariance statistics.
        if not self.sgd_update:
            self._register_input_hooks(model, dap_params)

    def _register_input_hooks(self, model: nn.Module, dap_params):
        """Attach hooks **only** to modules whose weights are in dap_params."""
        # Build a fast identity‐based lookup for which weights to hook
        dap_param_ids = {id(p) for p in dap_params}

        # Map each parameter object to its owning nn.Linear module
        param_to_module = {}
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if id(module.weight) in dap_param_ids:
                    param_to_module[module.weight] = module

        # Sanity check: ensure every dap_param got assigned
        unclaimed = [p for p in dap_params if p not in param_to_module]
        if unclaimed:
            raise ValueError(
                f"Some DAP params not owned by any Linear module: {unclaimed}"
            )

        # Attach one forward‐hook per relevant module to capture XᵀX in state[p]["C"]
        seen = set()
        for p_ref, module in param_to_module.items():
            if id(p_ref) in seen:
                continue
            seen.add(id(p_ref))

            def make_hook(p_ref):
                def hook(mod, inp, out):
                    # Skip accumulation and timing during eval (validation/inference)
                    if not mod.training:
                        return
                    # Time the XtX computation as it is the dominant cost during accumulation
                    dev = p_ref.device if (self.debug_timing and p_ref.is_cuda) else None
                    with SimpleTimer("xtx", dev, self._pending_timing_events, enabled=self.debug_timing) as timer:
                        X = inp[0].detach()
                        X = X.to(dtype=p_ref.dtype, device=p_ref.device)
                        X_flat = X.reshape(-1, X.shape[-1]).contiguous()
                        XtX = X_flat.transpose(0, 1) @ X_flat
                    self._XtX_update_time_accum += timer.seconds
                    if self.debug_timing:
                        self._XtX_update_count += 1
                    state = self.state[p_ref]

                    # Accumulate XtX and sample count across micro-batches; finalize in step().
                    if "C_accum_sum" not in state:
                        state["C_accum_sum"] = XtX
                        state["C_accum_count"] = int(X_flat.shape[0])
                    else:
                        state["C_accum_sum"].add_(XtX)
                        state["C_accum_count"] += int(X_flat.shape[0])

                return hook

            _hook = make_hook(p_ref)
            _hook = torch._dynamo.disable()(_hook)
            module.register_forward_hook(_hook, prepend=False)

    def _estimate_spectral_norm(self, C: torch.Tensor, psd=True, return_iters=True):
        if self.spectral_norm_estimator == "power_method":
            return power_method(
                C, max_iters=self.ns_pinv_steps, psd=psd, return_iters=return_iters
            )
        elif self.spectral_norm_estimator == "frobenius":
            if return_iters:
                return torch.linalg.norm(C, ord='fro'), 0
            else:
                return torch.linalg.norm(C, ord='fro')
        else:
            raise ValueError(f"Invalid spectral_norm_estimator: {self.spectral_norm_estimator}")

    def _precondition(
        self, tensor: torch.Tensor, C: torch.Tensor, p: torch.Tensor, step: int
    ):
        """Precondition a gradient-like tensor using covariance C.
        Returns (preconditioned_tensor, pm_time, pinv_time, pm_iters, ns_iters).
        """
        pm_time = 0.0
        pinv_time = 0.0
        pm_iters = 0
        ns_iters = 0

        C_op = C.to(self.op_dtype)
        v_op = tensor.to(self.op_dtype)

        if self.use_ns_pinv:
            device = p.device if p.is_cuda else None
            with SimpleTimer("power", device, self._pending_timing_events, enabled=self.debug_timing) as t:
                sig_max, pm_iters = self._estimate_spectral_norm(
                    C_op, psd=True, return_iters=True
                )
            pm_time += t.seconds
            sig_max = sig_max.item()
            eps = sig_max * self.rcond
            with SimpleTimer("pinv", device, self._pending_timing_events, enabled=self.debug_timing) as t2:
                if self.accelerated == "add_eps_1e-3":
                    Cinv, info = accelerated_ns_pinv(
                        C_op,
                        l=eps,
                        u=1.2*sig_max,
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
                        u=1.2*sig_max,
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
                        diagnostics=False
                    )
            pinv_time += t2.seconds
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
                with SimpleTimer("power", device, self._pending_timing_events, enabled=self.debug_timing) as t:
                    sig_max, pm_iters = self._estimate_spectral_norm(
                        C_op, psd=True, return_iters=True
                    )
                pm_time += t.seconds
                u_local = v_op / sig_max
            else:
                device = p.device if p.is_cuda else None
                with SimpleTimer("pinv", device, self._pending_timing_events, enabled=self.debug_timing) as t:
                    pinv_mat = torch.linalg.pinv(C_op, rtol=self.rcond)
                pinv_time += t.seconds
                u_local = v_op @ pinv_mat

        return u_local.to(p.dtype), pm_time, pinv_time, pm_iters, ns_iters

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
        Returns (u, pm_time, pinv_time, pm_iters, ns_iters), where timing/iters are zero if reused from cache.
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
            u, pm_time, pinv_time, pm_iters, ns_iters = self._precondition(
                tensor, C, p, step
            )
            # store a cloned tensor to avoid unexpected in-place mutations
            state[cache_val_key] = u.clone()
            state[cache_step_key] = step
            return u, pm_time, pinv_time, pm_iters, ns_iters
        else:
            u = state[cache_val_key]
            return u, 0.0, 0.0, 0, 0

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
            torch.cuda.synchronize()
            optimizer_step_t0 = time.perf_counter()
            total_power_method_time = 0.0
            total_pseudo_inverse_time = 0.0
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
            if not self.sgd_update:
                for p in group["params"]:
                    state = self.state[p]
                    C_sum = state.get("C_accum_sum", None)
                    count = state.get("C_accum_count", 0)
                    if C_sum is not None and count > 0:
                        C_new = C_sum / float(count)

                        # --- BEGIN simple C_ema update (no timing) ---
                        C_prev = state.get("C_ema", None)
                        if C_prev is None or self.ema_beta == 0.0:
                            state["C_ema"] = C_new.detach()
                        else:
                            state["C_ema"].mul_(self.ema_beta).add_(
                                C_new, alpha=1.0 - self.ema_beta
                            )
                        # track relative change only for debug reporting
                        if self.debug_timing and C_prev is not None:
                            diff_f = (C_new - C_prev).float()
                            num_f = diff_f.norm(p="fro")
                            denom = C_prev.float().norm(p="fro").clamp_min(1e-12)
                            rel_change = (1.0 - self.ema_beta) * (num_f / denom)
                            self._C_ema_rel_change_accum += float(rel_change.item())
                            self._C_ema_update_count += 1
                        # --- END simple C_ema update ---

                        # Clear accumulators for next step
                        del state["C_accum_sum"]
                        del state["C_accum_count"]

            # apply weight updates
            for i, p in enumerate(params):

                # sanity check
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

                if self.sgd_update:
                    # Plain (momentum) SGD: moments on raw gradient only
                    u = self._apply_first_moment(
                        g, state, momentum, group["nesterov"], dap_beta1, prefix=""
                    )
                    second_moment_src = u
                else:
                    # Build covariance (with optional damping)
                    C = state.get("C_ema", None)
                    if C is None:
                        # Not yet available: skip preconditioning this step
                        m_t = self._apply_first_moment(
                            g, state, momentum, group["nesterov"], dap_beta1, prefix=""
                        )
                        if use_nadam:
                            u = self._nadam_numerator(m_t, g, dap_beta1, step)
                        elif dap_beta1 > 0.0:
                            u = self._adam_numerator(m_t, dap_beta1, step)
                        else:
                            u = m_t
                        second_moment_src = u
                    else:
                        # Optionally form a damped covariance without mutating C_ema in-place
                        if damping:
                            trace_C = torch.trace(C)
                            damp_val = damping * trace_C / C.shape[0]
                            I = torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
                            C_eff = C + I * damp_val
                        else:
                            C_eff = C

                        # TODO: on iterations where we do not need to compute new preconditioner, we should skip power_method

                        # Preconditioning will be invoked via _precondition directly and timing aggregated inline
                        if moments_on_precond:
                            # First precondition the raw gradient, then apply first moment on preconditioned
                            u_pre, pm_time, pinv_time, pm_iters, ns_iters = (
                                self._precondition_cached(
                                    g, C_eff, p, step, cache_key="grad"
                                )
                            )
                            if self.debug_timing:
                                total_power_method_time += pm_time
                                total_pseudo_inverse_time += pinv_time
                                total_power_method_iters += pm_iters
                                total_ns_iters += ns_iters

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
                            # First compute momentum/EMA on raw gradient, then precondition
                            m_t = self._apply_first_moment(
                                g,
                                state,
                                momentum,
                                group["nesterov"],
                                dap_beta1,
                                prefix="",
                            )

                            if use_nadam:
                                num_grad = self._nadam_numerator(
                                    m_t, g, dap_beta1, step
                                )
                            elif dap_beta1 > 0.0:
                                num_grad = self._adam_numerator(m_t, dap_beta1, step)
                            else:
                                num_grad = m_t

                            u, pm_time, pinv_time, pm_iters, ns_iters = (
                                self._precondition_cached(
                                    num_grad, C_eff, p, step, cache_key="numgrad"
                                )
                            )
                            if self.debug_timing:
                                total_power_method_time += pm_time
                                total_pseudo_inverse_time += pinv_time
                                total_power_method_iters += pm_iters
                                total_ns_iters += ns_iters

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
            # Synchronize per-device if we recorded CUDA events
            if self._pending_timing_events:
                devices_to_sync = set(
                    dev
                    for (_, _, _, dev) in self._pending_timing_events
                    if dev is not None
                )
                for dev in devices_to_sync:
                    torch.cuda.synchronize(dev)
                power_time_add = 0.0
                pinv_time_add = 0.0
                xtx_time_add = 0.0
                for label, ev_start, ev_end, dev in self._pending_timing_events:
                    try:
                        ms = ev_start.elapsed_time(ev_end)
                    except Exception:
                        ms = 0.0

                    if label == "power":
                        power_time_add += ms / 1000.0
                    elif label == "pinv":
                        pinv_time_add += ms / 1000.0
                    elif label == "xtx":
                        xtx_time_add += ms / 1000.0

                total_power_method_time += power_time_add
                total_pseudo_inverse_time += pinv_time_add
                self._XtX_update_time_accum += xtx_time_add
                self._pending_timing_events.clear()

            optimizer_step_elapsed = time.perf_counter() - optimizer_step_t0

            if self.debug_timing:
                avg_C_ema_rel_change = self._C_ema_rel_change_accum / max(
                    num_dap_params, 1
                )
                xtx_time = self._XtX_update_time_accum
                xtx_updates = self._XtX_update_count

                print(
                    f"[DAP] step={self._debug_step_idx} "
                    f"XtX={xtx_time:.4f}s/{xtx_updates}upds "
                    f"power={total_power_method_time:.4f}s/{total_power_method_iters}it "
                    f"pinv={total_pseudo_inverse_time:.4f}s/{total_ns_iters}it "
                    f"params={num_dap_params} "
                    f"Crel={avg_C_ema_rel_change:.6f} "
                    f"total={optimizer_step_elapsed:.4f}s"
                )

            self._debug_step_idx += 1

            # Reset accumulators for next step
            self._C_ema_rel_change_accum = 0.0
            self._XtX_update_time_accum = 0.0
            self._XtX_update_count = 0

        return loss
