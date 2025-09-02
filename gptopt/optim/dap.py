import torch
from torch import nn
from torch.optim import Optimizer
from gptopt.linalg_utils import ns_pinv, ns_pinv_v2, power_method

import os
import sys
import time

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
        debug_timing_every: int = 1,
        use_bf16: bool = False,
        use_fp64: bool = False,
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

            if p.ndim >= 2 and not any(
                excluded in name for excluded in excluded_names
            ):
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
        self.debug_timing_every = max(1, int(debug_timing_every))
        self._debug_step_idx = 0
        
        # Precision control
        self.rcond = float(rcond)
        self.bf16 = bool(use_bf16)
        self.fp64 = bool(use_fp64)

        if self.bf16 and self.fp64:
            raise ValueError("use_bf16 and use_fp64 are mutually exclusive")
        
        # dtype used for power_method / pinv / ns_pinv operations
        if self.bf16:
            self.op_dtype = torch.bfloat16
        elif self.fp64:
            self.op_dtype = torch.float64
        else:
            self.op_dtype = torch.float32
        
        assert not (self.scalar and self.sgd_update), "choose either scalar or sgd update"

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
        for p_ref, module in param_to_module.items():
            def make_hook(p_ref):
                def hook(mod, inp, out):
                    X = inp[0].detach()
                    X_flat = X.reshape(-1, X.shape[-1])
                    # store covariance matrix C = Xᵀ X for this param
                    C_new = (X_flat.transpose(0, 1) @ X_flat) / X_flat.shape[0]
                    state = self.state[p_ref]

                    if "C_ema" not in state:
                        state["C_ema"] = C_new
                    else:
                        state["C_ema"].mul_(self.ema_beta).add_(
                            C_new, alpha=1 - self.ema_beta
                        )
                return hook

            module.register_forward_hook(make_hook(p_ref), prepend=False)

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

            params = [p for p in group["params"] if self.state[p]["use_dap"]]

            # apply weight updates
            for i, p in enumerate(params):

                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                # calc momentum / first moment
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1
                step = state["step"]

                if dap_beta1 > 0.0:
                    # Adam-style EMA of gradients overrides classical momentum
                    if "dap_moment1" not in state:
                        state["dap_moment1"] = torch.zeros_like(g)
                    m = state["dap_moment1"]
                    m.lerp_(g, 1 - dap_beta1)
                    g_eff = m
                else:
                    # Classic momentum (optionally Nesterov)
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g_eff = g.add(buf, alpha=momentum)
                    else:
                        g_eff = buf
                
                if self.sgd_update:
                    # Plain (momentum) SGD: use first-moment directly.
                    u = g_eff
                else:
                    C = self.state[p]["C_ema"]
                    if damping:
                        C.diagonal().add_(damping * torch.trace(C) / C.shape[0])

                    C_op = C.to(self.op_dtype)
                    g_op = g_eff.to(self.op_dtype)

                    if self.use_ns_pinv:
                        if self.debug_timing:
                            t_power_start = time.perf_counter()

                        sig_max, pm_iters = power_method(C_op, max_iters=self.ns_pinv_steps, psd=True, return_iters=True)
                        sig_max = sig_max.item()
                        total_power_method_iters += pm_iters
                        eps = sig_max * self.rcond

                        if self.debug_timing:
                            total_power_method_time += (time.perf_counter() - t_power_start)
                            t_pinv_start = time.perf_counter()

                        Cinv, info = ns_pinv_v2(C_op, eps=eps, max_steps=self.ns_pinv_steps, return_iters=True, diagnostics=False)
                        total_ns_iters += int(info.get("iterations", 0))

                        if self.debug_timing:
                            total_pseudo_inverse_time += (time.perf_counter() - t_pinv_start)

                        if torch.isnan(Cinv).any() or torch.isinf(Cinv).any():
                            # Save C matrix for debugging
                            debug_dir = "debug_matrices"
                            os.makedirs(debug_dir, exist_ok=True)
                            matrix_path = os.path.join(debug_dir, f"C_{self.param_to_name[p]}_step_{step}.pt")
                            grad_path = os.path.join(debug_dir, f"grad_{self.param_to_name[p]}_step_{step}.pt")
                            torch.save(C_op, matrix_path)
                            torch.save(g_op, grad_path)

                            # Exit if we encounter NaN/inf in pseudo-inverse
                            print(f"ERROR: NaN/inf detected in pseudo-inverse calculation for parameter {self.param_to_name[p]} at step {step}")
                            print(f"Matrix saved to debug_matrices/C_{self.param_to_name[p]}_step_{step}.pt")
                            sys.exit(1)
                        u_op = g_op @ Cinv
                    else:
                        if self.scalar:
                            if self.debug_timing:
                                t_power_start = time.perf_counter()

                            sig_max, pm_iters = power_method(C_op, max_iters=self.ns_pinv_steps, psd=True, return_iters=True)
                            u_op = g_op / sig_max
                            total_power_method_iters += pm_iters

                            if self.debug_timing:
                                total_power_method_time += (time.perf_counter() - t_power_start)
                        else:
                            # Precondition the gradient using the inverse covariance.
                            # Try lstsq first, fall back to pinv if it fails
                            # u_op = torch.linalg.lstsq(C_op, g_op.T).solution.T
                            if self.debug_timing:
                                t_pinv_start = time.perf_counter()
                            pinv_mat = torch.linalg.pinv(C_op)
                            if self.debug_timing:
                                total_pseudo_inverse_time += (time.perf_counter() - t_pinv_start)
                            u_op = g_op @ pinv_mat
                        
                    u = u_op.to(p.dtype)

                if self.debug_timing:
                    num_dap_params += 1

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # Bias corrections
                bc1 = 1 - dap_beta1 ** step
                bc2 = 1 - dap_beta2 ** step

                if dap_beta2 > 0.0:
                    # Adam-style normalization and bias correction (epsilon in denominator)
                    if "dap_moment2" not in state:
                        state["dap_moment2"] = torch.zeros_like(u)
                    v = state["dap_moment2"]
                    v.lerp_(u.square(), 1 - dap_beta2)
                    denom = v.sqrt() / (bc2 ** 0.5) + dap_eps
                    update = (u / bc1) / denom
                    p.data.add_(update, alpha=-lr)
                else:
                    # No second-moment normalization. Always apply first-moment bias correction (bc1=1 if beta1=0)
                    p.data.add_(u, alpha=-(lr / bc1))

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
            optimizer_step_elapsed = time.perf_counter() - optimizer_step_t0
            if (self._debug_step_idx % self.debug_timing_every) == 0:
                # Print a compact one-line summary
                print(
                    f"[DAP] step={self._debug_step_idx} "
                    f"power={total_power_method_time:.4f}s/{total_power_method_iters}it "
                    f"pinv={total_pseudo_inverse_time:.4f}s/{total_ns_iters}it "
                    f"params={num_dap_params} total={optimizer_step_elapsed:.4f}s"
                )
            self._debug_step_idx += 1

        return loss
    
    
