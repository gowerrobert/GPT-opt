import torch
import math
import warnings


def name_by_param(named_params):
    param_groups = list(named_params)
    name_by_param = {}
    if not isinstance(param_groups[0], dict):
        param_groups = [{"params": param_groups}]
    for group in param_groups:
        for name, p in group["params"]:
            if not p.requires_grad:
                continue
            name_by_param[p] = name
    return name_by_param


class MyAdamW(torch.optim.Optimizer):
    """AdamW that sees (name, param) and can treat attention Q/K specially.

    Assumes attn.c_attn.weight is stacked [Q; K; V] along dim 0.
    """

    def __init__(
        self,
        named_params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    ):
        self.name_by_param = name_by_param(named_params)

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        super().__init__(named_params, defaults)

    def _is_attn_qkv_weight(self, name: str) -> bool:
        return ("c_attn" in name) and name.endswith("weight")

    @torch.no_grad()
    def step(self, closure=None, ):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        updated = set()
        n_att_layers = 0
        residuals_n_layers = {}
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"] 

            for p in group["params"]:
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
                

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5

                name = self.name_by_param[p]
                updated.add(name) 
                if self._is_attn_qkv_weight(name) and p.ndim == 2:
                    # interpret rows as [Q; K; V]
                    n_embed = p.shape[1] 
                    assert p.shape[0] == 3 * n_embed, print(f"{p.shape}")
                    W_q, G_q = p[:n_embed, :],                  g[:n_embed, :]
                    W_k, G_k = p[n_embed:2 * n_embed, :],       g[n_embed:2 * n_embed, :] 
                    residuals = {}
                    residuals["W_q_norm"] = W_q.norm().item()
                    residuals["W_k_norm"] = W_k.norm().item()
                    residuals["G_k_norm"] = G_k.norm().item()
                    residuals["G_q_norm"] = G_q.norm().item()
                    residuals_n_layers[n_att_layers] = residuals
                    n_att_layers += 1 

                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2) 
                g = buf1 / (eps + buf2.sqrt())
                p.data.mul_(1 - lr * wd)
                p.data.add_(g, alpha=-lr / scale)

        # Sanity check: only require updates for params that had gradients
        names_with_grad = {
            self.name_by_param[p]
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        }
        missing = [n for n in names_with_grad if n not in updated]
        assert not missing, f"[MyAdamW] Params with grad but no update: {missing}"

        return loss, residuals_n_layers

