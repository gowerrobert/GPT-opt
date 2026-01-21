

def name_by_param(named_params):
    param_groups = list(named_params)
    name_by_param = {}
    params = []
    if not isinstance(param_groups[0], dict):
        param_groups = [{"params": param_groups}]
    for group in param_groups:
        for name, p in group["params"]:
            if not p.requires_grad:
                continue
            params.append(p)
            name_by_param[p] = name
    return params, name_by_param
