import numpy as np
from pathlib import Path
import json
import re

# --- Data loading ------------------------------------------------------------

HP_DIR_PATTERN = re.compile(r"^bs-(?P<bs>\d+)-lr-(?P<lr>[-+eE0-9\.]+)-wd-(?P<wd>[-+eE0-9\.]+)$")
# Match "-muonlr-<float>" and capture only the numeric value (no trailing '-')
MUON_LR_PATTERN = re.compile(
    r"(?:^|-)muonlr-(?P<val>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)"
)

def load_outputs(root_dir: str):
    """
    Load all JSON logs from:
        outputs/<model>/<data>/<method>/bs-<B>-lr-<LR>-wd-<WD>/*.json
    Attaches method, batch_size, learning_rate, weight_decay to each output dict.
    Assumes all directories follow this convention (no fallbacks).
    """
    base = Path(root_dir)
    if not base.exists():
        raise FileNotFoundError(f"{root_dir} does not exist")
    outputs = []
    for method_dir in base.iterdir():
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name
        for hp_dir in method_dir.iterdir():
            if not hp_dir.is_dir():
                continue
            m = HP_DIR_PATTERN.match(hp_dir.name)
            if not m:
                continue
            bs = int(m.group("bs"))
            lr = float(m.group("lr"))
            wd = float(m.group("wd"))
            for fp in hp_dir.glob("*.json"):
                try:
                    with fp.open("r") as f:
                        out = json.load(f)
                    out["method"] = method_name
                    out["batch_size"] = bs
                    out["learning_rate"] = lr
                    out["weight_decay"] = wd
                    out["filename"] = fp.name  # record the raw json filename
                    m_mu = MUON_LR_PATTERN.search(fp.name)
                    if m_mu:
                        try:
                            out["muon_lr"] = float(m_mu.group("val"))
                        except ValueError:
                            pass  # ignore parse errors silently
                    outputs.append(out)
                except Exception as e:
                    print(f"Skipping {fp}: {e}")
    return outputs

# --- Plot helpers ------------------------------------------------------------

def method_name(output):
    return output["method"]  # always present by assumption

def get_alpha_from_lr(lr, min_alpha=0.3, max_alpha=1.0, lr_range=None):
    if lr_range and lr_range[0] == lr_range[1]:
        return max_alpha
    return min_alpha + (max_alpha - min_alpha) * (lr - lr_range[0]) / (lr_range[1] - lr_range[0])

def percentage_of_epoch(output, field, num_epochs):
    total_iterations = len(output[field])
    return [i / total_iterations * num_epochs for i in range(total_iterations)]

def plot_data(ax, outputs, num_epochs, field, ylabel,
              colormap, linestylemap, lr_ranges, alpha_func,
              zorder_func=None, wallclock=False):
    plotted = set()
    for out in outputs:
        name = method_name(out)
        lr = out["learning_rate"]
        if name not in lr_ranges:
            continue
        alpha = alpha_func(lr, lr_range=lr_ranges[name])
        label = None
        if name not in plotted:
            r0, r1 = lr_ranges[name]
            label = f"{name} lr={r0:.4f}" if r0 == r1 else f"{name} lr in [{r0:.4f}, {r1:.4f}]"
        zorder = zorder_func(name) if zorder_func else 1

        if wallclock:
            assert len(out["step_times"]) % len(out[field]) == 0
            step_factor = len(out["step_times"]) // len(out[field])
            step_times = np.array(out["step_times"]).reshape(len(out[field]), step_factor).sum(axis=1)
            xs = np.cumsum(step_times)
            xlabel = "Seconds"
        else:
            xs = percentage_of_epoch(out, field, num_epochs)
            xlabel = "Epochs"

        ax.plot(xs,
                out[field],
                label=label,
                color=colormap[name],
                linewidth=2,
                linestyle=linestylemap[name],
                alpha=alpha,
                zorder=zorder)
        plotted.add(name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)

def plot_step_size_and_lr(ax, outputs, colormap, linestylemap, lr_ranges, alpha_func):
    plotted = set()
    for out in outputs:
        if 'step_size_list' not in out or 'learning_rates' not in out:
            continue
        name = method_name(out)
        lr = out["learning_rate"]
        if name not in lr_ranges:
            continue
        alpha = alpha_func(lr, lr_range=lr_ranges[name])
        label = None
        if name not in plotted:
            r0, r1 = lr_ranges[name]
            label = f"{name} lr={r0:.1e}" if r0 == r1 else f"{name} lr in [{r0:.1e}, {r1:.1e}]"

        ax.plot(range(len(out['step_size_list'])),
                out['step_size_list'],
                label=label,
                color=colormap[name],
                linewidth=2,
                linestyle=linestylemap[name],
                alpha=alpha)
        ax.plot(range(len(out['learning_rates'])),
                out['learning_rates'],
                color=colormap[name],
                linewidth=1.5,
                linestyle='--',
                alpha=alpha)
        plotted.add(name)
    return plotted

# --- Smoothing ---------------------------------------------------------------

def smoothen_curve_exp(data, num_points=None, beta=0.05):
    smooth = [data[0]]
    acc = data[0]
    total = len(data)
    if num_points is None:
        num_points = total
    interval = max(1, total // num_points)
    for i, v in enumerate(data):
        if np.isnan(v):
            continue
        acc = (1 - beta) * acc + beta * v
        if i % interval == 0:
            smooth.append(acc)
    return smooth

def smoothen_dict(d, num_points=None, beta=0.05):
    if 'losses' in d:
        d['losses'] = smoothen_curve_exp(d['losses'], num_points=None, beta=beta)
    # step_times not smoothed (wallclock grouping assumption)
    # step_size_list intentionally left untouched due to multi-value per iteration variants


