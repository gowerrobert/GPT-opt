import os
import re
import shlex
import json
from glob import glob
from itertools import product
from pathlib import Path
import numpy as np

import pandas as pd
from omegaconf import OmegaConf

from gptopt.utils import hash_config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


_RUNHYDRA_VAR_RE = re.compile(r"\$(\w+)|\$\{(\w+)\}")


def _find_repo_root() -> Path:
    start = Path(__file__).resolve() if "__file__" in globals() else Path.cwd().resolve()
    for p in [start, *start.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "README.md").exists():
            return p
    return Path.cwd().resolve()


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _parse_bash_var_defaults(script_path: str) -> dict:
    """
    Parse simple defaults like: var=${1:-0.001} or var=${2:-cosine}
    Returns {var: "0.001", ...} (strings).
    """
    defaults = {}
    text = Path(script_path).read_text().splitlines()
    pat = re.compile(r"^\s*([A-Za-z_]\w*)=\$\{\d+:-([^}]*)\}\s*$")
    for line in text:
        m = pat.match(line)
        if m:
            defaults[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return defaults


def _extract_run_hydra_overrides(script_path: str) -> list[str]:
    """
    Extract tokens after 'run_hydra.py' from the bash script (supports line continuations with '\').
    Keeps only Hydra-style overrides containing '=' (including '+key=val').
    """
    lines = Path(script_path).read_text().splitlines()

    cmd_lines = []
    capturing = False
    for line in lines:
        if not capturing and ("run_hydra.py" in line and "python" in line):
            capturing = True
        if capturing:
            cmd_lines.append(line.rstrip())
            if not line.rstrip().endswith("\\"):
                break

    if not cmd_lines:
        raise ValueError(f"Could not find a 'python ... run_hydra.py ...' command in {script_path}")

    cmd = " ".join([l[:-1] if l.endswith("\\") else l for l in cmd_lines])
    toks = shlex.split(cmd)

    # drop prefix up to and including run_hydra.py
    if "run_hydra.py" in toks:
        toks = toks[toks.index("run_hydra.py") + 1 :]

    # keep only hydra overrides (key=value, +key=value)
    overrides = [t for t in toks if "=" in t and not t.startswith((">", "2>", "|"))]
    return overrides


def _vars_in_overrides(overrides: list[str]) -> set[str]:
    vars_ = set()
    for t in overrides:
        for a, b in _RUNHYDRA_VAR_RE.findall(t):
            vars_.add(a or b)
    return vars_


def _format_filename_like_run_hydra(cfg) -> str:
    training_params = cfg["training"]["training_params"]
    opt_config = cfg["optimizer"]["optimizer_params"]
    model_config = cfg["model"]["config"]

    config_hash = hash_config(
        OmegaConf.to_container(opt_config),
        OmegaConf.to_container(training_params),
        OmegaConf.to_container(model_config),
    )

    file_name = f"{opt_config['name']}-lr-{opt_config['lr']}-{opt_config['lr_schedule']}"
    if "muon_lr" in opt_config:
        file_name += f"-muonlr-{opt_config['muon_lr']}"
    if "max_norm_tr" in opt_config:
        file_name += f"-maxnorm-{opt_config['max_norm_tr']}"
    file_name += f"-{config_hash}"
    return file_name + ".json"


def load_sweep_jsons(param_configs, script_name):
    """
    Load ONLY JSONs that would be written by run_hydra.py for:
      ./slurm_scripts/submit.sh <script_name> <param_configs> ...

    Mechanism:
      - parse script_name to get Hydra overrides
      - take Cartesian product over swept bash variables from param_configs
      - compose Hydra config for each combo to compute exact expected output filename
      - load only matching files from outputs/ and logs/
    """
    repo_root = _find_repo_root()
    print(repo_root)

    overrides_template = _extract_run_hydra_overrides(script_name)
    used_vars = _vars_in_overrides(overrides_template)

    sweep_cfg = _read_json(param_configs)
    var_defaults = _parse_bash_var_defaults(script_name)

    # Build sweep values (strings) for each used bash var
    sweep_vars = []
    sweep_vals = []
    for v in sorted(used_vars):
        if v in sweep_cfg:
            vals = sweep_cfg[v]
            if not isinstance(vals, list):
                vals = [vals]
            sweep_vars.append(v)
            sweep_vals.append([str(x) for x in vals])
        elif v in var_defaults:
            sweep_vars.append(v)
            sweep_vals.append([str(var_defaults[v])])
        else:
            raise ValueError(
                f"Variable '${v}' is used in {script_name} run_hydra overrides, "
                f"but not found in {param_configs} and no default like {v}=${{...:-...}} was parsed."
            )

    # Compose hydra configs and compute expected filenames
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra

    expected = {} 
    hydra_conf_dir = repo_root / "hydra_conf"
    if not hydra_conf_dir.exists():
        raise FileNotFoundError(f"Expected hydra config dir at {hydra_conf_dir}")

    for combo in product(*sweep_vals):
        subs = dict(zip(sweep_vars, combo))
        overrides = []
        for t in overrides_template:
            tt = t
            for k, val in subs.items():
                tt = tt.replace(f"${{{k}}}", val).replace(f"${k}", val)
            overrides.append(tt)

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(hydra_conf_dir), version_base=None):
            cfg = compose(config_name="config", overrides=overrides)

        expected[_format_filename_like_run_hydra(cfg)] = subs 
    print(len(expected), len(list(product(*sweep_vals))))

    # Load only files with those basenames
    search_roots = [repo_root / "outputs", repo_root / "logs"]
    rows = []
    used_names = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in glob(str(root / "**" / "*.json"), recursive=True):
            name = Path(path).name
            if name not in expected:
                continue
            # print(path)
            used_names.append(name)
            try:
                d = json.loads(Path(path).read_text())
            except Exception:
                continue

            losses = d.get("losses", [])
            val_losses = d.get("val_losses", [])
            kq_max = np.array(d.get("kq_max", []))
            if not isinstance(losses, list) or not losses:
                continue
            ri_values = {
                    "path": path,
                    "final_train_loss": losses[-1],
                    "min_val_loss": (min(val_losses) if isinstance(val_losses, list) and val_losses else float("nan")),
                    "kq_max": (kq_max.max() if kq_max.size else float("nan")),
                    "kq_median": (np.median(kq_max) if kq_max.size else float("nan")), 
                    "kq_mean": (kq_max.mean() if kq_max.size else float("nan")),
                } | expected[name]
            rows.append(
                ri_values
            )

    unused = set(expected.keys()) - set(used_names)
    if unused:
        print(f"{unused=}")
        for name in unused:
            print(expected[name])
    # print(search_roots)
    return pd.DataFrame(rows), unused



def plot_lr_sweep_over_models(
    df,
    colormap="husl",
    outfilename=None,
    ycol="min_val_loss",
    ylog=False,
    ymargin=0.05,
):
    """
    Plot a LR sweep from a DataFrame.

    Expects columns:
      - df["model"] (line grouping)
      - df["lr"] (x axis)
      - df[ycol] (y axis)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    xcol = "lr"
    if xcol not in df.columns:
        raise KeyError(f"Missing column: {xcol}")
    if "model" not in df.columns:
        raise KeyError("Missing column: model")
    if ycol not in df.columns:
        raise KeyError(f"Missing column: {ycol}")

    models = df["model"].unique()
    base_colors = sns.color_palette(colormap, len(models))
    colormap = {m: base_colors[i] for i, m in enumerate(models)}
    ylabel = {
        "min_val_loss": "Minimum Validation Loss",
        "final_train_loss": "Final Training Loss",
        "kq_max": "Maximum KQ Value",
        "kq_median": "Median KQ Value",
        "kq_mean": "Mean KQ Value",
    }

    fig, ax = plt.subplots(figsize=(6, 4))

    all_y = []
    for model, g in df.groupby("model", sort=False):
        g = g[[xcol, ycol]].dropna().copy()
        g[xcol] = pd.to_numeric(g[xcol], errors="coerce")
        g[ycol] = pd.to_numeric(g[ycol], errors="coerce")
        g = g.dropna(subset=[xcol, ycol]).sort_values(xcol, ascending=True)

        if g.empty:
            continue

        xs = g[xcol].to_numpy()
        ys = g[ycol].to_numpy()
        all_y.append(ys)

        ax.plot(
            xs,
            ys,
            label=str(model),
            color=colormap.get(model, "#000000"),
            linestyle=None,
            linewidth=3.0,
        )

    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel(ylabel.get(ycol, ycol))
    ax.legend(loc="upper right")
    ax.grid(axis="both", lw=0.4, ls="--", zorder=0)
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)

    # automatic y-limits + relative margin
    if all_y:
        y = np.concatenate(all_y)
        if ylog:
            y = y[np.isfinite(y) & (y > 0)]
        else:
            y = y[np.isfinite(y)]

        if y.size:
            ymin = float(np.min(y))
            ymax = float(np.max(y))
            if ylog:
                # multiplicative padding in log space
                pad = max(float(ymargin), 0.0)
                ax.set_ylim(ymin / (10 ** pad), ymax * (10 ** pad))
            else:
                # additive padding relative to range
                pad = (ymax - ymin) * float(ymargin)
                if pad == 0.0:
                    pad = abs(ymax) * float(ymargin) if ymax != 0 else 1e-6
                ax.set_ylim(ymin - pad, ymax + pad)

    if outfilename:
        fig.savefig(outfilename, format="pdf", bbox_inches="tight")
    return fig, ax

def plot_heatmat_grid(df_in: pd.DataFrame, cmap="viridis") -> None:
    """Seaborn heatmap of min_val_loss over (mu_frac, rho_over_lr), sorted axes."""

    label = {
        "mu_frac": r"$\mu_{\text{frac}}$",
        "rho_over_lr": r"$\rho/\mu$", 
        "min_val_loss": "Minimum Validation Loss",
        "fin_val_loss": "Final Validation Loss",
        "final_train_loss": "Final Training Loss",
    }

    def _pow10(x: float) -> str:
        if x == 0 or not np.isfinite(x):
            return r"$0$"
        e = int(np.round(np.log10(abs(x))))
        return rf"$10^{{{e}}}$"

    req = ["mu_frac", "rho_over_lr", "min_val_loss"]
    missing = [c for c in req if c not in df_in.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = df_in.copy()
    df["mu_frac"] = pd.to_numeric(df["mu_frac"], errors="coerce")
    df["rho_over_lr"] = pd.to_numeric(df["rho_over_lr"], errors="coerce")
    df["min_val_loss"] = pd.to_numeric(df["min_val_loss"], errors="coerce")
    df = df.dropna(subset=req)

    if df.empty:
        print("No rows with required numeric columns.")
        return

    # Sort x ascending; sort y ascending (then flip axis so it increases bottom->top)
    pivot = (
        df.pivot_table(
            index="rho_over_lr",
            columns="mu_frac",
            values="min_val_loss",
            aggfunc="min",
        )
        .sort_index(axis=0)  # rho_over_lr ascending
        .sort_index(axis=1)  # mu_frac ascending
    )

    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        print("Not enough variation in mu_frac/rho_over_lr for a heatmap.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        cbar_kws={"label": label["min_val_loss"]},
    )

    # Make rho_over_lr increase bottom -> top
    ax.invert_yaxis()

    ax.set_xlabel(label["mu_frac"])
    ax.set_ylabel(label["rho_over_lr"])
    ax.set_title(f"{label['min_val_loss']} over ({label['mu_frac']}, {label['rho_over_lr']})")

    ax.set_xticklabels([_pow10(float(x)) for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels([_pow10(float(y)) for y in pivot.index], rotation=0)

    fig.tight_layout()
