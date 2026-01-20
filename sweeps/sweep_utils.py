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
from matplotlib.ticker import ScalarFormatter, LogLocator, LogFormatterMathtext



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
    # print(repo_root)

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
    assert len(expected) == len(list(product(*sweep_vals)))

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
                    "min_val_loss": (min(val_losses) if val_losses else float("nan")),
                    "fin_val_loss": (val_losses[-1] if val_losses else float("nan")),
                    "kq_max": (kq_max.max() if kq_max.size else float("nan")),
                    "kq_median": (np.median(kq_max) if kq_max.size else float("nan")), 
                    "kq_mean": (kq_max.mean() if kq_max.size else float("nan")),
                } | expected[name]
            rows.append(
                ri_values
            )

    unused = set(expected.keys()) - set(used_names)
    if unused:
        print(f"Some files are missing: \n{unused}")
        for name in unused:
            print(expected[name])
    # print(search_roots)
    return pd.DataFrame(rows), unused


def load_sweep_jsons_all(d_conf):
    """
    Load ONLY JSONs that would be written by run_hydra.py for multiple models.

    Expected d_conf format:
        d_conf = {
            model_name: (param_configs, list_scripts),
            ...
        }

    Where:
      - param_configs: path (or JSON-able) sweep config like before
      - list_scripts: a script path/name OR a list/tuple of script paths/names

    Mechanism:
      - for each (model, script): compute the exact set of expected output basenames
      - scan outputs/ and logs/ ONCE, loading only matching basenames
      - add "model" column = model_name for each matched row

    Returns:
      (df, missing_by_model)
        - df: pd.DataFrame of matched runs
        - missing_by_model: dict[model_name -> set(basenames)] that were expected but not found
    """
    repo_root = _find_repo_root()

    from pathlib import Path
    from itertools import product
    import json
    import numpy as np
    import pandas as pd
    from glob import glob

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple, set)):
            return list(x)
        return [x]

    def _resolve_under_repo(p):
        p = Path(p)
        if p.is_absolute():
            return p
        cand = repo_root / p
        return cand if cand.exists() else p

    # -----------------------------
    # 1) Precompute expected basenames for *all* models/scripts
    # -----------------------------
    hydra_conf_dir = repo_root / "hydra_conf"
    if not hydra_conf_dir.exists():
        raise FileNotFoundError(f"Expected hydra config dir at {hydra_conf_dir}")

    # basename -> list[meta], where meta contains sweep vars + model/script
    expected_by_name = {}
    expected_names_by_model = {m: set() for m in d_conf.keys()}

    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra

    for model_name, (param_configs, list_scripts) in d_conf.items():
        scripts = _as_list(list_scripts)
        if not scripts:
            continue

        sweep_cfg = _read_json(param_configs)

        for script_name in scripts:
            script_path = _resolve_under_repo(script_name)

            overrides_template = _extract_run_hydra_overrides(str(script_path))
            used_vars = _vars_in_overrides(overrides_template)

            var_defaults = _parse_bash_var_defaults(str(script_path))

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
                        f"Variable '${v}' is used in {script_path} run_hydra overrides, "
                        f"but not found in {param_configs} and no default like {v}=${{...:-...}} was parsed."
                    )

            # sanity: ensure this sweep maps 1-1 to unique output basenames (like original assert)
            n_combos = 1
            for vals in sweep_vals:
                n_combos *= len(vals)

            local_names = set()

            GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=str(hydra_conf_dir), version_base=None):
                for combo in product(*sweep_vals):
                    subs = dict(zip(sweep_vars, combo))

                    overrides = []
                    for t in overrides_template:
                        tt = t
                        for k, val in subs.items():
                            tt = tt.replace(f"${{{k}}}", val).replace(f"${k}", val)
                        overrides.append(tt)

                    cfg = compose(config_name="config", overrides=overrides)
                    basename = _format_filename_like_run_hydra(cfg)

                    local_names.add(basename)

                    meta = {"model": model_name, "script": script_path.name} | subs
                    expected_by_name.setdefault(basename, []).append(meta)
                    expected_names_by_model[model_name].add(basename)

            if len(local_names) != n_combos:
                raise AssertionError(
                    f"{script_path}: expected {n_combos} unique output basenames, got {len(local_names)}. "
                    "This usually means different sweep combos collapse to the same output filename."
                )

    # -----------------------------
    # 2) Scan filesystem ONCE and load only matching basenames
    # -----------------------------
    search_roots = [repo_root / "outputs", repo_root / "logs"]
    rows = []
    used_names = set()
    found_names_by_model = {m: set() for m in d_conf.keys()}

    for root in search_roots:
        if not root.exists():
            continue

        for path in glob(str(root / "**" / "*.json"), recursive=True):
            name = Path(path).name
            metas = expected_by_name.get(name)
            if not metas:
                continue

            used_names.add(name)

            try:
                d = json.loads(Path(path).read_text())
            except Exception:
                continue

            losses = d.get("losses", [])
            val_losses = d.get("val_losses", [])
            kq_max = np.array(d.get("kq_max", []))

            if not isinstance(losses, list) or not losses:
                continue

            metrics = {
                "path": path,
                "final_train_loss": losses[-1],
                "min_val_loss": (min(val_losses) if val_losses else float("nan")),
                "fin_val_loss": (val_losses[-1] if val_losses else float("nan")),
                "kq_max": (kq_max.max() if kq_max.size else float("nan")),
                "kq_median": (np.median(kq_max) if kq_max.size else float("nan")),
                "kq_mean": (kq_max.mean() if kq_max.size else float("nan")),
            }

            # if multiple model/script configs map to same basename, emit one row per meta
            for meta in metas:
                found_names_by_model[meta["model"]].add(name)
                rows.append(metrics | meta)

    # -----------------------------
    # 3) Missing expected files (per model)
    # -----------------------------
    missing_by_model = {}
    for model_name, expected_names in expected_names_by_model.items():
        missing = set(expected_names) - set(found_names_by_model.get(model_name, set()))
        missing_by_model[model_name] = missing
        if missing:
            print(f"[{model_name}] Some files are missing ({len(missing)}):")
            for name in sorted(missing):
                print(" ", name)

    return pd.DataFrame(rows), missing_by_model



def plot_obj_viol_sweep_over_models(
    df: pd.DataFrame,
    *,
    method_col="model",
    xcol="mu",
    obj_col="obj",
    viol_col="viol",
    colormap="husl",
    outfilename=None,
    xlog=True,
    margin=0.05,
    linewidth=3.0,
    alpha=0.85,
    title="",
    legend_gap=0.12,
    right_space=0.72,
    figsize=(8.0, 4.2),
    obj_shift="none",     # "none" | "min"
    obj_ylog=False,
    obj_floor=None,
    # --- NEW ---
    obj_ylim=None,        # None or (ymin, ymax) in *plotted* objective coords
):
    for c in [method_col, xcol, obj_col, viol_col]:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    # floor for log(viol): keep lines even if viol hits 0
    v = pd.to_numeric(df[viol_col], errors="coerce").to_numpy()
    v = v[np.isfinite(v) & (v > 0)]
    viol_floor = float(v.min() * 0.5) if v.size else 1e-12
    viol_floor = max(viol_floor, 1e-300)

    if obj_shift not in ("none", "min"):
        raise ValueError(f"obj_shift must be 'none' or 'min', got {obj_shift!r}")

    # precompute objective shift (global min over plotted points)
    df_obj = df[[xcol, obj_col]].copy()
    df_obj[xcol] = pd.to_numeric(df_obj[xcol], errors="coerce")
    df_obj[obj_col] = pd.to_numeric(df_obj[obj_col], errors="coerce")
    df_obj = df_obj.dropna(subset=[xcol, obj_col])
    if xlog:
        df_obj = df_obj[df_obj[xcol] > 0]

    obj_shift_val = 0.0
    if obj_shift == "min" and not df_obj.empty:
        obj_shift_val = float(df_obj[obj_col].to_numpy().min())

    # floor for log(obj) after shift
    obj_floor_val = None
    if obj_ylog:
        vals = df_obj[obj_col].to_numpy()
        vals = vals[np.isfinite(vals)]
        vals = vals - obj_shift_val
        vals = vals[vals > 0]
        if obj_floor is not None:
            obj_floor_val = float(obj_floor)
        else:
            obj_floor_val = float(vals.min() * 0.5) if vals.size else 1e-12
        obj_floor_val = max(obj_floor_val, 1e-300)

    methods = df[method_col].dropna().unique()
    palette = sns.color_palette(colormap, len(methods))
    color_of = {m: palette[i] for i, m in enumerate(methods)}

    fig, ax_obj = plt.subplots(figsize=figsize)
    ax_viol = ax_obj.twinx()

    obj_vals_for_ylim = []
    viol_vals_plot = []

    for method, g in df.groupby(method_col, sort=False):
        g = g[[xcol, obj_col, viol_col]].copy()
        for c in [xcol, obj_col, viol_col]:
            g[c] = pd.to_numeric(g[c], errors="coerce")
        g = g.dropna(subset=[xcol]).sort_values(xcol)

        if xlog:
            g = g[g[xcol] > 0]
        if g.empty:
            continue

        col = color_of.get(method, "#000000")

        go = g.dropna(subset=[obj_col])
        if not go.empty:
            xs = go[xcol].to_numpy()
            ys = go[obj_col].to_numpy()

            if obj_shift == "min":
                ys = ys - obj_shift_val

            if obj_ylog:
                ys_plot = np.maximum(ys, obj_floor_val)
                obj_vals_for_ylim.append(ys_plot)
                ax_obj.plot(xs, ys_plot, color=col, lw=linewidth, alpha=alpha, ls="-", label=str(method))
            else:
                obj_vals_for_ylim.append(ys)
                ax_obj.plot(xs, ys, color=col, lw=linewidth, alpha=alpha, ls="-", label=str(method))

        gv = g.dropna(subset=[viol_col])
        if not gv.empty:
            xs = gv[xcol].to_numpy()
            ys = gv[viol_col].to_numpy()
            ys_plot = np.maximum(ys, viol_floor)
            viol_vals_plot.append(ys_plot)
            ax_viol.plot(xs, ys_plot, color=col, lw=linewidth, alpha=alpha * 0.9, ls="--", label=str(method))

    if xlog:
        ax_obj.set_xscale("log")
        ax_obj.xaxis.set_major_locator(LogLocator(base=10.0))
        ax_obj.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    if obj_ylog:
        ax_obj.set_yscale("log")
        ax_obj.yaxis.set_major_locator(LogLocator(base=10.0))
        ax_obj.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    else:
        sf = ScalarFormatter(useMathText=True)
        sf.set_scientific(True)
        sf.set_powerlimits((-3, 3))
        ax_obj.yaxis.set_major_formatter(sf)

    ax_viol.set_yscale("log")
    ax_viol.yaxis.set_major_locator(LogLocator(base=10.0))
    ax_viol.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax_obj.set_xlabel({"lr": "Learning Rate", "mu": r"$\mu$"}.get(xcol, xcol))
    ax_obj.set_ylabel(
        r"Obj (shifted):  $\mathrm{tr}(G^\top Z) - \min$" if obj_shift == "min"
        else r"Obj:  $\mathrm{tr}(G^\top Z)$"
    )
    ax_viol.set_ylabel(r"Viol:  $\|(\mathcal{A}(Z)-\beta)_+\|_\infty/\beta$")
    ax_obj.grid(axis="both", lw=0.4, ls="--", zorder=0)

    # --- objective y-limits ---
    if obj_ylim is not None:
        y0, y1 = obj_ylim
        if (y0 is not None) and (y1 is not None) and (y1 <= y0):
            raise ValueError(f"obj_ylim must satisfy ymax>ymin, got {obj_ylim}")
        if obj_ylog:
            if (y0 is not None and y0 <= 0) or (y1 is not None and y1 <= 0):
                raise ValueError(f"obj_ylim must be >0 when obj_ylog=True, got {obj_ylim}")
        ax_obj.set_ylim(y0, y1)
    else:
        if obj_vals_for_ylim:
            y = np.concatenate(obj_vals_for_ylim)
            y = y[np.isfinite(y)]
            if obj_ylog:
                y = y[y > 0]
                if y.size:
                    lo, hi = float(y.min()), float(y.max())
                    ax_obj.set_ylim(lo / (1.0 + margin), hi * (1.0 + margin))
            else:
                if y.size:
                    lo, hi = float(y.min()), float(y.max())
                    rng = hi - lo
                    pad = (rng * margin) if rng > 0 else (max(abs(lo), abs(hi), 1.0) * margin)
                    ax_obj.set_ylim(lo - pad, hi + pad)

    # viol y-limits (auto)
    if viol_vals_plot:
        y = np.concatenate(viol_vals_plot)
        y = y[np.isfinite(y) & (y > 0)]
        if y.size:
            lo, hi = float(y.min()), float(y.max())
            ax_viol.set_ylim(lo / (1.0 + margin), hi * (1.0 + margin))

    x_anchor = 1.0 + float(legend_gap)
    h1, l1 = ax_obj.get_legend_handles_labels()
    h2, l2 = ax_viol.get_legend_handles_labels()

    if h1:
        leg1 = ax_obj.legend(h1, l1, title=obj_col, loc="upper left",
                             bbox_to_anchor=(x_anchor, 1.0), frameon=False)
        ax_obj.add_artist(leg1)
    if h2:
        ax_obj.legend(h2, l2, title=viol_col, loc="lower left",
                      bbox_to_anchor=(x_anchor, 0.0), frameon=False)

    if title:
        ax_obj.set_title(title)

    fig.subplots_adjust(top=0.92, bottom=0.15, left=0.12, right=right_space)

    if outfilename:
        fig.savefig(outfilename, format="pdf", bbox_inches="tight")

    return fig, (ax_obj, ax_viol)



def plot_lr_sweep_over_models(
    df,
    colormap="husl",
    outfilename=None,
    ycol="min_val_loss",
    xcol="lr",
    ylog=False,
    ymargin=0.05,
    linewidth=3.0,
    alpha=0.8,
    title=""
):
    """
    Plot a LR sweep from a DataFrame.

    Expects columns:
      - df["model"] (line grouping)
      - df["lr"] (x axis)
      - df[ycol] (y axis)
    """ 
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
        "fin_val_loss": "Final Validation Loss",
        "final_train_loss": "Final Training Loss",
        "kq_max": "Maximum KQ Value",
        "kq_median": "Median KQ Value",
        "kq_mean": "Mean KQ Value",
        "r_true_res": "True PD Residual",
        "r_res": "Private PD Residual",
    }

    xlabel = {"lr": "Learning Rate",
              "mu": r"$\mu$"}[xcol]

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
            linewidth=linewidth,
            alpha=alpha
        )

    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel.get(ycol, ycol))

    # legend outside (right)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax.grid(axis="both", lw=0.4, ls="--", zorder=0)
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.78)

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
                pad = max(float(ymargin), 0.0)
                ax.set_ylim(ymin / (10 ** pad), ymax * (10 ** pad))
            else:
                pad = (ymax - ymin) * float(ymargin)
                if pad == 0.0:
                    pad = abs(ymax) * float(ymargin) if ymax != 0 else 1e-6
                ax.set_ylim(ymin - pad, ymax + pad)
    if title:
        ax.set_title(title)

    if outfilename:
        fig.savefig(outfilename, format="pdf", bbox_inches="tight")
    return fig, ax


def plot_heatmat_grid(
    df_in: pd.DataFrame,
    cmap="viridis",
    value="min_val_loss",
    annotate=True,
    sci_digits=1,      # digits after decimal in scientific notation inside cells
    flag_best=True,
    tick_fmt="g",      # axis tick format for mu_frac / rho_over_lr (e.g. "g", ".3e")
    dpi=120,
) -> None:
    """Seaborn heatmap over (mu_frac, rho_over_lr), sorted axes, optional numbers + best-cell flag."""
    label = {
        "mu_frac": r"$\mu_{\text{frac}}$",
        "rho_over_lr": r"$\rho/\gamma$",
        "min_val_loss": "Minimum Validation Loss",
        "fin_val_loss": "Final Validation Loss",
        "final_train_loss": "Final Training Loss",
        "kq_max": "Maximum KQ Value",
        "kq_median": "Median KQ Value",
        "kq_mean": "Mean KQ Value",
    }

    req = ["mu_frac", "rho_over_lr", value]
    missing = [c for c in req if c not in df_in.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = df_in.copy()
    df["mu_frac"] = pd.to_numeric(df["mu_frac"], errors="coerce")
    df["rho_over_lr"] = pd.to_numeric(df["rho_over_lr"], errors="coerce")
    df[value] = pd.to_numeric(df[value], errors="coerce")
    df = df.dropna(subset=req)
    if df.empty:
        print("No rows with required numeric columns.")
        return

    pivot = (
        df.pivot_table(index="rho_over_lr", columns="mu_frac", values=value, aggfunc="min")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        print("Not enough variation in mu_frac/rho_over_lr for a heatmap.")
        return

    annot_data, best_rc = None, None
    if annotate:
        fmt = f"{{:.{int(sci_digits)}e}}"
        annot_data = pivot.applymap(lambda v: "" if pd.isna(v) else fmt.format(float(v)))
        if flag_best:
            s = pivot.stack(dropna=True)
            if not s.empty:
                best_rc = s.idxmin()  # (rho_over_lr, mu_frac)
                annot_data.loc[best_rc[0], best_rc[1]] = "*" + fmt.format(float(pivot.loc[best_rc[0], best_rc[1]]))

    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        cbar_kws={"label": label.get(value, value)},
        annot=annot_data if annotate else False,
        fmt="",
        annot_kws={"fontsize": 6},
        linewidths=0.025,          
        linecolor="white", 
    )

    ax.invert_yaxis()
    ax.set_xlabel(label["mu_frac"])
    ax.set_ylabel(label["rho_over_lr"])
    ax.set_title(f"{label.get(value, value)} over ({label['mu_frac']}, {label['rho_over_lr']})")

    # exact mu_frac / rho_over_lr values on axes
    ax.set_xticklabels([format(float(x), tick_fmt) for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels([format(float(y), tick_fmt) for y in pivot.index], rotation=0)

    if flag_best and best_rc is not None:
        import matplotlib.patches as patches

        r, c = best_rc
        row = list(pivot.index).index(r)
        col = list(pivot.columns).index(c)
        ax.add_patch(patches.Rectangle((col, row), 1, 1, fill=False, edgecolor="red", linewidth=2.0))

    fig.tight_layout() 