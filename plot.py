import argparse
import matplotlib.pyplot as plt
import copy
import numpy as np
import matplotlib as mpl
from pathlib import Path
from gptopt.plot_utils import (
    load_outputs,
    method_name,
    get_alpha_from_lr,
    plot_data,
    plot_step_size_and_lr,
    smoothen_dict,
)

# Central style configuration
def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 2.0,
        "lines.linewidth": 4.8,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    plt.rcParams["text.usetex"] = False

apply_style()

def plot_final_loss_vs_lr(outputs, colormap, linestylemap, outfilename, figdir, val=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = {}
    field = 'val_losses' if val else 'losses'
    for out in outputs:
        name = method_name(out)
        lr = out['learning_rate']
        series = out.get(field)
        if not series:
            continue
        final_loss = series[-1]
        methods.setdefault(name, {'lrs': [], 'losses': []})
        methods[name]['lrs'].append(lr)
        methods[name]['losses'].append(final_loss)
        if val:
            print(name, "-lr-", lr, "-loss-", final_loss)

    # Optional teacher overlay (use average teacher losses if present)
    for out in outputs:
        if 'teach_losses' in out and 'teacher' not in methods:
            tmean = float(np.mean(out['teach_losses']))
            some_method = next(iter(methods)) if methods else None
            methods['teacher'] = {'lrs': (methods[some_method]['lrs'] if some_method else []),
                                  'losses': [tmean] * (len(methods[some_method]['lrs']) if some_method else 0)}

    lower_bound = 100.0
    upper_bound = 0.0
    for name, data in methods.items():
        if not data['lrs']:
            continue
        sorted_indices = sorted(range(len(data['lrs'])), key=lambda i: data['lrs'][i])
        sorted_lrs = [data['lrs'][i] for i in sorted_indices]
        sorted_losses = [data['losses'][i] for i in sorted_indices]
        ax.plot(sorted_lrs, sorted_losses, label=name,
                color=colormap.get(name, '#000000'),
                linestyle=linestylemap.get(name, None),
                linewidth=3.0)
        current_ub = np.max(sorted_losses)
        current_lb = np.min(sorted_losses)
        upper_bound = max(upper_bound, current_ub)
        lower_bound = min(lower_bound, current_lb)
    if upper_bound == 0.0:
        upper_bound = 10.0
    else:
        upper_bound = min(upper_bound * 1.1, 10.0)
    lower_bound = (lower_bound * 0.95) if lower_bound < 100.0 else 3.0
    ax.set_xscale('log')
    ax.set_ylim([lower_bound, upper_bound])
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Final Validation Loss' if val else 'Final Loss')
    plotfile = figdir / ('lr-sens-val.pdf' if val else 'lr-sens.pdf')
    ax.legend(loc='upper right')
    ax.grid(axis='both', lw=0.4, ls='--', zorder=0)
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
    fig.savefig(plotfile, format='pdf', bbox_inches='tight')

def plot_tuned_curves(outputs, colormap, linestylemap, outfilename, num_epochs, figdir, wallclock=False, val=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    tuned_methods = {}
    field = 'val_losses' if val else 'losses'
    for out in outputs:
        name = method_name(out)
        series = out.get(field)
        if not series:
            continue
        final = float(series[-1])
        if (name not in tuned_methods) or (final < tuned_methods[name]['best_loss']) or np.isnan(tuned_methods[name]['best_loss']):
            tuned_methods[name] = {'best_loss': final, 'best_lr': out['learning_rate'], 'outputs': dict(out)}
    print("Best Validation losses:" if val else "Best losses:")
    for name in tuned_methods:
        print(f"{name}: {tuned_methods[name]['best_loss']} at lr {tuned_methods[name]['best_lr']}")

    tuned_outputs = [tuned_methods[name]['outputs'] for name in tuned_methods]
    lr_ranges = {name: [tuned_methods[name]['best_lr']] * 2 for name in tuned_methods}
    plot_data(ax, tuned_outputs, num_epochs, field, 'Loss', colormap, linestylemap,
              lr_ranges, get_alpha_from_lr, wallclock=wallclock)
    if tuned_outputs:
        try:
            upper_bound = np.max([out[field][round(0.2 * len(out[field]))] for out in tuned_outputs if field in out and out[field]])
        except Exception:
            upper_bound = 10.0
    else:
        upper_bound = 10.0
    lower_bound = 100.0
    for out in tuned_outputs:
        if field in out and out[field]:
            lower_bound = float(min(lower_bound, np.min(out[field])))
    upper_bound = min(upper_bound, 10.0) if not np.isnan(upper_bound) else 10.0
    lower_bound = max(lower_bound, 3.0) if not np.isnan(lower_bound) else 3.0
    lower_bound *= 0.95
    ax.legend(loc='upper right')
    ax.set_ylim(lower_bound, upper_bound)
    fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    suffix = "_tuned"
    if wallclock:
        suffix += "_wallclock"
    if val:
        suffix += "_val"
    fig.savefig(figdir / (suffix.lstrip('_') + '.pdf'), format='pdf', bbox_inches='tight')

def main():
    # Example:
    # python plot.py gpt-tiny tiny_shakespeare
    parser = argparse.ArgumentParser(description='Plot outputs from outputs/<model>/<data>.')
    parser.add_argument('model', type=str)
    parser.add_argument('data', type=str)
    args = parser.parse_args()

    model_name, data_name = args.model, args.data
    outfilename = f"{model_name}-{data_name}"
    output_dir = f"outputs/{model_name}/{data_name}"
    figdir = Path("figures") / model_name / data_name
    figdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading outputs from {output_dir}")
    outputs = load_outputs(output_dir)
    print(f"Loaded {len(outputs)} outputs")

    for out in outputs:
        smoothen_dict(out, beta=0.05)

    colormap = {
        'sgd-m': '#B3CBB9',
        'adamw': '#FF6B35',
        'iams': '#61ACE5',
        'muon-momo': '#1B75BC',
        'scion': 'k',
        'muonmax-momo': '#FF00FF',
        'momo-adam': '#8B008B',
        'muon': '#008000',
        'adamw-schedulefree': '#006400',
    }
    linestylemap = {
        'sgd-m': None, 'iams': None, 'muon': None, 'muon-momo': None,
        'muonmax-momo': None, 'momo-adam': None, 'adamw': '--',
        'adamw-schedulefree': '--', 'scion': '--',
    }

    # Build lr ranges per method
    lr_ranges = {}
    for out in outputs:
        name = method_name(out)
        lr = out['learning_rate']
        if name not in lr_ranges:
            lr_ranges[name] = [lr, lr]
        else:
            lr_ranges[name][0] = min(lr_ranges[name][0], lr)
            lr_ranges[name][1] = max(lr_ranges[name][1], lr)

    mpl.rcParams.update(mpl.rcParamsDefault)
    apply_style()

    plot_final_loss_vs_lr(outputs, colormap, linestylemap, outfilename, figdir)
    plot_final_loss_vs_lr(outputs, colormap, linestylemap, outfilename, figdir, val=True)

    num_epochs = max((len(o.get('losses', [])) for o in outputs), default=1)
    initial_loss = next((o['losses'][0] for o in outputs if o.get('losses')), 1.0)
    upper_bound = initial_loss * 1.2

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    plot_data(ax, outputs, num_epochs, 'losses', 'Loss',
              colormap, linestylemap, lr_ranges, get_alpha_from_lr)
    lower_bound = min((min(o['losses']) for o in outputs if o.get('losses')), default=1.0) * 0.95
    ax.set_ylim(lower_bound, upper_bound)
    ax.legend(loc='upper right')
    fig.savefig(figdir / 'loss.pdf', format='pdf', bbox_inches='tight')

    # Example subset plots can be adjusted or removed; preserving structure:
    for subset in [['sgd-m', 'iams', 'muon-momo'], ['muon', 'momo-adam', 'adamw']]:
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        sub_outputs = [o for o in outputs if method_name(o) in subset]
        plot_data(ax, sub_outputs, num_epochs, 'learning_rates', 'Learning rate',
                  colormap, linestylemap, lr_ranges, get_alpha_from_lr)
        ax.legend(loc='upper right')
        suffix = 'lr' if subset[0] == 'sgd-m' else 'lr-adam'
        fig.savefig(figdir / (suffix + '.pdf'), format='pdf', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    plotted = plot_step_size_and_lr(ax, outputs, colormap, linestylemap, lr_ranges, get_alpha_from_lr)
    handles, labels = ax.get_legend_handles_labels()
    fixed_handles = [copy.copy(h) for h in handles]
    for h in fixed_handles:
        h.set_alpha(1.0)
    ax.legend(fixed_handles, labels, loc='upper right')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    fig.savefig(figdir / 'step_size.pdf', format='pdf', bbox_inches='tight')

    plot_tuned_curves(outputs, colormap, linestylemap, outfilename, num_epochs, figdir, val=False)
    plot_tuned_curves(outputs, colormap, linestylemap, outfilename, num_epochs, figdir, val=True)

if __name__ == "__main__":
    main()
