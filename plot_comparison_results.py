import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Consistent colour palette (colour-blind-friendly)
METHOD_COLORS = {
    'greedy':           '#2196F3',   # blue
    'random':           '#FF9800',   # orange
    'rl_policy':        '#4CAF50',   # green
    'solver':           '#9C27B0',   # purple
    'ml_filter_solver': '#F44336',   # red
    'rl_filter_solver': '#00BCD4',   # teal
}
METHOD_LABELS = {
    'greedy':           'Greedy',
    'random':           'Random',
    'rl_policy':        'RL Policy',
    'solver':           'Solver',
    'ml_filter_solver': 'ML+Solver',
    'rl_filter_solver': 'RL Filter+Solver',
}


def load_results(json_path: str) -> tuple[dict, dict]:
    #Return (config, results) from a comparison JSON file.
    with open(json_path) as f:
        data = json.load(f)
    return data.get('config', {}), data.get('results', {})


def method_color(name: str) -> str:
    return METHOD_COLORS.get(name, '#607D8B')


def method_label(name: str) -> str:
    return METHOD_LABELS.get(name, name)


def save_fig(fig, path: str, show: bool):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# Single-experiment plots
def plot_single(results: dict, config: dict, out_dir: str, title_prefix: str, show: bool):
    methods  = list(results.keys())
    colors   = [method_color(m) for m in methods]
    labels   = [method_label(m) for m in methods]
    x        = np.arange(len(methods))
    n_tasks  = config.get('num_tasks', '?')
    mode     = config.get('mode', '?')
    n_inst   = config.get('num_instances', '?')
    subtitle = f"mode={mode}  employees={config.get('num_employees','?')}  tasks={n_tasks}  n={n_inst}"

    # Plot 1: Feasible rate
    feasible = [results[m].get('feasible_rate', 0) for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, feasible, color=colors, edgecolor='black', linewidth=0.8, width=0.55)
    ax.bar_label(bars, fmt='%.1%%', padding=3, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Feasible solution rate')
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_title(f"{title_prefix}Feasible Rate\n{subtitle}", fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    save_fig(fig, os.path.join(out_dir, 'plot_cmp_feasible_rate.png'), show)

    # Plot 2: Assigned / unassigned tasks (grouped)
    assigned   = [results[m].get('avg_assigned', 0)   for m in methods]
    unassigned = [results[m].get('avg_unassigned', 0) for m in methods]
    std_assigned = [results[m].get('std_assigned', 0) for m in methods]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.38
    bars1 = ax.bar(x - width/2, assigned,   width, label='Assigned',   color=colors,
                   alpha=0.9, edgecolor='black', linewidth=0.8,
                   yerr=std_assigned, capsize=4)
    bars2 = ax.bar(x + width/2, unassigned, width, label='Unassigned', color=colors,
                   alpha=0.4, edgecolor='black', linewidth=0.8, hatch='//')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(f'Avg tasks (/ {n_tasks})')
    ax.set_title(f"{title_prefix}Assigned vs Unassigned Tasks\n{subtitle}", fontsize=11)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    save_fig(fig, os.path.join(out_dir, 'plot_cmp_tasks.png'), show)

    # Plot 3: Runtime
    runtimes = [results[m].get('avg_runtime_ms', 0) for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, runtimes, color=colors, edgecolor='black', linewidth=0.8, width=0.55)
    ax.bar_label(bars, fmt='%.1f ms', padding=3, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Avg runtime (ms)')
    ax.set_title(f"{title_prefix}Average Runtime\n{subtitle}", fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    # Log scale if solver present (can be 1000x slower than constructive methods)
    if max(runtimes) > 0 and max(runtimes) / (min(r for r in runtimes if r > 0) or 1) > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Avg runtime (ms, log scale)')
    save_fig(fig, os.path.join(out_dir, 'plot_cmp_runtime.png'), show)

    # Plot 4: ML filtering stats (if available)
    if 'ml_filter_solver' in results:
        ml = results['ml_filter_solver']
        labels_ml = ['Total\ninstances', 'Accepted\n(to solver)', 'Rejected\n(filtered out)']
        values_ml = [
            ml.get('total_instances', 0),
            ml.get('accepted_instances', 0),
            ml.get('rejected_instances', 0),
        ]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(labels_ml, values_ml, color=['#607D8B', '#4CAF50', '#F44336'],
               edgecolor='black', linewidth=0.8, width=0.5)
        ax.set_ylabel('Number of instances')
        ar = ml.get('acceptance_rate', 0)
        ax.set_title(
            f"{title_prefix}ML Filtering Pipeline\n"
            f"Acceptance rate: {ar:.1%} | Solver calls saved: {ml.get('solver_calls_saved',0)}",
            fontsize=11
        )
        ax.grid(axis='y', alpha=0.3)
        save_fig(fig, os.path.join(out_dir, 'plot_cmp_ml_filter.png'), show)


# Two-experiment grouped comparison (e.g. easy vs hard)
def plot_grouped(
    results_a: dict, label_a: str,
    results_b: dict, label_b: str,
    out_dir: str, title_prefix: str, show: bool
):
    # Grouped bar chart: two experiment settings side by side per method.
    # Use only methods present in both
    methods = [m for m in results_a if m in results_b]
    if not methods:
        print("  No common methods to compare — skipping grouped plot.")
        return

    labels  = [method_label(m) for m in methods]
    x       = np.arange(len(methods))
    width   = 0.38

    metrics = [
        ('feasible_rate',  'Feasible Rate',          True),
        ('avg_assigned',   'Avg Assigned Tasks',      False),
        ('avg_unassigned', 'Avg Unassigned Tasks',    False),
    ]

    for key, ylabel, pct in metrics:
        vals_a = [results_a[m].get(key, 0) for m in methods]
        vals_b = [results_b[m].get(key, 0) for m in methods]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width/2, vals_a, width, label=label_a,
               color='steelblue', edgecolor='black', linewidth=0.8, alpha=0.85)
        ax.bar(x + width/2, vals_b, width, label=label_b,
               color='tomato',    edgecolor='black', linewidth=0.8, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel + (' (%)' if pct else ''))
        if pct:
            ax.set_ylim(0, 1.15)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.set_title(f"{title_prefix}{label_a} vs {label_b} — {ylabel}", fontsize=11)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        fname = f"plot_grouped_{key}.png"
        save_fig(fig, os.path.join(out_dir, fname), show)


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot comparison results from compare_methods.py output'
    )
    parser.add_argument('--results',  type=str, required=True,
                        help='Path to comparison_results.json (primary)')
    parser.add_argument('--compare',  type=str, default='',
                        help='Second comparison_results.json for grouped easy vs hard plot')
    parser.add_argument('--labels',   nargs=2, default=['A', 'B'],
                        help='Labels for --results and --compare (default: A B)')
    parser.add_argument('--title',    type=str, default='',
                        help='Optional prefix for plot titles')
    parser.add_argument('--show',     action='store_true', default=False,
                        help='Display plots interactively')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.show:
        matplotlib.use('TkAgg')

    config_a, results_a = load_results(args.results)
    out_dir = os.path.dirname(args.results) or '.'
    title   = (args.title + ' — ') if args.title else ''

    print(f"\nPlotting: {args.results}")
    plot_single(results_a, config_a, out_dir, title, args.show)

    if args.compare and os.path.exists(args.compare):
        _, results_b = load_results(args.compare)
        out_dir_grouped = out_dir  # save grouped plots alongside primary
        print(f"\nGrouped comparison: {args.labels[0]} vs {args.labels[1]}")
        plot_grouped(results_a, args.labels[0], results_b, args.labels[1],
                     out_dir_grouped, title, args.show)

    print("\nDone.")
