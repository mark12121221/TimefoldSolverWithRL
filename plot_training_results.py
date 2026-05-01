import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')   # non-interactive backend; override with --show
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


# Helpers
def moving_average(values: list[float], window: int) -> np.ndarray:
    """Simple centred moving average with edge padding."""
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def save_fig(fig: plt.Figure, path: str, show: bool):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# Training history plots
def plot_training_history(csv_path: str, show: bool, window: int = 100):
    print(f"\nLoading training history: {csv_path}")

    # Load CSV — use pandas if available, otherwise plain csv
    if _PANDAS:
        import pandas as pd
        df = pd.read_csv(csv_path)
        episodes  = df['episode'].tolist()
        rewards   = df['reward'].tolist()
        assigned  = df['assigned'].tolist()
        feasibles = df['feasible'].tolist()
        baselines = df['baseline'].tolist()
    else:
        import csv
        episodes, rewards, assigned, feasibles, baselines = [], [], [], [], []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row['episode']))
                rewards.append(float(row['reward']))
                assigned.append(float(row['assigned']))
                feasibles.append(float(row['feasible']))
                baselines.append(float(row['baseline']))

    out_dir = os.path.dirname(csv_path) or '.'
    w = min(window, len(episodes) // 5 or 1)

    # Plot 1: Reward
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(episodes, rewards, color='steelblue', alpha=0.25, linewidth=0.8, label='Raw reward')
    ax.plot(episodes, moving_average(rewards, w), color='steelblue', linewidth=2,
            label=f'Moving avg (w={w})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total episode reward')
    ax.set_title('REINFORCE Training — Episode Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(out_dir, 'plot_reward.png'), show)

    # Plot 2: Feasible rate
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(episodes, moving_average(feasibles, w), color='seagreen', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Feasible rate (moving avg)')
    ax.set_title('REINFORCE Training — Feasible Solution Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(out_dir, 'plot_feasible_rate.png'), show)

    # Plot 3: Assigned tasks
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(episodes, moving_average(assigned, w), color='darkorange', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg assigned tasks (moving avg)')
    ax.set_title('REINFORCE Training — Assigned Tasks per Episode')
    ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(out_dir, 'plot_assigned_tasks.png'), show)

    # Plot 4: Baseline 
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(episodes, baselines, color='mediumpurple', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Baseline value b')
    ax.set_title('REINFORCE Baseline (EMA of Episode Return)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(out_dir, 'plot_baseline.png'), show)

    print(f"  Training plots saved to: {out_dir}/")


# Evaluation comparison plot
def plot_eval_comparison(json_path: str, show: bool):
    print(f"\nLoading evaluation results: {json_path}")
    with open(json_path) as f:
        results = json.load(f)

    rl     = results['rl']
    greedy = results['greedy']
    rand   = results['random']

    out_dir = os.path.dirname(json_path) or '.'

    # ---- Bar chart: assigned, unassigned, feasible rate ----
    methods = ['RL Policy', 'Greedy', 'Random']
    assigned_vals  = [rl['avg_assigned'],   greedy['avg_assigned'],   rand['avg_assigned']]
    unassigned_vals= [rl['avg_unassigned'],  greedy['avg_unassigned'], rand['avg_unassigned']]
    feasible_vals  = [rl['feasible_rate'],   greedy['feasible_rate'],  rand['feasible_rate']]

    # Std dev (only available for RL and greedy/random if stored)
    assigned_errs  = [rl.get('std_assigned', 0), greedy.get('std_assigned', 0), rand.get('std_assigned', 0)]

    x = np.arange(len(methods))
    width = 0.28
    colors = ['steelblue', 'seagreen', 'sandybrown']

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    # Subplot 1: assigned tasks
    axes[0].bar(x, assigned_vals, width=0.5, color=colors,
                yerr=assigned_errs, capsize=5, edgecolor='black', linewidth=0.7)
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods)
    axes[0].set_ylabel('Avg assigned tasks')
    axes[0].set_title('Assigned Tasks')
    axes[0].grid(axis='y', alpha=0.3)

    # Subplot 2: unassigned tasks
    axes[1].bar(x, unassigned_vals, width=0.5, color=colors,
                edgecolor='black', linewidth=0.7)
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods)
    axes[1].set_ylabel('Avg unassigned tasks')
    axes[1].set_title('Unassigned Tasks')
    axes[1].grid(axis='y', alpha=0.3)

    # Subplot 3: feasible rate
    axes[2].bar(x, feasible_vals, width=0.5, color=colors,
                edgecolor='black', linewidth=0.7)
    axes[2].set_xticks(x); axes[2].set_xticklabels(methods)
    axes[2].set_ylabel('Feasible rate')
    axes[2].set_title('Feasible Solution Rate')
    axes[2].set_ylim(0, 1.05)
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    axes[2].grid(axis='y', alpha=0.3)

    cfg = results.get('config', {})
    fig.suptitle(
        f"Method Comparison  |  mode={cfg.get('mode','?')}  "
        f"employees={cfg.get('num_employees','?')}  tasks={cfg.get('num_tasks','?')}  "
        f"n={cfg.get('num_episodes','?')} episodes",
        fontsize=11
    )

    save_fig(fig, os.path.join(out_dir, 'plot_comparison.png'), show)
    print(f"  Comparison plot saved to: {out_dir}/")


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot RL training history and/or evaluation comparison'
    )
    parser.add_argument('--history', type=str, default='',
                        help='Path to training_history.csv')
    parser.add_argument('--eval',    type=str, default='',
                        help='Path to eval_results.json')
    parser.add_argument('--window',  type=int, default=100,
                        help='Moving average window size for training plots (default: 100)')
    parser.add_argument('--show',    action='store_true', default=False,
                        help='Display plots interactively (requires a display)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.show:
        matplotlib.use('TkAgg')   # switch to interactive backend

    if not args.history and not args.eval:
        print("Provide at least one of --history or --eval. Use --help for usage.")
    else:
        if args.history:
            plot_training_history(args.history, show=args.show, window=args.window)
        if args.eval:
            plot_eval_comparison(args.eval, show=args.show)
