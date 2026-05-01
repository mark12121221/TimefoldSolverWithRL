import argparse
import subprocess
import sys
import os

# Preset experiment configurations

PRESETS = {
    'easy_small': dict(mode='easy', num_employees=5,  num_tasks=8,  num_instances=100),
    'easy_medium': dict(mode='easy', num_employees=8,  num_tasks=12, num_instances=100),
    'hard_small':  dict(mode='hard', num_employees=5,  num_tasks=8,  num_instances=100),
    'hard_medium': dict(mode='hard', num_employees=8,  num_tasks=12, num_instances=50),
}


def run_preset(name: str, preset: dict, shared_args: argparse.Namespace):
    """Run compare_methods.py for one preset configuration."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {name}")
    print("=" * 60)

    cmd = [
        sys.executable, 'compare_methods.py',
        '--mode',          preset['mode'],
        '--num-employees', str(preset['num_employees']),
        '--num-tasks',     str(preset['num_tasks']),
        '--num-instances', str(preset['num_instances']),
        '--run-tag',       name,
        '--seed',          str(shared_args.seed),
    ]

    if shared_args.rl_model:
        cmd += ['--rl-model', shared_args.rl_model]
    if shared_args.ml_model:
        cmd += ['--ml-model', shared_args.ml_model]
    if shared_args.jar_path:
        cmd += ['--jar-path', shared_args.jar_path]
    if shared_args.ml_threshold:
        cmd += ['--ml-threshold', str(shared_args.ml_threshold)]
    if shared_args.rl_filter_rule:
        cmd += ['--rl-filter-rule', shared_args.rl_filter_rule]
    if shared_args.rl_assign_threshold:
        cmd += ['--rl-assign-threshold', str(shared_args.rl_assign_threshold)]
    if shared_args.rl_reward_threshold is not None:
        cmd += ['--rl-reward-threshold', str(shared_args.rl_reward_threshold)]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: experiment '{name}' exited with code {result.returncode}")
    return result.returncode == 0


def main(args):
    to_run = args.only if args.only else list(PRESETS.keys())

    # Validate names
    unknown = [n for n in to_run if n not in PRESETS]
    if unknown:
        print(f"Unknown experiment names: {unknown}")
        print(f"Available: {list(PRESETS.keys())}")
        sys.exit(1)

    print(f"Running {len(to_run)} experiment(s): {to_run}")
    print(f"RL model:   {args.rl_model or '(not provided)'}")
    print(f"ML model:   {args.ml_model or '(not provided)'}")
    print(f"JAR path:   {args.jar_path or '(not provided — solver methods skipped)'}")

    passed = []
    failed = []
    for name in to_run:
        ok = run_preset(name, PRESETS[name], args)
        (passed if ok else failed).append(name)

    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Completed: {passed}")
    if failed:
        print(f"Failed:    {failed}")
    print(f"\nAll results in: results/")

    if len(passed) > 1:
        print(f"\nTo plot all results:")
        for name in passed:
            print(f"  python plot_comparison_results.py "
                  f"--results results/{name}/comparison_results.json --title \"{name}\"")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch runner for compare_methods.py presets'
    )
    parser.add_argument('--only',         nargs='+', default=[],
                        help='Run only these presets (default: all four)')
    parser.add_argument('--rl-model',     type=str, default='rl_artifacts/policy.pt',
                        help='Path to trained RL model')
    parser.add_argument('--ml-model',     type=str,
                        default='ml_artifacts/random_forest_feasibility_model.joblib',
                        help='Path to trained ML model')
    parser.add_argument('--jar-path',     type=str, default='',
                        help='Path to Timefold solver JAR')
    parser.add_argument('--ml-threshold',        type=float, default=0.7,
                        help='ML filtering threshold (default: 0.7)')
    parser.add_argument('--rl-filter-rule',      choices=['feasible', 'threshold', 'reward'],
                        default='feasible',
                        help='RL filter rule (default: feasible)')
    parser.add_argument('--rl-assign-threshold', type=float, default=0.75,
                        help='RL assign fraction threshold (default: 0.75)')
    parser.add_argument('--rl-reward-threshold', type=float, default=0.0,
                        help='RL reward threshold (default: 0.0)')
    parser.add_argument('--seed',                type=int, default=42,
                        help='Random seed (default: 42)')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
