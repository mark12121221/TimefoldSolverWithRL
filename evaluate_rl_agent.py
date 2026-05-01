import argparse
import json
import os
import random
import time

import numpy as np
import torch

from generate_instances import InstanceGenerator
from rl_environment import TaskAssignmentEnv, UNASSIGNED_ACTION
from rl_model import load_model


# Reproducibility

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Baselines

def greedy_baseline(instance: dict) -> dict:

    employees = instance['employees']
    tasks = instance['tasks']
    capacities = {emp['id']: emp['availableCapacity'] for emp in employees}
    assignments = {}

    for task in tasks:
        eligible = [
            emp for emp in employees
            if task['requiredSkill'] in emp['skills']
            and capacities[emp['id']] >= task['duration']
        ]
        if eligible:
            best = max(eligible, key=lambda e: capacities[e['id']])
            assignments[task['id']] = best['id']
            capacities[best['id']] -= task['duration']
        else:
            assignments[task['id']] = None

    return _build_summary(tasks, assignments)


def random_baseline(instance: dict) -> dict:

    employees = instance['employees']
    tasks = instance['tasks']
    capacities = {emp['id']: emp['availableCapacity'] for emp in employees}
    assignments = {}

    for task in tasks:
        eligible = [
            emp for emp in employees
            if task['requiredSkill'] in emp['skills']
            and capacities[emp['id']] >= task['duration']
        ]
        if eligible:
            chosen = random.choice(eligible)
            assignments[task['id']] = chosen['id']
            capacities[chosen['id']] -= task['duration']
        else:
            assignments[task['id']] = None

    return _build_summary(tasks, assignments)


def _build_summary(tasks: list, assignments: dict) -> dict:
    assigned = sum(1 for v in assignments.values() if v is not None)
    mandatory = [t for t in tasks if t.get('mandatory', False)]
    mandatory_ok = all(assignments.get(t['id']) is not None for t in mandatory)
    feasible = mandatory_ok and (assigned == len(tasks) or
                                 all(assignments[t['id']] is not None
                                     for t in tasks if t.get('mandatory', False)))
    # Re-check: feasible means no unassigned tasks remain (only unassigned if no eligible)
    # For evaluation purposes: feasible = all mandatory tasks assigned
    feasible = mandatory_ok
    return {
        'assigned': assigned,
        'unassigned': len(tasks) - assigned,
        'feasible_solution': feasible,
    }

# RL evaluation

def run_rl_episode(
    env: TaskAssignmentEnv,
    policy,
    instance: dict,
    device: torch.device,
    greedy: bool,
) -> tuple[float, dict, float]:
    
    state_np = env.reset(instance)
    total_reward = 0.0
    done = False

    t_start = time.time()
    while not done:
        state_tensor = torch.FloatTensor(state_np).to(device)

        # Get valid-action mask from environment and pass to policy
        mask = torch.BoolTensor(env.get_valid_action_mask()).to(device)

        with torch.no_grad():
            action, _ = policy.get_action(state_tensor, action_mask=mask, greedy=greedy)

        state_np, reward, done, _ = env.step(action.item())
        total_reward += reward

    return total_reward, env.get_episode_summary(), time.time() - t_start


# Main

def evaluate(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from: {args.model_path}")
    print(f"Run tag: {args.run_tag or '(none)'} | Seed: {args.seed}")
    policy = load_model(args.model_path, device=str(device))
    policy.eval()

    env = TaskAssignmentEnv()
    generator = InstanceGenerator(mode=args.mode, seed=args.seed)

    rl_rewards:     list[float] = []
    rl_assigned:    list[int]   = []
    rl_unassigned:  list[int]   = []
    rl_feasible:    list[bool]  = []
    rl_runtimes:    list[float] = []

    gr_assigned:  list[int]  = []
    gr_unassigned: list[int] = []
    gr_feasible:  list[bool] = []

    rnd_assigned:  list[int]  = []
    rnd_unassigned: list[int] = []
    rnd_feasible:  list[bool] = []

    print(f"\nEvaluating {args.num_episodes} instances | seed={args.seed} | "
          f"mode={args.mode} | employees={args.num_employees} | tasks={args.num_tasks}")
    print(f"RL action selection: {'greedy (argmax)' if args.greedy else 'stochastic (sampled)'}")
    print("-" * 65)

    for i in range(args.num_episodes):
        instance = generator.generate_instance(
            num_employees=args.num_employees,
            num_tasks=args.num_tasks,
            mode=args.mode,
        )

        # RL policy (with action mask)
        reward, rl_sum, elapsed = run_rl_episode(env, policy, instance, device, args.greedy)
        rl_rewards.append(reward)
        rl_assigned.append(rl_sum['assigned'])
        rl_unassigned.append(rl_sum['unassigned'])
        rl_feasible.append(rl_sum['feasible_solution'])
        rl_runtimes.append(elapsed)

        # Greedy baseline
        gr_sum = greedy_baseline(instance)
        gr_assigned.append(gr_sum['assigned'])
        gr_unassigned.append(gr_sum['unassigned'])
        gr_feasible.append(gr_sum['feasible_solution'])

        # Random baseline
        rnd_sum = random_baseline(instance)
        rnd_assigned.append(rnd_sum['assigned'])
        rnd_unassigned.append(rnd_sum['unassigned'])
        rnd_feasible.append(rnd_sum['feasible_solution'])

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.num_episodes} episodes done ...")

    # Results
    n = args.num_tasks

    def fmt(vals, pct=False):
        m, s = np.mean(vals), np.std(vals)
        return f"{m:.1%}+/-{s:.1%}" if pct else f"{m:.2f}+/-{s:.2f}"

    print("\n" + "=" * 75)
    print("EVALUATION RESULTS")
    print("=" * 75)
    print(f"\n{'Metric':<38} {'RL Policy':>14} {'Greedy':>14} {'Random':>14}")
    print("-" * 75)
    print(f"{'Avg episode reward':<38} {np.mean(rl_rewards):>14.2f} {'n/a':>14} {'n/a':>14}")
    print(f"{'Avg assigned tasks  (/ ' + str(n) + ')':<38} "
          f"{fmt(rl_assigned):>14} {fmt(gr_assigned):>14} {fmt(rnd_assigned):>14}")
    print(f"{'Avg unassigned tasks':<38} "
          f"{fmt(rl_unassigned):>14} {fmt(gr_unassigned):>14} {fmt(rnd_unassigned):>14}")
    print(f"{'Feasible solution rate':<38} "
          f"{fmt(rl_feasible, pct=True):>14} "
          f"{fmt(gr_feasible, pct=True):>14} "
          f"{fmt(rnd_feasible, pct=True):>14}")
    print(f"{'Avg RL runtime (ms)':<38} {np.mean(rl_runtimes)*1000:>14.2f} {'n/a':>14} {'n/a':>14}")
    print("-" * 75)
    print(f"Values shown as mean+/-std over {args.num_episodes} episodes.")
    print("\nNotes:")
    print("  - Feasible = all mandatory tasks assigned (no skill/capacity violations).")
    print("  - Greedy: assign to eligible employee with max remaining capacity.")
    print("  - Random: assign to uniformly random eligible employee.")
    print("  - RL uses action masking at inference (valid moves only).")

    # Optional: show last episode solution
    if args.show_example:
        print("\n--- Example RL Solution (last episode) ---")
        for task_id, emp_id in env.assignments.items():
            print(f"  {task_id} -> {emp_id if emp_id else 'UNASSIGNED'}")

    # Save metrics to JSON
    results = {
        'config': vars(args),
        'rl': {
            'avg_reward':     float(np.mean(rl_rewards)),
            'std_reward':     float(np.std(rl_rewards)),
            'avg_assigned':   float(np.mean(rl_assigned)),
            'std_assigned':   float(np.std(rl_assigned)),
            'avg_unassigned': float(np.mean(rl_unassigned)),
            'feasible_rate':  float(np.mean(rl_feasible)),
            'avg_runtime_ms': float(np.mean(rl_runtimes) * 1000),
        },
        'greedy': {
            'avg_assigned':   float(np.mean(gr_assigned)),
            'std_assigned':   float(np.std(gr_assigned)),
            'avg_unassigned': float(np.mean(gr_unassigned)),
            'feasible_rate':  float(np.mean(gr_feasible)),
        },
        'random': {
            'avg_assigned':   float(np.mean(rnd_assigned)),
            'std_assigned':   float(np.std(rnd_assigned)),
            'avg_unassigned': float(np.mean(rnd_unassigned)),
            'feasible_rate':  float(np.mean(rnd_feasible)),
        },
    }
    out_dir = os.path.dirname(args.model_path) or 'rl_artifacts'
    fname = f"eval_results_{args.run_tag}.json" if args.run_tag else "eval_results.json"
    json_path = os.path.join(out_dir, fname)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to: {json_path}")
    print(f"To plot: python plot_training_results.py --eval {json_path}")


# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained RL agent for sequential task assignment'
    )
    parser.add_argument('--model-path',    type=str,   default='rl_artifacts/policy.pt',
                        help='Path to trained model checkpoint (default: rl_artifacts/policy.pt)')
    parser.add_argument('--num-episodes',  type=int,   default=200,
                        help='Number of evaluation episodes (default: 200)')
    parser.add_argument('--num-employees', type=int,   default=5,
                        help='Employees per instance (default: 5)')
    parser.add_argument('--num-tasks',     type=int,   default=8,
                        help='Tasks per instance (default: 8)')
    parser.add_argument('--mode',          choices=['easy', 'hard'], default='easy',
                        help='Instance difficulty mode (default: easy)')
    parser.add_argument('--greedy',        action='store_true', default=True,
                        help='Use greedy (argmax) action selection (default: True)')
    parser.add_argument('--stochastic',    dest='greedy', action='store_false',
                        help='Use stochastic (sampled) action selection')
    parser.add_argument('--seed',          type=int,   default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--show-example',  action='store_true', default=False,
                        help='Print the last episode solution')
    parser.add_argument('--run-tag',        type=str,   default='',
                        help='Optional tag appended to the JSON output filename')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
