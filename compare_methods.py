import argparse
import csv
import json
import os
import random
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

from generate_instances import InstanceGenerator
from evaluate_rl_agent import greedy_baseline, random_baseline, run_rl_episode
from rl_environment import TaskAssignmentEnv


# Reproducibility

def set_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Solver method

def _java_cmd() -> str:
    java_home = os.environ.get('JAVA_HOME', '')
    if java_home:
        candidate = os.path.join(java_home, 'bin', 'java')
        if os.path.exists(candidate) or os.path.exists(candidate + '.exe'):
            return candidate
    return 'java'


def run_solver_on_instance(instance: dict, jar_path: str) -> dict | None:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(instance, f, indent=2)
        input_file = f.name
    output_file = input_file.replace('.json', '_result.json')

    try:
        t_start = time.time()
        result = subprocess.run(
            [_java_cmd(), '-jar', jar_path, input_file, output_file],
            capture_output=True, text=True, timeout=120
        )
        elapsed_ms = (time.time() - t_start) * 1000

        if result.returncode != 0:
            print(f"\n  [solver error] returncode={result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:300]}")
            return None

        if not os.path.exists(output_file):
            print(f"\n  [solver error] output file not created: {output_file}")
            return None

        with open(output_file, 'r', encoding='utf-8') as f:
            solver_result = json.load(f)

        return {
            'feasible':        bool(solver_result.get('feasible', False)),
            'assigned_tasks':  int(solver_result.get('assignedTasks', 0)),
            'unassigned_tasks': int(solver_result.get('unassignedTasks', len(instance['tasks']))),
            'runtime_ms':      int(solver_result.get('runtimeMs', elapsed_ms)),
        }
    except Exception as exc:
        print(f"\n  [solver error] exception: {exc}")
        return None
    finally:
        for p in (input_file, output_file):
            if os.path.exists(p):
                os.unlink(p)


# ML features (mirrors ml_filtering.py to avoid importing it directly,
# since ml_filtering.py requires a JAR path in its constructor)

def compute_ml_features(instance: dict) -> list[float]:
    #Compute the 10 instance-level features used by the ML feasibility model.
    employees = instance['employees']
    tasks = instance['tasks']

    total_workload  = sum(t['duration'] for t in tasks)
    total_capacity  = sum(e['availableCapacity'] for e in employees)
    capacity_ratio  = total_workload / total_capacity if total_capacity > 0 else 999.0

    num_skills = len({s for e in employees for s in e['skills']})

    cands = []
    for task in tasks:
        skill = task['requiredSkill']
        c = sum(1 for e in employees if skill in e['skills'])
        cands.append(c)

    n = len(cands)
    avg_cands   = sum(cands) / n if n else 0
    min_cands   = min(cands) if cands else 0
    frac_single = sum(1 for c in cands if c == 1) / n if n else 0
    frac_zero   = sum(1 for c in cands if c == 0) / n if n else 0

    return [
        len(employees), len(tasks), num_skills,
        total_workload, total_capacity, capacity_ratio,
        avg_cands, min_cands, frac_single, frac_zero,
    ]


# Per-method evaluation functions

def evaluate_constructive(instances: list[dict], method_fn) -> dict:
    
    assigned_list  = []
    unassigned_list = []
    feasible_list  = []
    runtime_list   = []

    for inst in instances:
        t = time.time()
        summary = method_fn(inst)
        runtime_list.append((time.time() - t) * 1000)
        assigned_list.append(summary['assigned'])
        unassigned_list.append(summary['unassigned'])
        feasible_list.append(float(summary['feasible_solution']))

    return _agg(assigned_list, unassigned_list, feasible_list, runtime_list)


def evaluate_rl(instances: list[dict], rl_model_path: str) -> dict:
    #Evaluate the trained RL policy on a list of instances.
    import torch
    from rl_model import load_model

    device = 'cpu'
    policy = load_model(rl_model_path, device=device)
    policy.eval()
    env = TaskAssignmentEnv()
    device_t = __import__('torch').device(device)

    assigned_list  = []
    unassigned_list = []
    feasible_list  = []
    runtime_list   = []
    reward_list    = []

    for inst in instances:
        reward, summary, elapsed = run_rl_episode(env, policy, inst, device_t, greedy=True)
        assigned_list.append(summary['assigned'])
        unassigned_list.append(summary['unassigned'])
        feasible_list.append(float(summary['feasible_solution']))
        runtime_list.append(elapsed * 1000)
        reward_list.append(reward)

    result = _agg(assigned_list, unassigned_list, feasible_list, runtime_list)
    result['avg_reward'] = float(np.mean(reward_list))
    result['std_reward'] = float(np.std(reward_list))
    return result


def evaluate_solver(instances: list[dict], jar_path: str) -> dict:
    # Run the Timefold solver on each instance and aggregate results.
    assigned_list  = []
    unassigned_list = []
    feasible_list  = []
    runtime_list   = []
    failed         = 0

    for i, inst in enumerate(instances):
        print(f"  Solver: {i+1}/{len(instances)} ...", end='\r')
        res = run_solver_on_instance(inst, jar_path)
        if res is None:
            failed += 1
            continue
        unassigned = res['unassigned_tasks']
        assigned   = res['assigned_tasks']
        assigned_list.append(assigned)
        unassigned_list.append(unassigned)
        feasible_list.append(float(res['feasible']))
        runtime_list.append(res['runtime_ms'])

    print()
    result = _agg(assigned_list, unassigned_list, feasible_list, runtime_list)
    result['solver_failures'] = failed
    return result


def evaluate_ml_filter_solver(
    instances: list[dict],
    ml_model_path: str,
    jar_path: str,
    threshold: float,
) -> dict:
    import joblib
    model = joblib.load(ml_model_path)

    accepted_instances = []
    rejected = 0

    feature_cols = [
        'num_employees', 'num_tasks', 'num_skills', 'total_required_workload',
        'total_available_capacity', 'capacity_ratio', 'avg_candidates_per_task',
        'min_candidates_per_task', 'fraction_single_candidate_tasks',
        'fraction_zero_candidate_tasks',
    ]

    for inst in instances:
        features = pd.DataFrame([compute_ml_features(inst)], columns=feature_cols)
        prob = model.predict_proba(features)[0, 1]
        if prob >= threshold:
            accepted_instances.append(inst)
        else:
            rejected += 1

    total = len(instances)
    accepted = len(accepted_instances)
    acceptance_rate = accepted / total if total > 0 else 0.0
    solver_calls_saved = rejected

    # Run solver only on accepted instances
    solver_results = evaluate_solver(accepted_instances, jar_path) if accepted_instances else {}

    result = solver_results.copy() if solver_results else {
        'avg_assigned': 0.0, 'std_assigned': 0.0,
        'avg_unassigned': 0.0, 'feasible_rate': 0.0,
        'avg_runtime_ms': 0.0,
    }
    result['acceptance_rate']    = acceptance_rate
    result['solver_calls_saved'] = solver_calls_saved
    result['total_instances']    = total
    result['accepted_instances'] = accepted
    result['rejected_instances'] = rejected
    return result


def evaluate_rl_filter_solver(
    instances: list[dict],
    rl_model_path: str,
    jar_path: str,
    filter_rule: str = 'feasible',
    assign_threshold_frac: float = 0.75,
    reward_threshold: float = 0.0,
) -> dict:

    import math
    import torch
    from rl_model import load_model

    device = 'cpu'
    policy = load_model(rl_model_path, device=device)
    policy.eval()
    env = TaskAssignmentEnv()
    device_t = __import__('torch').device(device)

    accepted_instances = []
    rejected = 0
    rl_assigned_before_list = []
    rl_reward_before_list = []

    for inst in instances:
        reward, summary, _ = run_rl_episode(env, policy, inst, device_t, greedy=True)
        rl_assigned_before_list.append(summary['assigned'])
        rl_reward_before_list.append(reward)

        num_tasks = len(inst['tasks'])
        if filter_rule == 'feasible':
            accept = summary['feasible_solution']
        elif filter_rule == 'threshold':
            min_assigned = math.ceil(assign_threshold_frac * num_tasks)
            accept = summary['assigned'] >= min_assigned
        elif filter_rule == 'reward':
            accept = reward >= reward_threshold
        else:
            raise ValueError(f"Unknown filter_rule: {filter_rule!r}")

        if accept:
            accepted_instances.append(inst)
        else:
            rejected += 1

    total = len(instances)
    accepted = len(accepted_instances)
    acceptance_rate = accepted / total if total > 0 else 0.0

    solver_results = evaluate_solver(accepted_instances, jar_path) if accepted_instances else {}
    result = solver_results.copy() if solver_results else {
        'avg_assigned': 0.0, 'std_assigned': 0.0,
        'avg_unassigned': 0.0, 'feasible_rate': 0.0,
        'avg_runtime_ms': 0.0,
    }
    result['acceptance_rate']               = acceptance_rate
    result['solver_calls_saved']            = rejected
    result['total_instances']               = total
    result['accepted_instances']            = accepted
    result['rejected_instances']            = rejected
    result['rl_filter_rule']                = filter_rule
    result['avg_rl_assigned_before_filter'] = (
        float(np.mean(rl_assigned_before_list)) if rl_assigned_before_list else 0.0
    )
    result['avg_rl_reward_before_filter']   = (
        float(np.mean(rl_reward_before_list)) if rl_reward_before_list else 0.0
    )
    return result


# Experimental: RL warm-start + solver (not in main pipeline)

def evaluate_rl_warmstart_solver(
    instances: list[dict],
    rl_model_path: str,
    jar_path: str,
) -> dict:
    import torch
    from rl_model import load_model

    device = 'cpu'
    policy = load_model(rl_model_path, device=device)
    policy.eval()
    env = TaskAssignmentEnv()
    device_t = __import__('torch').device(device)

    assigned_list = []
    unassigned_list = []
    feasible_list = []
    runtime_list = []
    rl_assigned_before_list = []
    failed = 0

    for i, inst in enumerate(instances):
        print(f"  RL+Solver: {i+1}/{len(instances)} ...", end='\r')

        # Step 1: run RL greedily — afterwards env.assignments holds the result
        run_rl_episode(env, policy, inst, device_t, greedy=True)
        rl_assignments = {
            tid: eid
            for tid, eid in env.assignments.items()
            if eid is not None
        }
        rl_assigned_before_list.append(len(rl_assignments))

        # Step 2: attach warm-start hints to a copy of the instance dict
        inst_warmstart = dict(inst)
        inst_warmstart['initialAssignments'] = rl_assignments

        # Step 3: run Timefold starting from the RL-produced solution
        res = run_solver_on_instance(inst_warmstart, jar_path)
        if res is None:
            failed += 1
            continue
        assigned_list.append(res['assigned_tasks'])
        unassigned_list.append(res['unassigned_tasks'])
        feasible_list.append(float(res['feasible']))
        runtime_list.append(res['runtime_ms'])

    print()
    result = _agg(assigned_list, unassigned_list, feasible_list, runtime_list)
    result['avg_rl_assigned_before_solver'] = (
        float(np.mean(rl_assigned_before_list)) if rl_assigned_before_list else 0.0
    )
    result['solver_failures'] = failed
    return result


# Aggregation helper

def _agg(assigned, unassigned, feasible, runtime) -> dict:
    if not assigned:
        return {
            'avg_assigned': 0.0, 'std_assigned': 0.0,
            'avg_unassigned': 0.0, 'feasible_rate': 0.0,
            'avg_runtime_ms': 0.0, 'n': 0,
        }
    return {
        'avg_assigned':   float(np.mean(assigned)),
        'std_assigned':   float(np.std(assigned)),
        'avg_unassigned': float(np.mean(unassigned)),
        'std_unassigned': float(np.std(unassigned)),
        'feasible_rate':  float(np.mean(feasible)),
        'avg_runtime_ms': float(np.mean(runtime)),
        'n':              len(assigned),
    }


# Output formatting

def print_table(results: dict[str, dict], num_tasks: int):
    #Print a structured comparison table to stdout.
    methods = list(results.keys())
    col_w = 18

    print("\n" + "=" * (28 + col_w * len(methods)))
    print("METHOD COMPARISON RESULTS")
    print("=" * (28 + col_w * len(methods)))

    header = f"{'Metric':<28}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(header)
    print("-" * (28 + col_w * len(methods)))

    def row(label, key, fmt='.2f', pct=False):
        vals = []
        for m in methods:
            v = results[m].get(key)
            if v is None:
                vals.append('n/a')
            elif pct:
                vals.append(f"{v:.1%}")
            else:
                vals.append(f"{v:{fmt}}")
        print(f"{label:<28}" + "".join(f"{v:>{col_w}}" for v in vals))

    row(f"Avg assigned (/ {num_tasks})", 'avg_assigned')
    row("Avg unassigned tasks", 'avg_unassigned')
    row("Feasible rate", 'feasible_rate', pct=True)
    row("Avg runtime (ms)", 'avg_runtime_ms', fmt='.1f')
    row("Avg reward", 'avg_reward', fmt='.2f')

    # Filter-method specific rows (ml_filter_solver and rl_filter_solver)
    if any('acceptance_rate' in results[m] for m in methods):
        row("Acceptance rate", 'acceptance_rate', pct=True)
        row("Solver calls saved", 'solver_calls_saved', fmt='.0f')

    # RL filter pre-filter diagnostics
    if any('avg_rl_assigned_before_filter' in results[m] for m in methods):
        row("RL assigned (pre-filter)", 'avg_rl_assigned_before_filter')
        row("RL reward (pre-filter)", 'avg_rl_reward_before_filter', fmt='.1f')

    print("-" * (28 + col_w * len(methods)))
    print("\nFeasibility note:")
    print("  Solver: hard constraint satisfaction (Timefold)")
    print("  RL/Greedy/Random: all mandatory tasks assigned + no skill/capacity violations")


# Save results

def save_results(results: dict, out_dir: str, config: dict):
    os.makedirs(out_dir, exist_ok=True)

    # JSON
    output = {'config': config, 'results': results}
    json_path = os.path.join(out_dir, 'comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # CSV
    csv_path = os.path.join(out_dir, 'comparison_results.csv')
    all_keys = sorted({k for r in results.values() for k in r})
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method'] + all_keys)
        for method, row in results.items():
            writer.writerow([method] + [row.get(k, '') for k in all_keys])
    print(f"CSV saved:  {csv_path}")
    print(f"\nTo plot: python plot_comparison_results.py --results {json_path}")


# Main

def compare(args):
    set_seed(args.seed)

    out_dir = os.path.join('results', args.run_tag) if args.run_tag else 'results'

    print(f"Run tag: {args.run_tag or '(none)'} | Seed: {args.seed}")
    print(f"Mode: {args.mode} | Employees: {args.num_employees} | "
          f"Tasks: {args.num_tasks} | Instances: {args.num_instances}")
    print(f"Output: {out_dir}/")
    print("-" * 60)

    # Generate instances (same set for all methods)
    generator = InstanceGenerator(mode=args.mode, seed=args.seed)
    instances = [
        generator.generate_instance(args.num_employees, args.num_tasks, mode=args.mode)
        for _ in range(args.num_instances)
    ]
    print(f"Generated {len(instances)} instances.")

    results: dict[str, dict] = {}

    # Greedy
    print("\n[1/6] Greedy baseline ...")
    results['greedy'] = evaluate_constructive(instances, greedy_baseline)

    # Random
    print("[2/6] Random baseline ...")
    results['random'] = evaluate_constructive(instances, random_baseline)

    # RL policy
    if args.rl_model and os.path.exists(args.rl_model):
        print(f"[3/6] RL policy ({args.rl_model}) ...")
        results['rl_policy'] = evaluate_rl(instances, args.rl_model)
    else:
        print("[3/6] RL policy: skipped (--rl-model not provided or file not found)")

    # Solver baseline
    if args.jar_path and os.path.exists(args.jar_path):
        print(f"[4/6] Solver baseline ({args.num_instances} solver calls) ...")
        results['solver'] = evaluate_solver(instances, args.jar_path)
    else:
        print("[4/6] Solver: skipped (--jar-path not provided or file not found)")

    # ML filter + Solver
    if args.jar_path and os.path.exists(args.jar_path) \
            and args.ml_model and os.path.exists(args.ml_model):
        print(f"[5/6] ML filter + Solver (threshold={args.ml_threshold}) ...")
        results['ml_filter_solver'] = evaluate_ml_filter_solver(
            instances, args.ml_model, args.jar_path, args.ml_threshold
        )
    else:
        print("[5/6] ML filter + Solver: skipped (--jar-path or --ml-model missing)")

    # RL filter + Solver
    if args.jar_path and os.path.exists(args.jar_path) \
            and args.rl_model and os.path.exists(args.rl_model):
        print(f"[6/6] RL filter + Solver (rule={args.rl_filter_rule}) ...")
        results['rl_filter_solver'] = evaluate_rl_filter_solver(
            instances, args.rl_model, args.jar_path,
            filter_rule=args.rl_filter_rule,
            assign_threshold_frac=args.rl_assign_threshold,
            reward_threshold=args.rl_reward_threshold,
        )
    else:
        print("[6/6] RL filter + Solver: skipped (--jar-path or --rl-model missing)")

    # Print table
    print_table(results, args.num_tasks)

    # Save
    config = vars(args)
    save_results(results, out_dir, config)


# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified method comparison for employee-task assignment'
    )
    parser.add_argument('--num-instances',  type=int,   default=100,
                        help='Number of instances to evaluate each method on (default: 100)')
    parser.add_argument('--num-employees',  type=int,   default=5,
                        help='Employees per instance (default: 5)')
    parser.add_argument('--num-tasks',      type=int,   default=8,
                        help='Tasks per instance (default: 8)')
    parser.add_argument('--mode',           choices=['easy', 'hard'], default='easy',
                        help='Instance difficulty mode (default: easy)')
    parser.add_argument('--seed',           type=int,   default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--run-tag',        type=str,   default='',
                        help='Tag for output subdirectory, e.g. easy_small')
    parser.add_argument('--rl-model',       type=str,   default='rl_artifacts/policy.pt',
                        help='Path to trained RL policy checkpoint')
    parser.add_argument('--ml-model',       type=str,
                        default='ml_artifacts/random_forest_feasibility_model.joblib',
                        help='Path to trained ML feasibility model')
    parser.add_argument('--ml-threshold',   type=float, default=0.7,
                        help='ML feasibility threshold for filtering (default: 0.7)')
    parser.add_argument('--jar-path',            type=str,   default='',
                        help='Path to Timefold solver JAR (enables solver and filter methods)')
    parser.add_argument('--rl-filter-rule',      choices=['feasible', 'threshold', 'reward'],
                        default='feasible',
                        help='RL filter rule: feasible=RL solution feasible, '
                             'threshold=RL assigned >= frac*tasks, reward=RL reward >= threshold '
                             '(default: feasible)')
    parser.add_argument('--rl-assign-threshold', type=float, default=0.75,
                        help='Fraction of tasks RL must assign for threshold rule (default: 0.75)')
    parser.add_argument('--rl-reward-threshold', type=float, default=0.0,
                        help='Minimum RL episode reward for reward rule (default: 0.0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    compare(args)
