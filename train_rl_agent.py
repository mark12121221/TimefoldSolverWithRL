import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim

from generate_instances import InstanceGenerator
from rl_environment import TaskAssignmentEnv, STATE_SIZE, ACTION_SIZE
from rl_model import PolicyNetwork, save_model


# Reproducibility


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Helpers

def compute_returns(rewards: list[float], gamma: float) -> list[float]:

    #Compute discounted returns G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
    #Returns a list of the same length as rewards.

    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def run_episode(
    env: TaskAssignmentEnv,
    policy: PolicyNetwork,
    instance: dict,
    device: torch.device,
) -> tuple[list, list, float, dict]:
    state_np = env.reset(instance)
    log_probs = []
    rewards = []
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state_np).to(device)
        action, log_prob = policy.get_action(state_tensor, action_mask=None, greedy=False)
        state_np, reward, done, _ = env.step(action.item())
        log_probs.append(log_prob)
        rewards.append(reward)

    return log_probs, rewards, sum(rewards), env.get_episode_summary()


def policy_gradient_update(
    optimizer: torch.optim.Optimizer,
    log_probs: list,
    advantages: list[float],
) -> float:

    #One REINFORCE gradient step.

    #Loss = −∑_t A_t · log π(a_t | s_t)   (negative → gradient ascent on return)

    #Advantages A_t = G_t − baseline are computed outside this function so the
    #baseline logic stays visible in train() and is not hidden inside the update.

    adv_tensor = torch.FloatTensor(advantages).to(log_probs[0].device)
    log_prob_tensor = torch.stack(log_probs)
    loss = -(log_prob_tensor * adv_tensor).sum()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for group in optimizer.param_groups for p in group['params']], max_norm=1.0
    )
    optimizer.step()
    return loss.item()


# Main training loop

def train(args):
    set_seed(args.seed)

    # Resolve output directory (optionally namespaced by run-tag)
    out_dir = os.path.join(args.model_output, args.run_tag) if args.run_tag else args.model_output
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Run tag: {args.run_tag or '(none)'} | Seed: {args.seed} | Device: {device}")
    print(f"State size: {STATE_SIZE}, Action size: {ACTION_SIZE}")
    print(f"Training for {args.episodes} episodes | mode={args.mode} | "
          f"employees={args.num_employees} | tasks={args.num_tasks}")
    print(f"Saving to: {out_dir}/")
    print("-" * 65)

    env = TaskAssignmentEnv()
    generator = InstanceGenerator(mode=args.mode, seed=args.seed)
    policy = PolicyNetwork(state_size=STATE_SIZE, action_size=ACTION_SIZE).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

    # Rolling window metrics (last WINDOW episodes)
    WINDOW = 100
    reward_window:     list[float] = []
    assigned_window:   list[float] = []
    unassigned_window: list[float] = []
    feasible_window:   list[bool]  = []

    # Running baseline b ≈ E[G_0] via EMA.  A_t = G_t − b  reduces gradient variance.
    baseline = 0.0
    EMA_ALPHA = 0.05

    # Per-episode history (saved to CSV at end)
    history: list[dict] = []

    start_time = time.time()

    for episode in range(1, args.episodes + 1):
        instance = generator.generate_instance(
            num_employees=args.num_employees,
            num_tasks=args.num_tasks,
            mode=args.mode,
        )

        log_probs, rewards, total_reward, summary = run_episode(env, policy, instance, device)
        returns = compute_returns(rewards, gamma=args.gamma)

        # Update baseline with G_0 (full episode return)
        baseline = (1 - EMA_ALPHA) * baseline + EMA_ALPHA * returns[0]

        # Advantages: A_t = G_t − b
        advantages = [G - baseline for G in returns]
        policy_gradient_update(optimizer, log_probs, advantages)

        # Track rolling metrics
        reward_window.append(total_reward)
        assigned_window.append(summary['assigned'])
        unassigned_window.append(summary['unassigned'])
        feasible_window.append(float(summary['feasible_solution']))
        if len(reward_window) > WINDOW:
            reward_window.pop(0)
            assigned_window.pop(0)
            unassigned_window.pop(0)
            feasible_window.pop(0)

        # Record per-episode data for history CSV
        history.append({
            'episode':    episode,
            'reward':     round(total_reward, 4),
            'assigned':   summary['assigned'],
            'unassigned': summary['unassigned'],
            'feasible':   int(summary['feasible_solution']),
            'baseline':   round(baseline, 4),
        })

        if episode % 100 == 0:
            print(
                f"Ep {episode:>5}/{args.episodes} | "
                f"Avg reward: {np.mean(reward_window):>8.2f} | "
                f"Assigned: {np.mean(assigned_window):.1f}/{args.num_tasks} | "
                f"Unassigned: {np.mean(unassigned_window):.1f} | "
                f"Feasible: {np.mean(feasible_window):.1%} | "
                f"Baseline: {baseline:.1f} | "
                f"Elapsed: {time.time() - start_time:.0f}s"
            )

    # Save artefacts

    model_path = os.path.join(out_dir, 'policy.pt')
    save_model(policy, model_path)
    print(f"\nModel saved        : {model_path}")

    config_path = os.path.join(out_dir, 'train_config.txt')
    with open(config_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    print(f"Config saved       : {config_path}")

    # Save per-episode history to CSV
    history_path = os.path.join(out_dir, 'training_history.csv')
    fieldnames = ['episode', 'reward', 'assigned', 'unassigned', 'feasible', 'baseline']
    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    print(f"Training history   : {history_path}")
    print(f"\nTo plot: python plot_training_results.py --history {history_path}")


# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train REINFORCE agent for sequential employee-task assignment'
    )
    parser.add_argument('--episodes',       type=int,   default=2000,
                        help='Number of training episodes (default: 2000)')
    parser.add_argument('--learning-rate',  type=float, default=1e-3,
                        help='Adam learning rate (default: 0.001)')
    parser.add_argument('--gamma',          type=float, default=0.99,
                        help='Discount factor for returns (default: 0.99)')
    parser.add_argument('--num-employees',  type=int,   default=5,
                        help='Number of employees per training instance (default: 5)')
    parser.add_argument('--num-tasks',      type=int,   default=8,
                        help='Number of tasks per training instance (default: 8)')
    parser.add_argument('--mode',           choices=['easy', 'hard'], default='easy',
                        help='Instance difficulty mode (default: easy)')
    parser.add_argument('--model-output',   type=str,   default='rl_artifacts',
                        help='Base directory for saved artefacts (default: rl_artifacts)')
    parser.add_argument('--run-tag',        type=str,   default='',
                        help='Optional tag; creates a subdirectory rl_artifacts/<tag>/')
    parser.add_argument('--seed',           type=int,   default=42,
                        help='Random seed for reproducibility (default: 42)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
