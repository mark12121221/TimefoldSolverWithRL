import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from generate_instances import InstanceGenerator


# Maximum number of employees the state/action space is padded to.
# This gives a fixed-size state vector regardless of instance size.
MAX_EMPLOYEES = 20

# All possible skills (including the phantom skill that makes tasks infeasible)
ALL_SKILLS = ['skillA', 'skillB', 'skillC', 'skillD', 'skillE', 'skillNone']
SKILL_TO_IDX = {s: i for i, s in enumerate(ALL_SKILLS)}

# State vector size (computed once, used by rl_model.py)
#   - current task features : len(ALL_SKILLS) + 3  (one-hot skill, duration_norm, priority_norm, mandatory)
#   - per employee (padded) : 2 * MAX_EMPLOYEES     (remaining_capacity_norm, has_required_skill)
#   - progress signals      : 2                     (tasks_done_frac, unassigned_frac)
STATE_SIZE = len(ALL_SKILLS) + 3 + 2 * MAX_EMPLOYEES + 2  # = 51

# Action space size: one action per padded employee slot + 1 "unassigned" action
ACTION_SIZE = MAX_EMPLOYEES + 1
UNASSIGNED_ACTION = MAX_EMPLOYEES  # index of the special "unassigned" action


class TaskAssignmentEnv:
    # Reward hyperparameters
    #
    # Design rationale:
    #   - Valid assignment: +10 base + priority bonus (up to +5).
    #   - Infeasible attempt (wrong skill or capacity exceeded): -10.
    #     Symmetric with the valid reward to create a strong learning signal.
    #   - Unassigned optional task: -2 (light; skipping is sometimes unavoidable).
    #   - Unassigned mandatory task: -10 (heavy; mandatory tasks must be covered).
    #   - Full-solution bonus: +5 * num_tasks, scales with problem size so the
    #     agent is incentivised to complete larger instances too.
    #
    REWARD_VALID_ASSIGN          = 10.0
    PENALTY_INFEASIBLE           = -10.0
    PENALTY_UNASSIGNED_OPTIONAL  = -2.0
    PENALTY_UNASSIGNED_MANDATORY = -10.0
    PRIORITY_BONUS_SCALE         = 1.0   # multiplied by task.priority (1–5)
    BONUS_FULL_SOLUTION_PER_TASK = 5.0   # total bonus = this × num_tasks

    def __init__(self):
        self.instance: Optional[Dict[str, Any]] = None
        self.employees: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []
        self.remaining_capacities: List[float] = []
        self.assignments: Dict[str, Optional[str]] = {}  # task_id -> employee_id or None
        self.task_idx: int = 0
        self.unassigned_count: int = 0
        self.infeasible_count: int = 0

        # Normalization constants (updated on reset)
        self._max_capacity: float = 1.0
        self._max_duration: float = 1.0

    # Public API
    def reset(self, instance: Dict[str, Any]) -> np.ndarray:
        self.instance = instance
        self.employees = instance['employees']
        self.tasks = instance['tasks']
        self.remaining_capacities = [
            float(emp['availableCapacity']) for emp in self.employees
        ]
        self.assignments = {}
        self.task_idx = 0
        self.unassigned_count = 0
        self.infeasible_count = 0

        # For normalization
        self._max_capacity = max(self.remaining_capacities) if self.remaining_capacities else 1.0
        self._max_duration = max(t['duration'] for t in self.tasks) if self.tasks else 1.0

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        assert self.instance is not None, "Call reset() before step()"
        assert self.task_idx < len(self.tasks), "Episode already finished"

        task = self.tasks[self.task_idx]
        reward = 0.0
        info: Dict[str, Any] = {}

        if action == UNASSIGNED_ACTION:
            # ---- UNASSIGNED action ----
            self.assignments[task['id']] = None
            self.unassigned_count += 1
            # Mandatory tasks carry a heavier penalty to teach the agent to
            # prioritize them. Optional tasks get a lighter penalty.
            if task.get('mandatory', False):
                reward = self.PENALTY_UNASSIGNED_MANDATORY
            else:
                reward = self.PENALTY_UNASSIGNED_OPTIONAL
            info['action_type'] = 'unassigned'
            info['mandatory'] = task.get('mandatory', False)

        elif 0 <= action < len(self.employees):
            # ---- Assign to employee[action] ----
            emp = self.employees[action]
            skill_ok = task['requiredSkill'] in emp['skills']
            capacity_ok = self.remaining_capacities[action] >= task['duration']

            if skill_ok and capacity_ok:
                # Valid assignment
                self.assignments[task['id']] = emp['id']
                self.remaining_capacities[action] -= task['duration']
                reward = (
                    self.REWARD_VALID_ASSIGN
                    + self.PRIORITY_BONUS_SCALE * task['priority']
                )
                info['action_type'] = 'valid_assign'
            else:
                # Infeasible assignment — mark as unassigned and penalise.
                # Penalty is symmetric with the valid-assign reward so the agent
                # gets a clear learning signal to avoid infeasible choices.
                self.assignments[task['id']] = None
                self.infeasible_count += 1
                reward = self.PENALTY_INFEASIBLE
                info['action_type'] = 'infeasible'
                info['reason'] = 'skill_mismatch' if not skill_ok else 'capacity_exceeded'

        else:
            # Action index >= num_employees but not UNASSIGNED — padded slot.
            # Treat as an unassigned optional action to handle gracefully.
            self.assignments[task['id']] = None
            self.unassigned_count += 1
            reward = self.PENALTY_UNASSIGNED_OPTIONAL
            info['action_type'] = 'out_of_range_unassigned'

        self.task_idx += 1
        done = self.task_idx >= len(self.tasks)

        if done:
            reward += self._end_of_episode_bonus()

        next_state = self._get_state() if not done else self._get_terminal_state()
        info['task_idx'] = self.task_idx
        info['unassigned_count'] = self.unassigned_count
        return next_state, reward, done, info

    def get_valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_SIZE, dtype=bool)
        if self.task_idx >= len(self.tasks):
            return mask

        task = self.tasks[self.task_idx]
        for i, emp in enumerate(self.employees):
            if (task['requiredSkill'] in emp['skills']
                    and self.remaining_capacities[i] >= task['duration']):
                mask[i] = True
        # Unassigned is always valid
        mask[UNASSIGNED_ACTION] = True
        return mask

    def get_episode_summary(self) -> Dict[str, Any]:
        """Return summary stats for the completed episode."""
        n_tasks = len(self.tasks)
        assigned = sum(1 for v in self.assignments.values() if v is not None)
        mandatory_tasks = [t for t in self.tasks if t.get('mandatory', False)]
        mandatory_assigned = sum(
            1 for t in mandatory_tasks if self.assignments.get(t['id']) is not None
        )
        feasible = (mandatory_assigned == len(mandatory_tasks)) and self.infeasible_count == 0

        return {
            'num_tasks': n_tasks,
            'assigned': assigned,
            'unassigned': n_tasks - assigned,
            'infeasible_actions': self.infeasible_count,
            'feasible_solution': feasible,
        }

    # State construction

    def _get_state(self) -> np.ndarray:
        """Build the state vector for the current (unprocessed) task."""
        if self.task_idx >= len(self.tasks):
            return self._get_terminal_state()

        task = self.tasks[self.task_idx]
        state = np.zeros(STATE_SIZE, dtype=np.float32)
        offset = 0

        # --- Current task: one-hot skill (6 dims) ---
        skill_idx = SKILL_TO_IDX.get(task['requiredSkill'], len(ALL_SKILLS) - 1)
        state[offset + skill_idx] = 1.0
        offset += len(ALL_SKILLS)

        # --- Current task: duration (normalized), priority (normalized), mandatory ---
        state[offset]     = task['duration'] / max(self._max_duration, 1.0)
        state[offset + 1] = task['priority'] / 5.0
        state[offset + 2] = float(task.get('mandatory', False))
        offset += 3

        # --- Per-employee features (padded to MAX_EMPLOYEES) ---
        for i in range(MAX_EMPLOYEES):
            if i < len(self.employees):
                emp = self.employees[i]
                cap_norm = self.remaining_capacities[i] / max(self._max_capacity, 1.0)
                has_skill = float(task['requiredSkill'] in emp['skills'])
                state[offset + 2 * i]     = cap_norm
                state[offset + 2 * i + 1] = has_skill
            # else: stays 0 (padded)
        offset += 2 * MAX_EMPLOYEES

        # --- Progress signals ---
        n_tasks = len(self.tasks)
        state[offset]     = self.task_idx / n_tasks          # fraction of tasks processed
        state[offset + 1] = self.unassigned_count / n_tasks  # fraction unassigned so far
        # offset += 2  (end of vector)

        return state

    def _get_terminal_state(self) -> np.ndarray:
        #Return a zero state when the episode is done.
        return np.zeros(STATE_SIZE, dtype=np.float32)

    # Reward helper
    def _end_of_episode_bonus(self) -> float:
        mandatory_tasks = [t for t in self.tasks if t.get('mandatory', False)]
        if not mandatory_tasks:
            return 0.0
        all_mandatory_assigned = all(
            self.assignments.get(t['id']) is not None for t in mandatory_tasks
        )
        if all_mandatory_assigned and self.infeasible_count == 0:
            return self.BONUS_FULL_SOLUTION_PER_TASK * len(self.tasks)
        return 0.0


# Quick smoke-test
if __name__ == '__main__':
    gen = InstanceGenerator(mode='easy', seed=42)
    instance = gen.generate_instance(num_employees=4, num_tasks=6)

    env = TaskAssignmentEnv()
    state = env.reset(instance)
    print(f"State size: {len(state)}  (expected {STATE_SIZE})")
    print(f"Action size: {ACTION_SIZE}")

    done = False
    total_reward = 0.0
    step = 0
    while not done:
        # Random policy
        action = np.random.randint(0, ACTION_SIZE)
        state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        print(f"Step {step}: action={action}, reward={reward:.1f}, info={info}")

    summary = env.get_episode_summary()
    print(f"\nEpisode summary: {summary}")
    print(f"Total reward: {total_reward:.1f}")
