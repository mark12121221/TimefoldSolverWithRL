import json
import random
import os
from typing import List, Dict, Any


class InstanceGenerator:
    def __init__(self, mode: str = 'easy', seed: int | None = None):
        self.mode = mode
        self.skills = ['skillA', 'skillB', 'skillC', 'skillD', 'skillE']
        self.random = random.Random(seed)

    def generate_instance(self, num_employees: int, num_tasks: int, mode: str | None = None) -> Dict[str, Any]:
        if mode is not None:
            self.mode = mode

        employees = self._generate_employees(num_employees)

        # Target percentage of unsolved problems: easy ~12%, hard ~10%
        # (hard mode additionally gives ~25-30% natural infeasibility
        # due to skill-bottleneck and limited capacity → total ~35-40%)
        infeasible_rate = 0.16 if self.mode == 'easy' else 0.10
        force_infeasible = self.random.random() < infeasible_rate

        tasks = self._generate_tasks(num_tasks, employees, force_infeasible)

        return {
            'employees': employees,
            'tasks': tasks
        }

    def _generate_employees(self, num_employees: int) -> List[Dict[str, Any]]:
        employees = []
        for i in range(num_employees):
            if self.mode == 'easy':
                num_skills = self.random.randint(2, 4)
                employee_skills = self.random.sample(self.skills, num_skills)
                capacity = self.random.randint(25, 50)
            else:
                # hard: 1-3 skills, capacity moderately limited
                num_skills = self.random.randint(1, 3)
                employee_skills = self.random.sample(self.skills, num_skills)
                capacity = self.random.randint(15, 30)

            employees.append({
                'id': f'E{i+1}',
                'skills': employee_skills,
                'availableCapacity': capacity
            })
        return employees

    # A skill that no employee possesses—guarantees that the problem cannot be solved
    _PHANTOM_SKILL = 'skillNone'

    def _generate_tasks(self, num_tasks: int, employees: List[Dict[str, Any]],
                        force_infeasible: bool = False) -> List[Dict[str, Any]]:
        tasks = []
        all_skills = set()
        for emp in employees:
            all_skills.update(emp['skills'])

        all_skills = list(all_skills)
        # Skills that no employee possesses
        impossible_skills = [s for s in self.skills if s not in all_skills]

        # If the instance should be infeasible — one random task will get
        # a skill that no one has → hard constraint guaranteed to be violated
        infeasible_idx = self.random.randint(0, num_tasks - 1) if force_infeasible else -1

        for i in range(num_tasks):
            if i == infeasible_idx:
                required_skill = self._PHANTOM_SKILL
            elif self.mode == 'easy':
                required_skill = self.random.choice(all_skills)
            else:
                required_skill = self.random.choice(all_skills)

            if self.mode == 'easy':
                duration = self.random.randint(1, 5)
            else:
                duration = self.random.randint(3, 10)

            priority = self.random.randint(1, 5)
            mandatory = self.random.random() < 0.7

            tasks.append({
                'id': f'T{i+1}',
                'requiredSkill': required_skill,
                'duration': duration,
                'priority': priority,
                'mandatory': mandatory
            })

        return tasks

    def save_instance(self, instance: Dict[str, Any], filename: str):
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(instance, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate task assignment instances')
    parser.add_argument('--num-employees', type=int, default=5, help='Number of employees')
    parser.add_argument('--num-tasks', type=int, default=10, help='Number of tasks')
    parser.add_argument('--mode', choices=['easy', 'hard'], default='easy', help='Generation mode')
    parser.add_argument('--output', type=str, default='data/instance.json', help='Output file')

    args = parser.parse_args()

    generator = InstanceGenerator(mode=args.mode)
    instance = generator.generate_instance(args.num_employees, args.num_tasks, mode=args.mode)
    generator.save_instance(instance, args.output)

    print(f"Generated instance with {len(instance['employees'])} employees and {len(instance['tasks'])} tasks")
    print(f"Saved to {args.output}")
