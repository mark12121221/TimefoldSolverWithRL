import json
import csv
import subprocess
import os
import tempfile
from generate_instances import InstanceGenerator
from typing import Dict, Any, List

class DatasetCollector:
    def __init__(self, java_jar_path: str, csv_path: str = 'instances_dataset.csv'):
        self.java_jar_path = java_jar_path
        self.csv_path = csv_path
        self.generator = InstanceGenerator()

        # Initialize CSV if not exists
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'num_employees', 'num_tasks', 'num_skills', 'total_required_workload',
                    'total_available_capacity', 'capacity_ratio', 'avg_candidates_per_task',
                    'min_candidates_per_task', 'fraction_single_candidate_tasks',
                    'fraction_zero_candidate_tasks', 'is_feasible', 'score', 'runtime_ms',
                    'unassigned_tasks', 'mode'
                ])

    def collect_sample(self, num_employees: int, num_tasks: int, mode: str) -> bool:
        # Generate instance
        instance = self.generator.generate_instance(num_employees, num_tasks, mode=mode)

        # Calculate features
        features = self._calculate_features(instance, mode)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(instance, f, indent=2)
            input_file = f.name

        output_file = input_file.replace('.json', '_result.json')

        try:
            # Call Java solver
            result = subprocess.run([
                'java', '-jar', self.java_jar_path, input_file, output_file
            ], capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"Solver failed: {result.stderr}")
                return False

            # Read result
            with open(output_file, 'r') as f:
                solver_result = json.load(f)

            # Save to CSV
            row = [
                    features['num_employees'], features['num_tasks'], features['num_skills'],
                    features['total_required_workload'], features['total_available_capacity'],
                    features['capacity_ratio'], features['avg_candidates_per_task'],
                    features['min_candidates_per_task'], features['fraction_single_candidate_tasks'],
                    features['fraction_zero_candidate_tasks'], int(solver_result['feasible']),
                    solver_result['score'], solver_result['runtimeMs'],
                    solver_result['unassignedTasks'], mode
                ]

            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(f"Collected sample: feasible={solver_result['feasible']}, score={solver_result['score']}")
            return True

        except Exception as e:
            print(f"Error collecting sample: {e}")
            return False
        finally:
            # Cleanup
            if os.path.exists(input_file):
                os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)

    def collect_given_instance(self, instance: Dict[str, Any], mode: str) -> bool:
        # Takes an existing instance (does not generate a new one) and runs the solver.
        features = self._calculate_features(instance, mode)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(instance, f, indent=2)
            input_file = f.name

        output_file = input_file.replace('.json', '_result.json')

        try:
            result = subprocess.run([
                'java', '-jar', self.java_jar_path, input_file, output_file
            ], capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"Solver failed: {result.stderr}")
                return False

            with open(output_file, 'r') as f:
                solver_result = json.load(f)

            row = [
                features['num_employees'], features['num_tasks'], features['num_skills'],
                features['total_required_workload'], features['total_available_capacity'],
                features['capacity_ratio'], features['avg_candidates_per_task'],
                features['min_candidates_per_task'], features['fraction_single_candidate_tasks'],
                features['fraction_zero_candidate_tasks'], int(solver_result['feasible']),
                solver_result['score'], solver_result['runtimeMs'],
                solver_result['unassignedTasks'], mode
            ]

            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(f"Collected sample: feasible={solver_result['feasible']}, score={solver_result['score']}")
            return True

        except Exception as e:
            print(f"Error collecting sample: {e}")
            return False
        finally:
            if os.path.exists(input_file):
                os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)

    def _calculate_features(self, instance: Dict[str, Any], mode: str) -> Dict[str, Any]:
        employees = instance['employees']
        tasks = instance['tasks']

        num_employees = len(employees)
        num_tasks = len(tasks)
        num_skills = len(set(skill for emp in employees for skill in emp['skills']))

        total_required_workload = sum(task['duration'] for task in tasks)
        total_available_capacity = sum(emp['availableCapacity'] for emp in employees)
        # workload / capacity: >1 indicates an overload (same as in generate_dataset.py and ml_filtering.py)
        capacity_ratio = total_required_workload / total_available_capacity if total_available_capacity > 0 else 999

        # Candidates per task
        candidates_per_task = []
        for task in tasks:
            skill = task['requiredSkill']
            candidates = sum(1 for emp in employees if skill in emp['skills'])
            candidates_per_task.append(candidates)

        avg_candidates_per_task = sum(candidates_per_task) / len(candidates_per_task) if candidates_per_task else 0
        min_candidates_per_task = min(candidates_per_task) if candidates_per_task else 0
        fraction_single_candidate_tasks = sum(1 for c in candidates_per_task if c == 1) / len(candidates_per_task) if candidates_per_task else 0
        fraction_zero_candidate_tasks = sum(1 for c in candidates_per_task if c == 0) / len(candidates_per_task) if candidates_per_task else 0

        return {
            'num_employees': num_employees,
            'num_tasks': num_tasks,
            'num_skills': num_skills,
            'total_required_workload': total_required_workload,
            'total_available_capacity': total_available_capacity,
            'capacity_ratio': capacity_ratio,
            'avg_candidates_per_task': avg_candidates_per_task,
            'min_candidates_per_task': min_candidates_per_task,
            'fraction_single_candidate_tasks': fraction_single_candidate_tasks,
            'fraction_zero_candidate_tasks': fraction_zero_candidate_tasks,
            'mode': mode
        }

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Collect dataset samples')
    parser.add_argument('--jar-path', type=str, required=True, help='Path to Timefold solver JAR')
    parser.add_argument('--csv-path', type=str, default='instances_dataset.csv', help='Output CSV path')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to collect')
    parser.add_argument('--num-employees', type=int, default=5, help='Number of employees per instance')
    parser.add_argument('--num-tasks', type=int, default=10, help='Number of tasks per instance')
    parser.add_argument('--mode', choices=['easy', 'hard'], default='easy', help='Generation mode')

    args = parser.parse_args()

    collector = DatasetCollector(args.jar_path, args.csv_path)

    collected = 0
    for i in range(args.num_samples):
        if collector.collect_sample(args.num_employees, args.num_tasks, args.mode):
            collected += 1
        print(f"Progress: {collected}/{args.num_samples}")

    print(f"Collected {collected} samples out of {args.num_samples}")