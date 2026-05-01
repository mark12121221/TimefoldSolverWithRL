import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from generate_instances import InstanceGenerator
from collect_dataset import DatasetCollector


class MLFilteringPipeline:
    def __init__(
        self,
        model_path: str,
        java_jar_path: str,
        threshold: float = 0.7,
        baseline_csv: str = 'results/baseline_dataset.csv',
        filtered_csv: str = 'results/filtered_dataset.csv'
    ):
        self.model = joblib.load(model_path)
        self.java_jar_path = java_jar_path
        self.threshold = threshold

        os.makedirs('results', exist_ok=True)

        self.baseline_collector = DatasetCollector(java_jar_path, baseline_csv)
        self.filtered_collector = DatasetCollector(java_jar_path, filtered_csv)

    def calculate_features(self, instance: Dict[str, Any]) -> np.ndarray:
        employees = instance['employees']
        tasks = instance['tasks']

        num_employees = len(employees)
        num_tasks = len(tasks)
        num_skills = len(set(skill for emp in employees for skill in emp['skills']))

        total_required_workload = sum(task['duration'] for task in tasks)
        total_available_capacity = sum(emp['availableCapacity'] for emp in employees)

        capacity_ratio = (
            total_required_workload / total_available_capacity
            if total_available_capacity > 0 else 999
        )

        candidates_per_task = []
        for task in tasks:
            skill = task['requiredSkill']
            candidates = sum(1 for emp in employees if skill in emp['skills'])
            candidates_per_task.append(candidates)

        avg_candidates_per_task = sum(candidates_per_task) / len(candidates_per_task) if candidates_per_task else 0
        min_candidates_per_task = min(candidates_per_task) if candidates_per_task else 0
        fraction_single_candidate_tasks = (
            sum(1 for c in candidates_per_task if c == 1) / len(candidates_per_task)
            if candidates_per_task else 0
        )
        fraction_zero_candidate_tasks = (
            sum(1 for c in candidates_per_task if c == 0) / len(candidates_per_task)
            if candidates_per_task else 0
        )

        features = [
            num_employees,
            num_tasks,
            num_skills,
            total_required_workload,
            total_available_capacity,
            capacity_ratio,
            avg_candidates_per_task,
            min_candidates_per_task,
            fraction_single_candidate_tasks,
            fraction_zero_candidate_tasks
        ]

        return np.array(features).reshape(1, -1)

    def predict_feasibility(self, instance: Dict[str, Any]) -> float:
        features = self.calculate_features(instance)
        probability = self.model.predict_proba(features)[0, 1]
        return probability

    def run_baseline_experiment(self, num_samples: int, num_employees: int, num_tasks: int, mode: str):
        print("Running baseline experiment (no ML filtering)...")
        results = []

        for i in range(num_samples):
            success = self.baseline_collector.collect_sample(num_employees, num_tasks, mode)
            if success:
                results.append(True)
            print(f"Baseline: {len(results)}/{i+1} successful")

        return results

    def run_ml_filtered_experiment(self, num_samples: int, num_employees: int, num_tasks: int, mode: str):
        print("Running ML-assisted filtering experiment...")
        results = []
        accepted = 0
        rejected = 0
        total_solver_calls = 0

        generator = InstanceGenerator(mode=mode)

        for i in range(num_samples):
            instance = generator.generate_instance(num_employees, num_tasks, mode=mode)
            prob_feasible = self.predict_feasibility(instance)

            if prob_feasible >= self.threshold:
                accepted += 1
                success = self.filtered_collector.collect_given_instance(instance, mode)
                total_solver_calls += 1
                if success:
                    results.append(True)
            else:
                rejected += 1

            print(
                f"ML Filtering: {accepted}/{accepted + rejected} accepted, "
                f"{rejected} rejected, {total_solver_calls} solver calls"
            )

        return results, accepted, rejected, total_solver_calls

    def compare_experiments(
        self,
        baseline_results: List[bool],
        filtered_results: List[bool],
        accepted: int,
        rejected: int,
        total_solver_calls: int
    ):
        print("\n=== Experiment Comparison ===")

        baseline_successful = sum(baseline_results)
        baseline_total = len(baseline_results)
        baseline_feasible_rate = baseline_successful / baseline_total if baseline_total > 0 else 0

        filtered_successful = sum(filtered_results)
        filtered_total = len(filtered_results)
        filtered_feasible_rate = filtered_successful / filtered_total if filtered_total > 0 else 0

        total_generated = accepted + rejected
        acceptance_rate = accepted / total_generated if total_generated > 0 else 0.0

        print(f"Baseline: {baseline_successful}/{baseline_total} feasible ({baseline_feasible_rate:.1%})")
        print(f"ML Filtered: {filtered_successful}/{filtered_total} feasible ({filtered_feasible_rate:.1%})")
        print(f"Acceptance rate: {accepted}/{total_generated} ({acceptance_rate:.1%})")
        print(f"Solver calls saved: {rejected} out of {total_generated}")

        if os.path.exists('results/filtered_dataset.csv'):
            df = pd.read_csv('results/filtered_dataset.csv')
            if not df.empty:
                avg_runtime = df['runtime_ms'].mean()
                avg_unassigned = df['unassigned_tasks'].mean()
                print(f"Average runtime of kept instances: {avg_runtime:.0f} ms")
                print(f"Average unassigned tasks of kept instances: {avg_unassigned:.2f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='ML-assisted filtering pipeline')
    parser.add_argument('--model-path', type=str, default='ml_artifacts/random_forest_feasibility_model.joblib', help='Path to trained model')
    parser.add_argument('--jar-path', type=str, required=True, help='Path to Timefold solver JAR')
    parser.add_argument('--threshold', type=float, default=0.7, help='Feasibility probability threshold')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of samples per experiment')
    parser.add_argument('--num-employees', type=int, default=5, help='Number of employees per instance')
    parser.add_argument('--num-tasks', type=int, default=10, help='Number of tasks per instance')
    parser.add_argument('--mode', choices=['easy', 'hard'], default='easy', help='Generation mode')

    args = parser.parse_args()

    pipeline = MLFilteringPipeline(args.model_path, args.jar_path, args.threshold)

    baseline_results = pipeline.run_baseline_experiment(
        args.num_samples, args.num_employees, args.num_tasks, args.mode
    )

    filtered_results, accepted, rejected, solver_calls = pipeline.run_ml_filtered_experiment(
        args.num_samples, args.num_employees, args.num_tasks, args.mode
    )

    pipeline.compare_experiments(
        baseline_results, filtered_results, accepted, rejected, solver_calls
    )


if __name__ == '__main__':
    main()