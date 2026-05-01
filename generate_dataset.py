import random
import pandas as pd

NUM_INSTANCES = 200000
OUTPUT_FILE = "instances_dataset_baseline.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def generate_instance():
    mode = random.choice(["easy", "hard"])

    num_employees = random.randint(10, 30)
    num_tasks = random.randint(20, 80)
    num_skills = random.randint(3, 8)

    employee_capacities = [random.randint(25, 60) for _ in range(num_employees)]
    total_available_capacity = sum(employee_capacities)

    candidate_counts = []
    single_candidate_tasks = 0
    zero_candidate_tasks = 0

    if mode == "easy":
        # Easy: moderate intensity, greater flexibility
        target_capacity_ratio = random.uniform(0.65, 0.92)
        total_required_workload = int(total_available_capacity * target_capacity_ratio)
        num_tasks = max(20, min(num_tasks, 70))

        task_durations = []
        remaining = total_required_workload
        for i in range(num_tasks):
            if i == num_tasks - 1:
                duration = max(1, remaining)
            else:
                max_allowed = min(4, max(1, remaining - (num_tasks - i - 1)))
                duration = random.randint(1, max_allowed)
            task_durations.append(duration)
            remaining -= duration

        for _ in range(num_tasks):
            r = random.random()

            if r < 0.01:
                candidates = 0
            elif r < 0.08:
                candidates = 1
            else:
                low = max(2, num_employees // 4)
                high = max(low, min(num_employees, num_employees // 2))
                candidates = random.randint(low, high)

            candidate_counts.append(candidates)

            if candidates == 1:
                single_candidate_tasks += 1
            if candidates == 0:
                zero_candidate_tasks += 1

    else:
        # Hard: high workload, less flexibility, but not completely "broken" instance
        target_capacity_ratio = random.uniform(0.95, 1.15)
        total_required_workload = int(total_available_capacity * target_capacity_ratio)
        num_tasks = max(20, min(num_tasks, 80))

        task_durations = []
        remaining = total_required_workload
        for i in range(num_tasks):
            if i == num_tasks - 1:
                duration = max(1, remaining)
            else:
                # Reserve at least 2 for each remaining task ((2, ...))
                max_allowed = min(7, max(2, remaining - 2 * (num_tasks - i - 1)))
                duration = random.randint(2, max_allowed)
            task_durations.append(duration)
            remaining -= duration

        for _ in range(num_tasks):
            r = random.random()

            if r < 0.08:
                candidates = 0
            elif r < 0.40:
                candidates = 1
            elif r < 0.75:
                candidates = 2
            else:
                low = 3
                high = max(low, min(num_employees, max(3, num_employees // 4)))
                candidates = random.randint(low, high)

            candidate_counts.append(candidates)

            if candidates == 1:
                single_candidate_tasks += 1
            if candidates == 0:
                zero_candidate_tasks += 1

    avg_candidates_per_task = sum(candidate_counts) / len(candidate_counts)
    min_candidates_per_task = min(candidate_counts)
    fraction_single_candidate_tasks = single_candidate_tasks / num_tasks
    fraction_zero_candidate_tasks = zero_candidate_tasks / num_tasks

    capacity_ratio = (
        total_required_workload / total_available_capacity
        if total_available_capacity > 0 else 999
    )

    # Baseline label
    is_feasible = 1

    if mode == "easy":
        if fraction_zero_candidate_tasks > 0.03:
            is_feasible = 0
        elif capacity_ratio > 0.98:
            is_feasible = 0
        elif avg_candidates_per_task < 2.0:
            is_feasible = 0
    else:
        if fraction_zero_candidate_tasks > 0.09:
            is_feasible = 0
        elif capacity_ratio > 1.10:
            is_feasible = 0
        elif capacity_ratio > 1.03 and avg_candidates_per_task < 1.5:
            is_feasible = 0
        elif fraction_single_candidate_tasks > 0.40 and avg_candidates_per_task < 1.6:
            is_feasible = 0

    return {
        "mode": mode,
        "num_employees": num_employees,
        "num_tasks": num_tasks,
        "num_skills": num_skills,
        "total_required_workload": total_required_workload,
        "total_available_capacity": total_available_capacity,
        "capacity_ratio": round(capacity_ratio, 4),
        "avg_candidates_per_task": round(avg_candidates_per_task, 4),
        "min_candidates_per_task": min_candidates_per_task,
        "fraction_single_candidate_tasks": round(fraction_single_candidate_tasks, 4),
        "fraction_zero_candidate_tasks": round(fraction_zero_candidate_tasks, 4),
        "is_feasible": is_feasible,
    }


def generate_dataset(num_instances: int):
    rows = [generate_instance() for _ in range(num_instances)]
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Created file: {OUTPUT_FILE}")
    print(f"Number of instances: {len(df)}")

    print("\nDistribution of label is_feasible:")
    print(df["is_feasible"].value_counts())

    print("\nDistribution of mode:")
    print(df["mode"].value_counts())

    print("\nCross-tab mode x is_feasible:")
    print(pd.crosstab(df["mode"], df["is_feasible"]))

    print("\nAverage values by mode:")
    print(
        df.groupby("mode")[
            [
                "capacity_ratio",
                "avg_candidates_per_task",
                "fraction_single_candidate_tasks",
                "fraction_zero_candidate_tasks",
            ]
        ].mean()
    )

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    generate_dataset(NUM_INSTANCES)