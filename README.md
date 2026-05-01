# Master's Thesis: Combinatorial Optimization with ML-Assisted Feasibility Prediction

This project implements a pipeline for generating task assignment problem instances, solving them with Timefold, and using machine learning to predict instance feasibility for improved efficiency.

## Project Structure

```
.
├── timefold_solver/          # Java Timefold solver project
│   ├── pom.xml
│   └── src/main/java/com/example/solver/
│       ├── Employee.java
│       ├── Task.java
│       ├── TaskAssignment.java
│       ├── ScheduleSolution.java
│       ├── ScheduleConstraintProvider.java
│       └── SolverRunner.java
├── generate_instances.py     # Python instance generator
├── collect_dataset.py        # Python-Timefold integration for dataset collection
├── train_ml.py              # ML model training script
├── ml_filtering.py          # ML-assisted filtering pipeline
├── instances_dataset.csv    # Collected dataset
├── ml_artifacts/            # Trained models and metadata
├── data/                    # Generated instances
├── results/                 # Experiment results
└── README.md
```

## Prerequisites

- Java 17+
- Maven 3.6+
- Python 3.8+
- scikit-learn
- pandas
- numpy

## Installation

### Java Dependencies

```bash
cd timefold_solver
mvn clean compile
mvn package
```

This creates `target/timefold-solver-1.0-SNAPSHOT.jar`

### Python Dependencies

```bash
pip install scikit-learn pandas numpy
```

## Usage

### 1. Generate Instances

```bash
python generate_instances.py --num-employees 5 --num-tasks 10 --mode easy --output data/instance.json
```

### 2. Run Timefold Solver

```bash
java -jar timefold_solver/target/timefold-solver-1.0-SNAPSHOT.jar data/instance.json data/result.json
```

### 3. Collect Dataset

```bash
python collect_dataset.py --jar-path timefold_solver/target/timefold-solver-1.0-SNAPSHOT.jar --num-samples 100 --mode easy
```

### 4. Train ML Models

```bash
python train_ml.py
```

### 5. Run ML-Assisted Filtering

```bash
python ml_filtering.py --jar-path timefold_solver/target/timefold-solver-1.0-SNAPSHOT.jar --num-samples 50
```

## Configuration

### Instance Generation Modes

- `easy`: Higher capacity, shorter tasks, more feasible instances
- `hard`: Lower capacity, longer tasks, fewer feasible instances

### ML Features

- `num_employees`: Number of employees
- `num_tasks`: Number of tasks
- `num_skills`: Total unique skills
- `total_required_workload`: Sum of task durations
- `total_available_capacity`: Sum of employee capacities
- `capacity_ratio`: Capacity / workload
- `avg_candidates_per_task`: Average employees per skill
- `min_candidates_per_task`: Minimum employees per skill
- `fraction_single_candidate_tasks`: Tasks with only 1 candidate
- `fraction_zero_candidate_tasks`: Tasks with no candidates

## Experiment Results

The ML-assisted filtering should show:

- Higher acceptance rate of feasible instances
- Reduced solver calls for infeasible instances
- Improved overall pipeline efficiency

## Unified Experimental Comparison

The comparison pipeline evaluates all methods on the same instances and
exports thesis-ready tables and plots.

### Files

| File | Purpose |
|------|---------|
| `compare_methods.py` | Run all methods on shared instances; save CSV + JSON |
| `run_experiments.py` | Batch launcher for 4 preset experiment configs |
| `plot_comparison_results.py` | Bar charts from comparison JSON |

### Quick start (no solver required)

```bash
# Single experiment — RL vs greedy vs random
python compare_methods.py --num-instances 100 --run-tag easy_small

# All 4 presets (easy/hard × small/medium)
python run_experiments.py --rl-model rl_artifacts/policy.pt

# All 6 methods including RL warm-start + solver
python compare_methods.py \
    --jar-path timefold_solver/target/timefold-solver-1.0-SNAPSHOT.jar \
    --rl-model rl_artifacts/policy.pt \
    --ml-model ml_artifacts/random_forest_feasibility_model.joblib \
    --num-instances 100 --run-tag easy_full
```

> The sixth method (`rl_filter_solver`) runs automatically when both `--rl-model`
> and `--jar-path` are provided. Use `--rl-filter-rule` to choose the filter criterion.

### Generate plots

```bash
# Single experiment plots (feasible rate, tasks, runtime)
python plot_comparison_results.py --results results/easy_small/comparison_results.json

# Grouped easy vs hard comparison
python plot_comparison_results.py \
    --results results/easy_small/comparison_results.json \
    --compare results/hard_small/comparison_results.json \
    --labels "Easy" "Hard"
```

### Output structure

```
results/
  easy_small/
    comparison_results.csv      <- thesis table
    comparison_results.json     <- machine-readable data
    plot_cmp_feasible_rate.png  <- bar chart
    plot_cmp_tasks.png
    plot_cmp_runtime.png
    plot_grouped_*.png          <- easy vs hard (if --compare used)
  hard_small/ ...
  easy_medium/ ...
  hard_medium/ ...
```

## RL Filter + Solver (Experimental Hybrid Method)

This method uses the trained RL policy as a fast pre-screening probe before
calling the Timefold solver, analogous to `ml_filter_solver` but using actual
RL rollout outcomes instead of a supervised classifier.

### How it works

1. **RL probe**: The trained policy runs greedily on each instance and produces
   an assignment with a measurable quality signal (feasibility, assigned count, reward).
2. **Filter decision**: If the RL outcome passes the filter rule, the instance is
   accepted and sent to the solver. Otherwise it is rejected and solver is skipped.
3. **Solver phase**: Timefold runs normally on accepted instances only.

### Filter rules (`--rl-filter-rule`)

| Rule | Accept condition | When to use |
|------|-----------------|-------------|
| `feasible` *(default)* | RL constructs a fully feasible solution | Cleanest; directly tests constructive solvability |
| `threshold` | RL assigns ≥ `--rl-assign-threshold` × num_tasks | If feasibility is too strict |
| `reward` | RL episode reward ≥ `--rl-reward-threshold` | Reward-based confidence |

### What this does NOT prove

- It does not guarantee that accepted instances are better-solved by the solver.
- It does not show the RL policy is better than the ML classifier as a filter.
- Acceptance rate depends on how well the RL policy was trained.

### Commands

```bash
# Sanity check on 10 instances (feasibility filter, default)
python compare_methods.py \
    --jar-path timefold_solver/target/timefold-solver-1.0-SNAPSHOT.jar \
    --rl-model rl_artifacts/policy.pt \
    --num-instances 10 --run-tag sanity_rl_filter

# Full benchmark (all 6 methods)
python compare_methods.py \
    --jar-path timefold_solver/target/timefold-solver-1.0-SNAPSHOT.jar \
    --rl-model rl_artifacts/policy.pt \
    --ml-model ml_artifacts/random_forest_feasibility_model.joblib \
    --num-instances 100 --run-tag easy_full

# Threshold-based filter (accept if RL assigns ≥ 75% of tasks)
python compare_methods.py \
    --jar-path timefold_solver/target/timefold-solver-1.0-SNAPSHOT.jar \
    --rl-model rl_artifacts/policy.pt \
    --rl-filter-rule threshold --rl-assign-threshold 0.75 \
    --num-instances 100 --run-tag easy_threshold_filter

# Batch experiments across all presets
python run_experiments.py \
    --rl-model rl_artifacts/policy.pt \
    --ml-model ml_artifacts/random_forest_feasibility_model.joblib \
    --jar-path timefold_solver/target/timefold-solver-1.0-SNAPSHOT.jar

# Plot results (rl_filter_solver appears automatically)
python plot_comparison_results.py \
    --results results/easy_full/comparison_results.json
```

### Thesis note

Present `rl_filter_solver` as an **experimental RL-based prefilter**: the RL policy
acts as a fast constructive probe of instance solvability, and the solver is invoked
only when the probe succeeds. Key honest claims:
- *What it does*: uses the RL episode outcome (feasibility, assigned count, or reward)
  as a binary gate before the solver.
- *How it differs from `ml_filter_solver`*: ML filter uses a trained supervised model
  on instance-level features; RL filter uses actual policy execution on the instance.
  RL filter is slower to screen (runs a full episode) but requires no separate
  training phase for the filter itself.
- *How it differs from standalone `rl_policy`*: the RL result is discarded after
  filtering; the final solution always comes from the solver.
- *Limitations*: acceptance rate is bounded by RL policy quality; if the RL policy
  was not trained on the same distribution, the filter may reject or accept the
  wrong instances. Report acceptance rates and solver calls saved honestly and
  compare them numerically to `ml_filter_solver`.

---

## RL Pipeline (Reinforcement Learning)

The RL pipeline trains a neural policy to construct assignments **sequentially**
(one task at a time), without requiring a solver at inference time.

### Files

| File | Purpose |
|------|---------|
| `rl_environment.py` | Gym-style env: state, action, reward, episode logic |
| `rl_model.py` | PyTorch policy network (3-layer MLP, ~33k params) |
| `train_rl_agent.py` | REINFORCE training loop with baseline |
| `evaluate_rl_agent.py` | Evaluation vs greedy/random baselines |
| `plot_training_results.py` | Plotting utility for training curves and comparison charts |
| `rl_artifacts/` | Saved models, configs, training history, eval results |

### Install RL dependency

```bash
pip install torch
```

### Train

```bash
# Basic run (2000 episodes, 5 employees, 8 tasks)
python train_rl_agent.py --episodes 2000 --num-employees 5 --num-tasks 8

# Hard mode, named run
python train_rl_agent.py --episodes 5000 --mode hard --num-employees 8 --num-tasks 12 --run-tag hard_8x12

# Outputs: rl_artifacts/policy.pt, training_history.csv, train_config.txt
```

### Evaluate

```bash
# Evaluate against greedy and random baselines
python evaluate_rl_agent.py --model-path rl_artifacts/policy.pt --num-episodes 200

# Hard mode evaluation, named output
python evaluate_rl_agent.py --model-path rl_artifacts/hard_8x12/policy.pt --mode hard --run-tag hard_8x12

# Outputs: rl_artifacts/eval_results.json
```

### Plot results

```bash
# Training curves (reward, feasible rate, assigned tasks, baseline)
python plot_training_results.py --history rl_artifacts/training_history.csv

# Comparison bar chart (RL vs Greedy vs Random)
python plot_training_results.py --eval rl_artifacts/eval_results.json

# Both at once
python plot_training_results.py \
    --history rl_artifacts/training_history.csv \
    --eval    rl_artifacts/eval_results.json

# Outputs: plot_reward.png, plot_feasible_rate.png, plot_comparison.png, ...
```

### RL design summary

- **State**: current task features (one-hot skill, duration, priority, mandatory) +
  per-employee remaining capacity and skill-match flag (padded to MAX_EMPLOYEES=20) +
  episode progress signals. Total: 51 dimensions.
- **Action**: integer — assign to employee *i*, or UNASSIGNED (index 20).
- **Reward**: +10 valid assignment, −10 infeasible attempt, −10 mandatory unassigned,
  −2 optional unassigned, +priority bonus (1–5), +5×num_tasks full-solution bonus.
- **Algorithm**: REINFORCE with EMA baseline (Sutton & Barto §13.4).
- **Inference**: action masking applied (only valid employees selectable).

## Thesis Content Suggestions

1. **Problem Formulation**: Describe task assignment problem with constraints
2. **Instance Generation**: Explain realistic instance creation
3. **Solver Implementation**: Detail Timefold constraints and scoring
4. **Dataset Collection**: Show label distribution and feature analysis
5. **ML Model Development**: Compare Logistic Regression vs Random Forest
6. **Filtering Pipeline**: Demonstrate efficiency improvements
7. **Results Analysis**: Statistical comparison of baseline vs ML-assisted

## Improvements

- Add more ML features (task duration distribution, skill sparsity)
- Implement XGBoost/MLP models
- Add time window constraints
- Include employee preferences
- Experiment with different feasibility thresholds

## Troubleshooting

- Ensure Java 17+ is used for Timefold
- Check that all Python dependencies are installed
- Verify JAR path in scripts
- Monitor solver timeouts for complex instances
