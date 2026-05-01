#!/bin/bash

# Master's Thesis Pipeline Runner
# This script runs the complete pipeline: generate -> solve -> collect -> train -> filter

set -e

echo "=== Master's Thesis Pipeline ==="

# Configuration
JAR_PATH="timefold_solver/target/timefold-solver-1.0-SNAPSHOT.jar"
CSV_PATH="instances_dataset.csv"
MODEL_PATH="ml_artifacts/random_forest_feasibility_model.joblib"

# Step 1: Build Java project
echo "Step 1: Building Java solver..."
cd timefold_solver
mvn clean package -q
cd ..
echo "✓ Java project built"

# Step 2: Collect dataset (optional, if not exists)
if [ ! -f "$CSV_PATH" ] || [ $(wc -l < "$CSV_PATH") -lt 10 ]; then
    echo "Step 2: Collecting dataset..."
    python collect_dataset.py --jar-path "$JAR_PATH" --num-samples 50 --mode easy
    python collect_dataset.py --jar-path "$JAR_PATH" --num-samples 50 --mode hard
    echo "✓ Dataset collected"
else
    echo "✓ Dataset already exists"
fi

# Step 3: Train ML model
echo "Step 3: Training ML model..."
python train_ml.py
echo "✓ ML model trained"

# Step 4: Run ML-assisted filtering experiment
echo "Step 4: Running ML-assisted filtering..."
python ml_filtering.py --jar-path "$JAR_PATH" --num-samples 100
echo "✓ Filtering experiment completed"

echo "=== Pipeline completed successfully ==="
echo "Check results/ for experiment outputs"
echo "Check ml_artifacts/ for trained models"