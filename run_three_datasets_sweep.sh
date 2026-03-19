#!/bin/bash
# Run full rho sweep on 3 datasets: LifeExpectancy, BreastCancerWisconsin, SeoulBikeData
# 7 rho values × 5 trials = 35 experiments per dataset

set -e  # Exit on error

# Activate virtual environment
source privateggmenv/bin/activate

echo "=============================================================================="
echo "LAUNCHING FULL RHO SWEEP ON 3 DATASETS"
echo "=============================================================================="
echo "Datasets: LifeExpectancy (d=15), BreastCancerWisconsin (d=9), SeoulBikeData (d=11)"
echo "Rho values: 0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 5.0"
echo "Trials per rho: 1"
echo "Total experiments: 3 datasets × 7 rho × 1 trial = 21 experiments"
echo "=============================================================================="
echo ""

# LifeExpectancy (d=15)
echo "[1/3] Starting LifeExpectancy (d=15)..."
python3 experiments/test_iterative_annealing.py \
    --data-type real \
    --data-name LifeExpectancy \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 5.0 \
    --n-trials 1 \
    --alpha 0.3 \
    --allow-remeasurement &

PID1=$!
echo "LifeExpectancy launched in background (PID: $PID1)"
echo ""

# BreastCancerWisconsin (d=9)
echo "[2/3] Starting BreastCancerWisconsin (d=9)..."
python3 experiments/test_iterative_annealing.py \
    --data-type real \
    --data-name BreastCancerWisconsin \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 5.0 \
    --n-trials 1 \
    --alpha 0.3 \
    --allow-remeasurement &

PID2=$!
echo "BreastCancerWisconsin launched in background (PID: $PID2)"
echo ""

# SeoulBikeData (d=11)
echo "[3/3] Starting SeoulBikeData (d=11)..."
python3 experiments/test_iterative_annealing.py \
    --data-type real \
    --data-name SeoulBikeData \
    --rho-values 0.0001 0.001 0.01 0.1 1.0 2.0 5.0 \
    --n-trials 1 \
    --alpha 0.3 \
    --allow-remeasurement &

PID3=$!
echo "SeoulBikeData launched in background (PID: $PID3)"
echo ""

echo "=============================================================================="
echo "ALL 3 DATASETS LAUNCHED IN PARALLEL"
echo "=============================================================================="
echo "Process IDs:"
echo "  LifeExpectancy:          PID $PID1"
echo "  BreastCancerWisconsin:   PID $PID2"
echo "  SeoulBikeData:           PID $PID3"
echo ""
echo "Monitor progress with:"
echo "  tail -f nohup.out"
echo "  ps aux | grep test_iterative_annealing"
echo ""
echo "Waiting for all processes to complete..."
echo "=============================================================================="

# Wait for all background processes
wait $PID1
echo "✓ LifeExpectancy completed"

wait $PID2
echo "✓ BreastCancerWisconsin completed"

wait $PID3
echo "✓ SeoulBikeData completed"

echo ""
echo "=============================================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=============================================================================="
echo "Results saved to:"
echo "  results/linesearch/annealing_privacy_analysis_LifeExpectancy_d15_alpha0.3_remeasure.json"
echo "  results/linesearch/annealing_privacy_analysis_BreastCancerWisconsin_d9_alpha0.3_remeasure.json"
echo "  results/linesearch/annealing_privacy_analysis_SeoulBikeData_d11_alpha0.3_remeasure.json"
echo "=============================================================================="
