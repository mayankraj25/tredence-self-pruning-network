# Self-Pruning Neural Network on CIFAR-10

> Tredence AI Engineering Internship — Case Study Submission

## Overview

A feed-forward neural network that **learns to prune itself during training** by attaching learnable sigmoid gates to every weight. An L1 sparsity penalty drives most gates to exactly zero, effectively removing connections without any post-training step.

## Project Structure

```
├── self_pruning_cifar10.py   # Main script (all code in one file as required)
├── REPORT.md                 # Analysis report with results table & plots
├── requirements.txt          # Python dependencies
└── outputs/                  # Generated after running the script
    ├── gate_dist_lambda_*.png
    ├── training_curves.png
    └── results_summary.csv
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run all three λ experiments (30 epochs each)
python self_pruning_cifar10.py --lambdas 1e-4 5e-4 2e-3 --epochs 30

# Custom run
python self_pruning_cifar10.py --lambdas 1e-4 5e-4 2e-3 --epochs 50 --batch 128 --out_dir ./my_outputs
```

## Key Concepts

| Component | Description |
|-----------|-------------|
| `PrunableLinear` | Custom layer with `gate_scores` parameter per weight; gates = sigmoid(gate_scores) |
| Sparsity Loss | L1 norm of all gate values — constant gradient pushes gates to exactly 0 |
| Total Loss | `CrossEntropy + λ × SparsityLoss` |
| λ sweep | Low (1e-4) / Medium (5e-4) / High (2e-3) to show sparsity–accuracy trade-off |

## Results

| Lambda (λ) | Test Accuracy | Sparsity (%) |
|:----------:|:-------------:|:------------:|
| 1e-4 | ~53% | ~20% |
| 5e-4 | ~50% | ~58% |
| 2e-3 | ~43% | ~80% |

*(Fill in your actual numbers after running the script.)*

## Hardware

Auto-detects CUDA → MPS → CPU. 30 epochs on CPU ≈ 15–25 min. On GPU < 5 min.