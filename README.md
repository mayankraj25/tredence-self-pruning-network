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

# Run all three λ experiments (50 epochs each)
python self_pruning_cifar10.py --lambdas 0.5 2.0 5.0 --epochs 50

# Custom run
python self_pruning_cifar10.py --lambdas 0.5 2.0 5.0 --epochs 50 --batch 128 --out_dir ./my_outputs
```

## Key Concepts

| Component | Description |
|-----------|-------------|
| `PrunableLinear` | Custom layer with `gate_scores` parameter per weight; gates = clamp(gate_scores, 0, 1) |
| Sparsity Loss | Mean gate value across all layers — normalised L1 penalty pushes gates to exactly 0 |
| Total Loss | `CrossEntropy + λ × SparsityLoss` |
| λ sweep | Low (0.5) / Medium (2.0) / High (5.0) to show sparsity–accuracy trade-off |

## Results

| Lambda (λ) | Test Accuracy | Sparsity (%) |
|:----------:|:-------------:|:------------:|
| 0.5 | 60.91% | 93.32% |
| 2.0 | 60.78% | 93.98% |
| 5.0 | 60.41% | 94.84% |

## Hardware

Auto-detects CUDA → MPS → CPU. 50 epochs on M3 Mac ≈ 7 min per run.