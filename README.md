# Replicability in Transfer Learning

This project investigates the replicability of transfer learning with adaptive data selection strategies.

## Project Structure

- `data/` - Data loading and preprocessing utilities
- `models/` - Model implementation for transfer learning
- `selection_strategies/` - Implementations of different selection strategies
- `evaluation/` - Replicability metrics and visualizations
- `experiments/` - Experiment runner and configurations
- `main.py` - Main entry point for running experiments
- `run_single_experiment.py` - Simplified script for testing single strategies

## Setup

1. Install dependencies:
```bash
pip install torch transformers datasets matplotlib tqdm scipy

python main.py --config path/to/config.json --output_dir ./results --num_runs 5

# Single
python run_single_experiment.py --strategy uniform --small
```