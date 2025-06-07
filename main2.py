"""
Experiment script for adaptive selection strategy analysis.

This script runs experiments to:
1. Compare selection strategies and measure empirical selection sensitivity
2. Run experiments with varying sample sizes
3. Generate plots comparing theoretical and empirical results
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import logging
from tqdm import tqdm

# Import project modules (adjust paths if needed)
from data.data_loader import MultiNLIData, NLIDataset, set_seed
from models.model import RoBERTaForNLI
from selection_strategies.uniform_strategy import UniformStrategy
from selection_strategies.importance_weighting import ImportanceWeightingStrategy
from selection_strategies.confidence_sampling import ConfidenceSamplingStrategy
from selection_strategies.curriculum import CurriculumLearningStrategy
from evaluation.metrics import ReplicabilityMetrics


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def compute_empirical_selection_sensitivity(strategy, dataset, model, n_samples=100):
    """
    Compute empirical selection sensitivity by perturbing the dataset.
    
    Args:
        strategy: Selection strategy object
        dataset: Dataset object
        model: Model object
        n_samples: Number of perturbations to perform
        
    Returns:
        Empirical selection sensitivity (Δ_Q)
    """
    logger.info(f"Computing empirical selection sensitivity for {strategy.__class__.__name__}")
    
    # Get initial weights
    initial_outputs = model._get_outputs_for_dataset(dataset, batch_size=32)
    initial_weights = strategy.update_weights(
        dataset=dataset,
        model_outputs=initial_outputs,
        epoch=0,
        global_step=0
    )
    
    sensitivities = []
    n_examples = len(dataset)
    
    # Perform n_samples perturbations
    for i in tqdm(range(n_samples)):
        # Create perturbed dataset by replacing one example
        perturbed_dataset = dataset
        idx_to_replace = np.random.randint(0, n_examples)
        idx_replacement = np.random.randint(0, n_examples)
        
        # Get weights for perturbed dataset
        perturbed_outputs = initial_outputs.copy()  # Assuming shallow copy is sufficient
        
        # Update the outputs for the replaced example
        example = dataset[idx_replacement]
        perturbed_outputs["predictions"][idx_to_replace] = perturbed_outputs["predictions"][idx_replacement]
        perturbed_outputs["confidence"][idx_to_replace] = perturbed_outputs["confidence"][idx_replacement]
        perturbed_outputs["loss"][idx_to_replace] = perturbed_outputs["loss"][idx_replacement]
        
        perturbed_weights = strategy.update_weights(
            dataset=perturbed_dataset,
            model_outputs=perturbed_outputs,
            epoch=0,
            global_step=0
        )
        
        # Compute total variation distance
        tv_distance = 0.5 * torch.sum(torch.abs(initial_weights - perturbed_weights)).item()
        sensitivities.append(tv_distance)
    
    return np.mean(sensitivities)


def run_strategy_experiments(output_dir, num_runs=10, base_seed=42):
    """
    Run experiments for all selection strategies and compute metrics.
    
    Args:
        output_dir: Directory to save results
        num_runs: Number of runs per strategy
        base_seed: Base random seed
        
    Returns:
        Dictionary with results
    """
    logger.info("Running strategy experiments")
    
    # Initialize data handler
    data_handler = MultiNLIData(model_name="roberta-base", max_length=128)
    data_handler.load_data()
    
    # Create strategies
    strategies = {
        "uniform": UniformStrategy(seed=base_seed),
        "importance_weighting": ImportanceWeightingStrategy(weight_by="genre", smoothing_factor=0.8, seed=base_seed),
        "confidence_sampling": ConfidenceSamplingStrategy(temperature=0.2, min_weight=0.1, max_weight=10.0, seed=base_seed),
        "curriculum": CurriculumLearningStrategy(difficulty_metric="loss", start_ratio=0.25, end_ratio=1.0, pace="exp", seed=base_seed)
    }
    
    # Theoretical selection sensitivities
    theoretical_sensitivities = {
        "uniform": 0.0,
        "importance_weighting": 0.625,  # 1/2λ with λ=0.8
        "confidence_sampling": 5.0,     # 1/τ with τ=0.2
        "curriculum": 6.67              # Based on estimates
    }
    
    results = {
        "strategies": {},
        "theoretical_sensitivities": theoretical_sensitivities,
        "empirical_sensitivities": {},
        "pretraining": {
            "direct": {},
            "two_phase": {}
        }
    }
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, "results")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Prepare data splits
    data_splits = data_handler.prepare_data_splits(
        source_sample_size=15000,
        target_sample_size=6000,
        seed=base_seed
    )
    
    # Create datasets
    source_dataset = NLIDataset(data_splits["source_train"])
    target_dataset = NLIDataset(data_splits["target_train"])
    
    # First, calculate empirical selection sensitivity for each strategy
    logger.info("Calculating empirical selection sensitivities")
    base_model = RoBERTaForNLI(model_name="roberta-base", seed=base_seed)
    
    for strategy_name, strategy in strategies.items():
        sensitivity = compute_empirical_selection_sensitivity(
            strategy=strategy,
            dataset=target_dataset,
            model=base_model,
            n_samples=100
        )
        results["empirical_sensitivities"][strategy_name] = sensitivity
        logger.info(f"Strategy: {strategy_name}, Empirical Δ_Q: {sensitivity:.4f}, Theoretical Δ_Q: {theoretical_sensitivities[strategy_name]:.4f}")
    
    # Run direct fine-tuning experiments
    logger.info("Running direct fine-tuning experiments")
    direct_ft_results = {}
    
    for strategy_name, strategy in strategies.items():
        logger.info(f"Strategy: {strategy_name}")
        strategy_results = {
            "accuracies": [],
            "matched_accuracies": [],
            "mismatched_accuracies": []
        }
        
        for run_idx in range(num_runs):
            run_seed = base_seed + run_idx
            set_seed(run_seed)
            
            # Create model
            model = RoBERTaForNLI(model_name="roberta-base", seed=run_seed)
            
            # Set selection strategy
            model.set_selection_strategy(strategy)
            
            # Train model on target data
            logger.info(f"Run {run_idx+1}/{num_runs}, Seed: {run_seed}")
            training_history = model.train(
                train_dataset=target_dataset,
                eval_matched_dataset=data_splits["matched_dev"],
                eval_mismatched_dataset=data_splits["mismatched_dev"],
                batch_size=32,
                num_epochs=3,
                learning_rate=2e-5
            )
            
            # Evaluate
            eval_results = model.evaluate(
                matched_dataset=data_splits["matched_dev"],
                mismatched_dataset=data_splits["mismatched_dev"],
                batch_size=32
            )
            
            # Store results
            strategy_results["accuracies"].append(eval_results["mismatched_accuracy"])
            strategy_results["matched_accuracies"].append(eval_results["matched_accuracy"])
            strategy_results["mismatched_accuracies"].append(eval_results["mismatched_accuracy"])
        
        # Calculate statistics
        mismatched_accuracies = np.array(strategy_results["mismatched_accuracies"])
        failure_rate = ReplicabilityMetrics.compute_replicability_failure_rate(
            mismatched_accuracies, epsilon=0.01
        )
        
        direct_ft_results[strategy_name] = {
            "mean": float(np.mean(mismatched_accuracies)),
            "std": float(np.std(mismatched_accuracies)),
            "failure_rate": failure_rate,
            "accuracies": strategy_results["accuracies"]
        }
        
        logger.info(f"Strategy: {strategy_name}, Mean: {direct_ft_results[strategy_name]['mean']:.4f}, Std: {direct_ft_results[strategy_name]['std']:.4f}, Failure Rate: {failure_rate:.4f}")
    
    results["pretraining"]["direct"] = direct_ft_results
    
    # Run two-phase fine-tuning experiments
    logger.info("Running two-phase fine-tuning experiments")
    two_phase_ft_results = {}
    
    # First, pretrain a base model on source domain
    base_model = RoBERTaForNLI(model_name="roberta-base", seed=base_seed)
    base_model.train(
        train_dataset=source_dataset,
        eval_matched_dataset=data_splits["matched_dev"],
        eval_mismatched_dataset=data_splits["mismatched_dev"],
        batch_size=32,
        num_epochs=3,
        learning_rate=2e-5
    )
    pretrained_model_path = os.path.join(output_dir, "pretrained_model")
    os.makedirs(pretrained_model_path, exist_ok=True)
    base_model.save_model(pretrained_model_path)
    
    for strategy_name, strategy in strategies.items():
        logger.info(f"Strategy: {strategy_name}")
        strategy_results = {
            "accuracies": [],
            "matched_accuracies": [],
            "mismatched_accuracies": []
        }
        
        for run_idx in range(num_runs):
            run_seed = base_seed + run_idx
            set_seed(run_seed)
            
            # Create model from pretrained checkpoint
            model = RoBERTaForNLI(model_name="roberta-base", seed=run_seed)
            model.load_model(pretrained_model_path)
            
            # Set selection strategy
            model.set_selection_strategy(strategy)
            
            # Fine-tune on target data
            logger.info(f"Run {run_idx+1}/{num_runs}, Seed: {run_seed}")
            training_history = model.train(
                train_dataset=target_dataset,
                eval_matched_dataset=data_splits["matched_dev"],
                eval_mismatched_dataset=data_splits["mismatched_dev"],
                batch_size=32,
                num_epochs=3,
                learning_rate=2e-5
            )
            
            # Evaluate
            eval_results = model.evaluate(
                matched_dataset=data_splits["matched_dev"],
                mismatched_dataset=data_splits["mismatched_dev"],
                batch_size=32
            )
            
            # Store results
            strategy_results["accuracies"].append(eval_results["mismatched_accuracy"])
            strategy_results["matched_accuracies"].append(eval_results["matched_accuracy"])
            strategy_results["mismatched_accuracies"].append(eval_results["mismatched_accuracy"])
        
        # Calculate statistics
        mismatched_accuracies = np.array(strategy_results["mismatched_accuracies"])
        failure_rate = ReplicabilityMetrics.compute_replicability_failure_rate(
            mismatched_accuracies, epsilon=0.01
        )
        
        two_phase_ft_results[strategy_name] = {
            "mean": float(np.mean(mismatched_accuracies)),
            "std": float(np.std(mismatched_accuracies)),
            "failure_rate": failure_rate,
            "accuracies": strategy_results["accuracies"]
        }
        
        logger.info(f"Strategy: {strategy_name}, Mean: {two_phase_ft_results[strategy_name]['mean']:.4f}, Std: {two_phase_ft_results[strategy_name]['std']:.4f}, Failure Rate: {failure_rate:.4f}")
    
    results["pretraining"]["two_phase"] = two_phase_ft_results
    
    # Save results
    results_path = os.path.join(results_dir, "strategy_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    generate_strategy_plots(results, plots_dir)
    
    return results


def run_sample_size_experiments(output_dir, sample_sizes=None, num_runs=10, base_seed=42, epsilon=0.01):
    """
    Run experiments with varying sample sizes.
    
    Args:
        output_dir: Directory to save results
        sample_sizes: List of sample sizes to test
        num_runs: Number of runs per configuration
        base_seed: Base random seed
        epsilon: Threshold for replicability failure
        
    Returns:
        Dictionary with results
    """
    logger.info("Running sample size experiments")
    
    if sample_sizes is None:
        sample_sizes = [1000, 2000, 3000, 5000, 7000, 10000]
    
    # Initialize data handler
    data_handler = MultiNLIData(model_name="roberta-base", max_length=128)
    data_handler.load_data()
    
    # Create strategies
    strategies = {
        "uniform": UniformStrategy(seed=base_seed),
        "importance_weighting": ImportanceWeightingStrategy(weight_by="genre", smoothing_factor=0.8, seed=base_seed),
        "confidence_sampling": ConfidenceSamplingStrategy(temperature=0.2, min_weight=0.1, max_weight=10.0, seed=base_seed),
        "curriculum": CurriculumLearningStrategy(difficulty_metric="loss", start_ratio=0.25, end_ratio=1.0, pace="exp", seed=base_seed)
    }
    
    # Theoretical selection sensitivities
    theoretical_sensitivities = {
        "uniform": 0.0,
        "importance_weighting": 0.625,  # 1/2λ with λ=0.8
        "confidence_sampling": 5.0,     # 1/τ with τ=0.2
        "curriculum": 6.67              # Based on estimates
    }
    
    # Constants for theoretical bound
    c = 1.0  # Scaling constant
    
    results = {
        "sample_sizes": sample_sizes,
        "theoretical_sensitivities": theoretical_sensitivities,
        "strategies": {},
        "theoretical_bounds": {}
    }
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, "results")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Train a source-domain model to use for two-phase fine-tuning
    logger.info("Training source domain model")
    data_splits = data_handler.prepare_data_splits(
        source_sample_size=15000,
        target_sample_size=10000,  # Using largest size to get full dataset
        seed=base_seed
    )
    source_dataset = NLIDataset(data_splits["source_train"])
    
    base_model = RoBERTaForNLI(model_name="roberta-base", seed=base_seed)
    base_model.train(
        train_dataset=source_dataset,
        eval_matched_dataset=data_splits["matched_dev"],
        eval_mismatched_dataset=data_splits["mismatched_dev"],
        batch_size=32,
        num_epochs=3,
        learning_rate=2e-5
    )
    pretrained_model_path = os.path.join(output_dir, "source_pretrained_model")
    os.makedirs(pretrained_model_path, exist_ok=True)
    base_model.save_model(pretrained_model_path)
    
    # Run experiments for each strategy and sample size
    for strategy_name, strategy in strategies.items():
        logger.info(f"Strategy: {strategy_name}")
        results["strategies"][strategy_name] = {}
        
        # Calculate theoretical bound for each sample size
        theoretical_bounds = []
        for sample_size in sample_sizes:
            bound = 2 * np.exp(-(epsilon**2 * sample_size) / (2 * c**2 * (theoretical_sensitivities[strategy_name]**2 + 1e-10)))
            theoretical_bounds.append(float(bound))
        
        results["theoretical_bounds"][strategy_name] = theoretical_bounds
        
        # Run experiments for each sample size
        for sample_size in sample_sizes:
            logger.info(f"Sample size: {sample_size}")
            
            # Prepare data with this sample size
            target_data_splits = data_handler.prepare_data_splits(
                source_sample_size=15000,
                target_sample_size=sample_size,
                seed=base_seed
            )
            target_dataset = NLIDataset(target_data_splits["target_train"])
            
            run_results = {
                "accuracies": [],
                "failure_rates": []
            }
            
            for run_idx in range(num_runs):
                run_seed = base_seed + run_idx
                set_seed(run_seed)
                
                # Create model from pretrained checkpoint
                model = RoBERTaForNLI(model_name="roberta-base", seed=run_seed)
                model.load_model(pretrained_model_path)
                
                # Set selection strategy
                model.set_selection_strategy(strategy)
                
                # Fine-tune on target data
                logger.info(f"Run {run_idx+1}/{num_runs}, Seed: {run_seed}")
                training_history = model.train(
                    train_dataset=target_dataset,
                    eval_matched_dataset=data_splits["matched_dev"],
                    eval_mismatched_dataset=data_splits["mismatched_dev"],
                    batch_size=32,
                    num_epochs=3,
                    learning_rate=2e-5
                )
                
                # Evaluate
                eval_results = model.evaluate(
                    matched_dataset=data_splits["matched_dev"],
                    mismatched_dataset=data_splits["mismatched_dev"],
                    batch_size=32
                )
                
                # Store accuracy
                run_results["accuracies"].append(eval_results["mismatched_accuracy"])
            
            # Calculate statistics
            accuracies = np.array(run_results["accuracies"])
            failure_rate = ReplicabilityMetrics.compute_replicability_failure_rate(
                accuracies, epsilon=epsilon
            )
            
            results["strategies"][strategy_name][str(sample_size)] = {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies)),
                "failure_rate": failure_rate,
                "accuracies": run_results["accuracies"]
            }
            
            logger.info(f"Sample size: {sample_size}, Mean: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}, Failure Rate: {failure_rate:.4f}")
    
    # Save results
    results_path = os.path.join(results_dir, "sample_size_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    generate_sample_size_plots(results, plots_dir, epsilon)
    
    return results


def generate_strategy_plots(results, plots_dir):
    """
    Generate plots for strategy comparison.
    
    Args:
        results: Dictionary with experiment results
        plots_dir: Directory to save plots
    """
    logger.info("Generating strategy comparison plots")
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Empirical vs Theoretical selection sensitivity
    plt.figure(figsize=(10, 6))
    
    strategies = list(results["theoretical_sensitivities"].keys())
    theoretical_values = [results["theoretical_sensitivities"][s] for s in strategies]
    empirical_values = [results["empirical_sensitivities"][s] for s in strategies]
    
    bars = np.arange(len(strategies))
    width = 0.35
    
    plt.bar(bars - width/2, theoretical_values, width, label='Theoretical', color='b', alpha=0.6)
    plt.bar(bars + width/2, empirical_values, width, label='Empirical', color='r', alpha=0.6)
    
    plt.xlabel('Selection Strategy', fontsize=12)
    plt.ylabel('Selection Sensitivity (Δ_Q)', fontsize=12)
    plt.title('Theoretical vs Empirical Selection Sensitivity', fontsize=14)
    plt.xticks(bars, strategies)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'sensitivity_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 2: Performance vs Replicability (Direct Fine-tuning)
    plt.figure(figsize=(10, 6))
    
    direct_ft = results["pretraining"]["direct"]
    strategies = list(direct_ft.keys())
    
    for strategy in strategies:
        plt.scatter(direct_ft[strategy]["failure_rate"], direct_ft[strategy]["mean"], 
                   label=strategy, s=100)
    
    plt.xlabel('Replicability Failure Rate (lower is better)', fontsize=12)
    plt.ylabel('Accuracy (higher is better)', fontsize=12)
    plt.title('Performance vs Replicability Trade-off (Direct Fine-tuning)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'performance_replicability_direct.png'), dpi=300)
    plt.close()
    
    # Plot 3: Performance vs Replicability (Two-Phase Fine-tuning)
    plt.figure(figsize=(10, 6))
    
    two_phase_ft = results["pretraining"]["two_phase"]
    strategies = list(two_phase_ft.keys())
    
    for strategy in strategies:
        plt.scatter(two_phase_ft[strategy]["failure_rate"], two_phase_ft[strategy]["mean"], 
                   label=strategy, s=100)
    
    plt.xlabel('Replicability Failure Rate (lower is better)', fontsize=12)
    plt.ylabel('Accuracy (higher is better)', fontsize=12)
    plt.title('Performance vs Replicability Trade-off (Two-Phase Fine-tuning)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'performance_replicability_two_phase.png'), dpi=300)
    plt.close()
    
    # Plot 4: Direct vs Two-Phase comparison
    plt.figure(figsize=(10, 6))
    
    # Prepare data for grouped bar chart
    direct_means = [direct_ft[s]["mean"] for s in strategies]
    two_phase_means = [two_phase_ft[s]["mean"] for s in strategies]
    direct_failure_rates = [direct_ft[s]["failure_rate"] for s in strategies]
    two_phase_failure_rates = [two_phase_ft[s]["failure_rate"] for s in strategies]
    
    # Create subplots for accuracy and failure rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    bars = np.arange(len(strategies))
    width = 0.35
    
    # Accuracy comparison
    ax1.bar(bars - width/2, direct_means, width, label='Direct FT', color='b', alpha=0.6)
    ax1.bar(bars + width/2, two_phase_means, width, label='Two-Phase FT', color='g', alpha=0.6)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Performance Comparison', fontsize=14)
    ax1.set_xticks(bars)
    ax1.set_xticklabels(strategies)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Failure rate comparison
    ax2.bar(bars - width/2, direct_failure_rates, width, label='Direct FT', color='b', alpha=0.6)
    ax2.bar(bars + width/2, two_phase_failure_rates, width, label='Two-Phase FT', color='g', alpha=0.6)
    ax2.set_ylabel('Replicability Failure Rate', fontsize=12)
    ax2.set_title('Replicability Comparison', fontsize=14)
    ax2.set_xticks(bars)
    ax2.set_xticklabels(strategies)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'direct_vs_two_phase.png'), dpi=300)
    plt.close()


def generate_sample_size_plots(results, plots_dir, epsilon):
    """
    Generate plots for sample size experiments.
    
    Args:
        results: Dictionary with experiment results
        plots_dir: Directory to save plots
        epsilon: Threshold used for replicability failure
    """
    logger.info("Generating sample size experiment plots")
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    sample_sizes = results["sample_sizes"]
    strategies = list(results["strategies"].keys())
    
    # Plot 1: Failure rate vs sample size for each strategy
    plt.figure(figsize=(10, 6))
    
    for strategy in strategies:
        # Extract empirical failure rates
        empirical_rates = [results["strategies"][strategy][str(size)]["failure_rate"] for size in sample_sizes]
        
        # Get theoretical bounds
        theoretical_bounds = results["theoretical_bounds"][strategy]
        
        # Plot both empirical and theoretical
        plt.plot(sample_sizes, empirical_rates, 'o-', label=f'{strategy} (Empirical)')
        plt.plot(sample_sizes, theoretical_bounds, '--', label=f'{strategy} (Theoretical)', alpha=0.6)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Size (log scale)', fontsize=12)
    plt.ylabel('Replicability Failure Rate (log scale)', fontsize=12)
    plt.title(f'Replicability Failure Rate vs Sample Size (ε = {epsilon})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'failure_rate_vs_sample_size.png'), dpi=300)
    plt.close()
    
    # Plot 2: Minimum sample size required for target failure rate
    target_rates = [0.3, 0.2, 0.1, 0.05]
    required_samples = {rate: {} for rate in target_rates}
    
    for strategy in strategies:
        theoretical_sensitivity = results["theoretical_sensitivities"][strategy]
        
        for rate in target_rates:
            # Calculate required sample size from theoretical bound
            # ρ ≤ 2exp(-ε²n/2c²Δ_Q²) => n ≥ -2c²Δ_Q²ln(ρ/2)/ε²
            c = 1.0  # Scaling constant
            required_n = -2 * (c**2) * (theoretical_sensitivity**2 + 1e-10) * np.log(rate/2) / (epsilon**2)
            required_samples[rate][strategy] = required_n
    
    # Create bar chart of required sample sizes
    plt.figure(figsize=(12, 8))
    
    for i, rate in enumerate(target_rates):
        plt.subplot(2, 2, i+1)
        
        strategy_samples = [required_samples[rate][s] for s in strategies]
        plt.bar(strategies, strategy_samples, alpha=0.7)
        plt.yscale('log')
        plt.ylabel('Required Sample Size (log)', fontsize=10)
        plt.title(f'Failure Rate ≤ {rate}', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'required_sample_size.png'), dpi=300)
    plt.close()
    
    # Plot 3: Selection sensitivity vs minimum sample size
    plt.figure(figsize=(10, 6))
    
    sensitivities = [results["theoretical_sensitivities"][s] for s in strategies]
    samples_for_01 = [required_samples[0.1][s] for s in strategies]
    
    plt.scatter(sensitivities, samples_for_01, s=100)
    
    # Add strategy labels
    for i, strategy in enumerate(strategies):
        plt.annotate(strategy, (sensitivities[i], samples_for_01[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Add quadratic fit line
    if len(sensitivities) > 1:  # Only if we have more than one point
        x_fit = np.linspace(min(sensitivities), max(sensitivities), 100)
        # n ∝ Δ_Q²
        coef = np.polyfit(sensitivities, samples_for_01, 2)
        y_fit = coef[0] * x_fit**2 + coef[1] * x_fit + coef[2]
        plt.plot(x_fit, y_fit, 'r--', alpha=0.7, label='Quadratic Fit')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Selection Sensitivity (Δ_Q) (log scale)', fontsize=12)
    plt.ylabel('Required Sample Size for ρ ≤ 0.1 (log scale)', fontsize=12)
    plt.title('Sample Size Requirement vs Selection Sensitivity', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'sensitivity_vs_sample_size.png'), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run replicability experiments")
    parser.add_argument("--output_dir", type=str, default="./results2", help="Directory to save results")
    parser.add_argument("--strategy_experiments", action="store_true", help="Run strategy comparison experiments")
    parser.add_argument("--sample_size_experiments", action="store_true", help="Run sample size experiments")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs per experiment")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"experiments_{timestamp}")
    
    logger.info(f"Starting experiments with output directory: {output_dir}")
    
    if args.strategy_experiments:
        logger.info("Running strategy comparison experiments")
        strategy_results = run_strategy_experiments(
            output_dir=os.path.join(output_dir, "strategy_experiments"),
            num_runs=args.num_runs,
            base_seed=args.seed
        )
    
    if args.sample_size_experiments:
        logger.info("Running sample size experiments")
        sample_size_results = run_sample_size_experiments(
            output_dir=os.path.join(output_dir, "sample_size_experiments"),
            sample_sizes=[1000, 2000, 3000, 5000, 7000, 10000],
            num_runs=args.num_runs,
            base_seed=args.seed,
            epsilon=0.01
        )
    
    logger.info(f"Experiments completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()