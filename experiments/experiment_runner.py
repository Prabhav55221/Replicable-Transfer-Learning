# experiments/experiment_runner.py

import os
import torch
import numpy as np
import random
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# Setup logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                   datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    Runner for replicability experiments with multiple selection strategies.
    """
    
    def __init__(
        self,
        data_handler,
        model_class,
        strategies: Dict[str, Any],
        output_dir: str = "./results",
        num_runs: int = 5,
        epsilon: float = 0.01,
        metrics_class=None,
        seed: int = 42
    ):
        """
        Initialize the experiment runner.
        
        Args:
            data_handler: Data handler object
            model_class: Model class to use
            strategies: Dictionary mapping strategy names to strategy objects
            output_dir: Directory to save results
            num_runs: Number of runs per strategy
            epsilon: Threshold for replicability failure
            metrics_class: Class with metrics implementation
            seed: Base random seed
        """
        self.data_handler = data_handler
        self.model_class = model_class
        self.strategies = strategies
        self.output_dir = output_dir
        self.num_runs = num_runs
        self.epsilon = epsilon
        self.metrics_class = metrics_class
        self.base_seed = seed
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            "metadata": {
                "num_runs": num_runs,
                "epsilon": epsilon,
                "base_seed": seed,
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            },
            "strategies": {},
            "summary": {}
        }
        
        # Initialize model and representation storage
        self.models = {}
        self.representations = {}
    
    def set_seed(self, seed):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def run_experiments(
        self,
        training_kwargs: Dict[str, Any] = None
    ):
        """
        Run experiments for all strategies with multiple seeds.
        
        Args:
            training_kwargs: Additional keyword arguments for model training
        """
        logger.info(f"Starting experiments with {len(self.strategies)} strategies, {self.num_runs} runs each")
        
        if training_kwargs is None:
            training_kwargs = {}
        
        # Run experiments for each strategy
        for strategy_name, strategy in self.strategies.items():
            logger.info(f"Running experiments for strategy: {strategy_name}")
            
            # Initialize storage for this strategy
            self.results["strategies"][strategy_name] = {
                "runs": [],
                "accuracies": [],
                "matched_accuracies": [],
                "mismatched_accuracies": [],
                "selection_sensitivity": []
            }
            
            self.models[strategy_name] = []
            self.representations[strategy_name] = []
            
            # Run multiple times with different seeds
            for run_idx in range(self.num_runs):
                # Set seed for this run
                run_seed = self.base_seed + run_idx
                self.set_seed(run_seed)
                
                logger.info(f"Run {run_idx+1}/{self.num_runs} with seed {run_seed}")
                
                # Prepare data with this seed
                data_splits = self.data_handler.prepare_data_splits(
                    source_sample_size=training_kwargs.get("source_sample_size", None),
                    target_sample_size=training_kwargs.get("target_sample_size", None),
                    seed=run_seed
                )
                
                # Create datasets
                from torch.utils.data import Dataset
                
                class DummyDataset(Dataset):
                    def __init__(self, dataset):
                        self.dataset = dataset
                        self.selection_weights = torch.ones(len(dataset)) / len(dataset)
                    
                    def __len__(self):
                        return len(self.dataset)
                    
                    def __getitem__(self, idx):
                        return self.dataset[idx]
                        
                    def update_weights(self, new_weights):
                        self.selection_weights = new_weights
                    
                    def get_weighted_sampler(self):
                        from torch.utils.data import WeightedRandomSampler
                        weights = self.selection_weights.clone()
                        if weights.sum() > 0:
                            weights = weights / weights.sum()
                        else:
                            weights = torch.ones_like(weights) / len(weights)
                        return WeightedRandomSampler(weights, len(weights), replacement=True)
                
                source_dataset = DummyDataset(data_splits["source_train"])
                target_dataset = DummyDataset(data_splits["target_train"])
                
                # Create model
                model = self.model_class(seed=run_seed)
                
                # Set selection strategy
                strategy_instance = copy.deepcopy(strategy)  # Clone to avoid shared state
                model.set_selection_strategy(strategy_instance)
                
                # Train model
                run_output_dir = os.path.join(
                    self.output_dir, 
                    f"{strategy_name}_run{run_idx}"
                )
                
                # Combine training kwargs with run-specific settings
                run_training_kwargs = {
                    "output_dir": run_output_dir,
                    **training_kwargs
                }

                # Create a copy of training kwargs without experiment-specific parameters
                model_training_kwargs = {k: v for k, v in run_training_kwargs.items() 
                                        if k not in ["source_sample_size", "target_sample_size", "pretrain_on_source"]}
                
                # Train on source data first if pre-training is enabled
                if training_kwargs.get("pretrain_on_source", False):
                    logger.info("Pre-training on source domain")
                    model.train(
                        train_dataset=source_dataset,
                        eval_matched_dataset=data_splits["matched_dev"],
                        eval_mismatched_dataset=data_splits["mismatched_dev"],
                        **model_training_kwargs
                    )

                # Train on target data
                logger.info("Fine-tuning on target domain")
                training_history = model.train(
                    train_dataset=target_dataset,
                    eval_matched_dataset=data_splits["matched_dev"],
                    eval_mismatched_dataset=data_splits["mismatched_dev"],
                    **model_training_kwargs
                )
                
                # Evaluate final model
                eval_results = model.evaluate(
                    matched_dataset=data_splits["matched_dev"],
                    mismatched_dataset=data_splits["mismatched_dev"],
                    batch_size=training_kwargs.get("batch_size", 32)
                )
                
                # Extract representations for similarity analysis
                representations = model.extract_representations(
                    dataset=data_splits["mismatched_dev"],
                    batch_size=training_kwargs.get("batch_size", 32),
                    layer_index=-1
                )
                
                # Store results for this run
                run_results = {
                    "seed": run_seed,
                    "training_history": training_history,
                    "eval_results": eval_results,
                    "selection_sensitivity": strategy_instance.get_selection_sensitivity()
                }
                
                self.results["strategies"][strategy_name]["runs"].append(run_results)
                self.results["strategies"][strategy_name]["accuracies"].append(eval_results["mismatched_accuracy"])
                self.results["strategies"][strategy_name]["matched_accuracies"].append(eval_results["matched_accuracy"])
                self.results["strategies"][strategy_name]["mismatched_accuracies"].append(eval_results["mismatched_accuracy"])
                self.results["strategies"][strategy_name]["selection_sensitivity"].append(strategy_instance.get_selection_sensitivity())
                
                # Store model and representations
                self.models[strategy_name].append(model)
                self.representations[strategy_name].append(representations["representations"])
                
                # Save intermediate results
                self.save_results()
            
            # Compute summary metrics for this strategy
            self._compute_strategy_summary(strategy_name)
        
        # Compute cross-strategy comparisons
        self._compute_cross_strategy_comparisons()
        
        # Save final results
        self.save_results()
        
        logger.info("All experiments completed")
    
    def _compute_strategy_summary(self, strategy_name):
        """
        Compute summary metrics for a strategy.
        
        Args:
            strategy_name: Name of the strategy
        """
        # Get accuracies
        accuracies = self.results["strategies"][strategy_name]["accuracies"]
        
        # Compute performance stability metrics
        stability_metrics = self.metrics_class.compute_performance_stability(accuracies)
        
        # Compute empirical replicability failure rate
        failure_rate = self.metrics_class.compute_replicability_failure_rate(
            accuracies, self.epsilon
        )
        
        # Compute representation similarities
        representations = self.representations[strategy_name]
        n_runs = len(representations)
        similarity_matrix = np.zeros((n_runs, n_runs))
        
        for i in range(n_runs):
            for j in range(n_runs):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = self.metrics_class.compute_centered_kernel_alignment(
                        representations[i], representations[j]
                    )
        
        # Average pairwise similarity
        avg_similarity = np.sum(similarity_matrix) / (n_runs * n_runs - n_runs)
        
        # Store summary metrics
        summary = {
            "stability_metrics": stability_metrics,
            "replicability_failure_rate": failure_rate,
            "avg_representation_similarity": float(avg_similarity),
            "representation_similarity_matrix": similarity_matrix.tolist(),
            "avg_selection_sensitivity": np.mean(self.results["strategies"][strategy_name]["selection_sensitivity"])
        }
        
        self.results["summary"][strategy_name] = summary
    
    def _compute_cross_strategy_comparisons(self):
        """Compute cross-strategy comparison metrics."""
        strategy_names = list(self.strategies.keys())
        
        # Compare variances
        accuracy_groups = {
            name: self.results["strategies"][name]["accuracies"]
            for name in strategy_names
        }
        
        variance_comparison = self.metrics_class.compare_variances(accuracy_groups)
        
        # Compare mean performance
        mean_performances = {
            name: self.results["summary"][name]["stability_metrics"]["mean"]
            for name in strategy_names
        }
        
        # Compare replicability failure rates
        failure_rates = {
            name: self.results["summary"][name]["replicability_failure_rate"]
            for name in strategy_names
        }
        
        # Cross-strategy representation similarity
        cross_similarity = {}
        for name1 in strategy_names:
            for name2 in strategy_names:
                if name1 < name2:  # To avoid duplicate pairs
                    similarities = []
                    for i in range(self.num_runs):
                        for j in range(self.num_runs):
                            sim = self.metrics_class.compute_centered_kernel_alignment(
                                self.representations[name1][i],
                                self.representations[name2][j]
                            )
                            similarities.append(sim)
                    cross_similarity[f"{name1}_vs_{name2}"] = np.mean(similarities)
        
        # Store cross-strategy comparisons
        self.results["cross_strategy"] = {
            "variance_comparison": variance_comparison,
            "mean_performances": mean_performances,
            "failure_rates": failure_rates,
            "cross_representation_similarity": cross_similarity
        }
    
    def save_results(self):
        """Save current results to disk."""
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save results as JSON
        results_path = os.path.join(self.output_dir, "experiment_results.json")
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def visualize_results(self):
        """Visualize experiment results."""
        # Ensure we have results
        if not self.results["summary"]:
            logger.warning("No results to visualize")
            return
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Accuracy distribution plot
        accuracy_groups = {
            name: self.results["strategies"][name]["accuracies"]
            for name in self.strategies.keys()
        }
        
        self.metrics_class.plot_accuracy_distribution(
            accuracy_groups,
            title="Accuracy Distribution by Selection Strategy",
            save_path=os.path.join(plots_dir, "accuracy_distribution.png")
        )
        
        # 2. Representation similarity matrices
        for strategy_name in self.strategies.keys():
            sim_matrix = np.array(self.results["summary"][strategy_name]["representation_similarity_matrix"])
            model_names = [f"Run {i+1}" for i in range(self.num_runs)]
            
            self.metrics_class.plot_representation_similarity_matrix(
                sim_matrix,
                model_names,
                title=f"Representation Similarity - {strategy_name}",
                save_path=os.path.join(plots_dir, f"similarity_matrix_{strategy_name}.png")
            )
        
        # 3. Replicability vs. Performance plot
        plt.figure(figsize=(10, 6))
        
        x_values = []
        y_values = []
        labels = []
        
        for name in self.strategies.keys():
            x = self.results["summary"][name]["replicability_failure_rate"]
            y = self.results["summary"][name]["stability_metrics"]["mean"]
            x_values.append(x)
            y_values.append(y)
            labels.append(name)
        
        plt.scatter(x_values, y_values, s=100)
        
        for i, label in enumerate(labels):
            plt.annotate(
                label,
                (x_values[i], y_values[i]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left"
            )
        
        plt.xlabel("Replicability Failure Rate")
        plt.ylabel("Mean Accuracy")
        plt.title("Replicability vs. Performance Trade-off")
        plt.grid(linestyle="--", alpha=0.7)
        
        plt.savefig(os.path.join(plots_dir, "replicability_vs_performance.png"), dpi=300)
        plt.close()
        
        # 4. Selection Sensitivity vs. Replicability plot
        plt.figure(figsize=(10, 6))
        
        x_values = []
        y_values = []
        labels = []
        
        for name in self.strategies.keys():
            x = self.results["summary"][name]["avg_selection_sensitivity"]
            y = self.results["summary"][name]["replicability_failure_rate"]
            x_values.append(x)
            y_values.append(y)
            labels.append(name)
        
        plt.scatter(x_values, y_values, s=100)
        
        for i, label in enumerate(labels):
            plt.annotate(
                label,
                (x_values[i], y_values[i]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left"
            )
        
        plt.xlabel("Selection Sensitivity (Δ_Q)")
        plt.ylabel("Replicability Failure Rate")
        plt.title("Selection Sensitivity vs. Replicability")
        plt.grid(linestyle="--", alpha=0.7)
        
        plt.savefig(os.path.join(plots_dir, "sensitivity_vs_replicability.png"), dpi=300)
        plt.close()
        
        logger.info(f"Visualizations saved to {plots_dir}")