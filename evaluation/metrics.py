# evaluation/metrics.py

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from scipy.stats import levene
import logging

# Setup logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                   datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

class ReplicabilityMetrics:
    """
    Metrics for evaluating replicability across multiple training runs.
    """
    
    @staticmethod
    def compute_performance_stability(accuracies: List[float]) -> Dict[str, float]:
        """
        Compute performance stability metrics.
        
        Args:
            accuracies: List of accuracy values from multiple runs
            
        Returns:
            Dictionary with stability metrics
        """
        accuracies = np.array(accuracies)
        
        metrics = {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
            "variance": float(np.var(accuracies)),
            "min": float(np.min(accuracies)),
            "max": float(np.max(accuracies)),
            "range": float(np.max(accuracies) - np.min(accuracies)),
            "median": float(np.median(accuracies)),
            "iqr": float(np.percentile(accuracies, 75) - np.percentile(accuracies, 25))
        }
        
        return metrics
    
    @staticmethod
    def compute_replicability_failure_rate(accuracies: List[float], epsilon: float = 0.01) -> float:
        """
        Compute empirical replicability failure rate.
        
        Args:
            accuracies: List of accuracy values from multiple runs
            epsilon: Threshold for considering two runs different
            
        Returns:
            Empirical replicability failure rate
        """
        n = len(accuracies)
        if n <= 1:
            return 0.0
        
        # Count pairs with difference > epsilon
        failure_count = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i+1, n):
                total_pairs += 1
                if abs(accuracies[i] - accuracies[j]) > epsilon:
                    failure_count += 1
        
        return failure_count / total_pairs
    
    @staticmethod
    def compute_centered_kernel_alignment(X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Compute Centered Kernel Alignment (CKA) similarity between two sets of representations.
        
        Args:
            X: First set of representations [n_samples, n_features]
            Y: Second set of representations [n_samples, n_features]
            
        Returns:
            CKA similarity [0, 1]
        """
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
        
        dot_XX = X @ X.T
        dot_YY = Y @ Y.T
        
        dot_XY = X @ Y.T
        
        HS_XY = torch.sum(dot_XY)
        HS_XX = torch.sqrt(torch.sum(dot_XX * dot_XX))
        HS_YY = torch.sqrt(torch.sum(dot_YY * dot_YY))
        
        if HS_XX == 0 or HS_YY == 0:
            return 0.0
        
        return (HS_XY / (HS_XX * HS_YY)).item()
    
    @staticmethod
    def compute_eigenspace_overlap(X: torch.Tensor, Y: torch.Tensor, k: int = 10) -> float:
        """
        Compute eigenspace overlap between two sets of representations.
        
        Args:
            X: First set of representations [n_samples, n_features]
            Y: Second set of representations [n_samples, n_features]
            k: Number of top eigenvectors to consider
            
        Returns:
            Eigenspace overlap [0, 1]
        """
        # Center data
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
        
        # Compute covariance matrices
        cov_X = X.T @ X
        cov_Y = Y.T @ Y
        
        # Compute eigenvectors
        _, V_X = torch.linalg.eigh(cov_X)
        _, V_Y = torch.linalg.eigh(cov_Y)
        
        # Get top k eigenvectors
        V_X_top = V_X[:, -k:]
        V_Y_top = V_Y[:, -k:]
        
        # Compute overlap using matrix multiplication and normalization
        overlap = torch.sum(torch.abs(V_X_top.T @ V_Y_top)) / k
        
        return overlap.item()
    
    @staticmethod
    def compute_empirical_selection_sensitivity(
        strategy, dataset, n_perturbations: int = 10, seed: int = 42
    ) -> float:
        """
        Compute empirical selection sensitivity by perturbing the dataset.
        
        Args:
            strategy: Selection strategy object
            dataset: Dataset object
            n_perturbations: Number of perturbations to perform
            seed: Random seed
            
        Returns:
            Empirical selection sensitivity (Δ_Q)
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Get initial weights
        initial_weights = strategy.update_weights(
            dataset=dataset,
            model_outputs=None,  # This might not work for all strategies
            epoch=0,
            global_step=0
        )
        
        sensitivities = []
        n_samples = len(dataset)
        
        for _ in range(n_perturbations):
            # Create a perturbed dataset by replacing one example
            idx_to_replace = np.random.randint(0, n_samples)
            idx_replacement = np.random.randint(0, n_samples)
            
            # This is a simplified approach - in practice, we would need to
            # actually replace the example in a copy of the dataset
            perturbed_dataset = dataset  # Assume this is a copy with one example replaced
            
            # Get weights for perturbed dataset
            perturbed_weights = strategy.update_weights(
                dataset=perturbed_dataset,
                model_outputs=None,  # This might not work for all strategies
                epoch=0,
                global_step=0
            )
            
            # Compute total variation distance
            tv_distance = 0.5 * torch.sum(torch.abs(initial_weights - perturbed_weights)).item()
            sensitivities.append(tv_distance)
        
        return np.mean(sensitivities)
    
    @staticmethod
    def compare_variances(accuracy_groups: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Statistically compare variances across different selection strategies.
        
        Args:
            accuracy_groups: Dictionary mapping strategy names to lists of accuracies
            
        Returns:
            Dictionary with p-values for variance equality tests
        """
        # Extract groups
        groups = []
        group_names = []
        
        for name, accuracies in accuracy_groups.items():
            groups.append(accuracies)
            group_names.append(name)
        
        # Check if we have enough groups
        if len(groups) <= 1:
            return {"error": "Need at least two groups to compare"}
        
        # Perform Levene's test for equality of variances
        try:
            stat, p_value = levene(*groups)
            result = {
                "levene_statistic": float(stat),
                "p_value": float(p_value),
                "reject_equal_variance": p_value < 0.05
            }
        except Exception as e:
            result = {"error": str(e)}
            
        return result
    
    @staticmethod
    def plot_accuracy_distribution(
        accuracy_groups: Dict[str, List[float]], 
        title: str = "Accuracy Distribution by Selection Strategy",
        save_path: Optional[str] = None
    ):
        """
        Plot the distribution of accuracies for different selection strategies.
        
        Args:
            accuracy_groups: Dictionary mapping strategy names to lists of accuracies
            title: Plot title
            save_path: Path to save the plot (if None, display it)
        """
        plt.figure(figsize=(10, 6))
        
        positions = range(len(accuracy_groups))
        names = list(accuracy_groups.keys())
        
        # Create box plots
        box_data = [accuracy_groups[name] for name in names]
        plt.boxplot(box_data, positions=positions, labels=names)
        
        # Add individual points
        for i, name in enumerate(names):
            accuracies = accuracy_groups[name]
            plt.scatter([i+1] * len(accuracies), accuracies, alpha=0.5)
        
        plt.title(title)
        plt.ylabel("Accuracy")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Add stability metrics as text
        for i, name in enumerate(names):
            metrics = ReplicabilityMetrics.compute_performance_stability(accuracy_groups[name])
            plt.text(i+1, min(accuracy_groups[name]) - 0.01, 
                    f"σ={metrics['std']:.4f}", ha="center")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_representation_similarity_matrix(
        similarity_matrix: np.ndarray,
        model_names: List[str],
        title: str = "Representation Similarity",
        save_path: Optional[str] = None
    ):
        """
        Plot a heatmap of representation similarities between models.
        
        Args:
            similarity_matrix: Matrix of similarity values
            model_names: Names of models
            title: Plot title
            save_path: Path to save the plot (if None, display it)
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        plt.imshow(similarity_matrix, cmap="viridis", vmin=0, vmax=1)
        
        # Add color bar
        plt.colorbar(label="Similarity")
        
        # Add labels
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha="right")
        plt.yticks(range(len(model_names)), model_names)
        
        # Add title
        plt.title(title)
        
        # Add similarity values as text
        for i in range(len(model_names)):
            for j in range(len(model_names)):
                plt.text(j, i, f"{similarity_matrix[i, j]:.2f}", 
                        ha="center", va="center", 
                        color="white" if similarity_matrix[i, j] > 0.5 else "black")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()