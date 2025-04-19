# selection_strategies/curriculum.py

import torch
import numpy as np
from .base_strategy import BaseSelectionStrategy

class CurriculumLearningStrategy(BaseSelectionStrategy):
    """
    Curriculum learning strategy.
    Introduces examples in order of increasing difficulty.
    """
    
    def __init__(self, difficulty_metric="loss", start_ratio=0.25, end_ratio=1.0, 
                 epochs_to_max=3, pace="linear", seed=42):
        """
        Initialize the curriculum learning strategy.
        
        Args:
            difficulty_metric: Metric to determine difficulty ("loss", "confidence")
            start_ratio: Initial fraction of data to use
            end_ratio: Final fraction of data to use
            epochs_to_max: Number of epochs to reach end_ratio
            pace: Pacing function ("linear", "exp", "log")
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)
        self.difficulty_metric = difficulty_metric
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.epochs_to_max = epochs_to_max
        self.pace = pace
        self.difficulty_scores = None
        self.sorted_indices = None
        
    def _compute_difficulty(self, model_outputs):
        """
        Compute difficulty scores for all examples.
        
        Args:
            model_outputs: Dictionary with model outputs
            
        Returns:
            Tensor of difficulty scores
        """
        if self.difficulty_metric == "loss":
            # Higher loss = more difficult
            return model_outputs["loss"]
        elif self.difficulty_metric == "confidence":
            # Lower confidence = more difficult
            return 1.0 - model_outputs["confidence"]
        else:
            raise ValueError(f"Unknown difficulty metric: {self.difficulty_metric}")
    
    def _calculate_pace_ratio(self, epoch):
        """
        Calculate the pacing ratio based on current epoch.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Current ratio of data to use
        """
        # Limit progress to epochs_to_max
        progress = min(epoch / self.epochs_to_max, 1.0)
        
        # Apply pacing function
        if self.pace == "linear":
            pace_ratio = progress
        elif self.pace == "exp":
            pace_ratio = np.exp(3 * progress - 3)
        elif self.pace == "log":
            # Ensure progress is not exactly 0 to avoid log(0)
            pace_ratio = np.log(1 + 9 * max(progress, 1e-10)) / np.log(10)
        else:
            pace_ratio = progress
            
        # Interpolate between start and end ratios
        current_ratio = self.start_ratio + pace_ratio * (self.end_ratio - self.start_ratio)
        return current_ratio
        
    def update_weights(self, dataset, model_outputs, epoch, global_step) -> torch.Tensor:
        """
        Update weights based on curriculum pacing.
        
        Args:
            dataset: Dataset to update weights for
            model_outputs: Dictionary with model outputs
            epoch: Current epoch
            global_step: Current global step
            
        Returns:
            Tensor of updated weights
        """
        # Compute difficulty scores if not already done
        if self.difficulty_scores is None or epoch == 0:
            self.difficulty_scores = self._compute_difficulty(model_outputs)
            self.sorted_indices = torch.argsort(self.difficulty_scores)
        
        # Calculate current curriculum ratio
        current_ratio = self._calculate_pace_ratio(epoch)
        
        # Determine number of examples to include
        num_examples = len(dataset)
        active_count = int(current_ratio * num_examples)
        
        # Initialize all weights to zero
        weights = torch.zeros(num_examples)
        
        # Set weights for included examples (easiest first)
        included_indices = self.sorted_indices[:active_count]
        weights[included_indices] = 1.0
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones_like(weights) / len(weights)
            
        return weights
    
    def get_selection_sensitivity(self) -> float:
        """
        Calculate the selection sensitivity (Δ_Q) of curriculum learning.
        
        Returns:
            Selection sensitivity estimate
        """
        # Curriculum learning has moderate sensitivity
        # Sensitivity depends on how quickly curriculum advances
        sensitivity = 0.5 / self.epochs_to_max
        return min(0.8, sensitivity)  # Cap at 0.8