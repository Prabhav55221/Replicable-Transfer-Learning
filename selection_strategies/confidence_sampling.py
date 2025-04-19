# selection_strategies/confidence_sampling.py

import torch
import torch.nn.functional as F
from .base_strategy import BaseSelectionStrategy

class ConfidenceSamplingStrategy(BaseSelectionStrategy):
    """
    Confidence-based sampling strategy.
    Assigns higher weights to examples where the model has low confidence.
    """
    
    def __init__(self, temperature=1.0, min_weight=0.1, max_weight=10.0, seed=42):
        """
        Initialize the confidence sampling strategy.
        
        Args:
            temperature: Temperature parameter to control weight distribution
            min_weight: Minimum weight to assign to any example
            max_weight: Maximum weight to assign to any example
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)
        self.temperature = temperature
        self.min_weight = min_weight
        self.max_weight = max_weight
        
    def update_weights(self, dataset, model_outputs, epoch, global_step) -> torch.Tensor:
        """
        Update weights based on model confidence.
        
        Args:
            dataset: Dataset to update weights for
            model_outputs: Dictionary with model outputs
            epoch: Current epoch
            global_step: Current global step
            
        Returns:
            Tensor of updated weights
        """
        # Get confidence scores from model outputs
        confidence = model_outputs["confidence"]  # Shape: [num_examples]
        
        # Compute weights as inverse of confidence (higher weight for low confidence)
        # Apply temperature to control the sharpness of the distribution
        weights = (1.0 - confidence) ** (1.0 / self.temperature)
        
        # Clip weights to avoid extreme values
        weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones_like(weights) / len(weights)
            
        return weights
    
    def get_selection_sensitivity(self) -> float:
        """
        Calculate the selection sensitivity (Δ_Q) of confidence sampling.
        For confidence-based sampling, Δ_Q is high because selection depends
        strongly on model outputs which can change between runs.
        
        Returns:
            Selection sensitivity estimate
        """
        # Higher temperature means lower sensitivity
        return 1.0 / self.temperature