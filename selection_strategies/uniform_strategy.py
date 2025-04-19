# selection_strategies/uniform.py

import torch
from .base_strategy import BaseSelectionStrategy

class UniformStrategy(BaseSelectionStrategy):
    """
    Uniform data selection strategy (baseline).
    All examples have equal probability of selection.
    """
    
    def update_weights(self, dataset, model_outputs, epoch, global_step) -> torch.Tensor:
        """
        Update weights to be uniform across all examples.
        
        Args:
            dataset: Dataset to update weights for
            model_outputs: Dictionary with model outputs (not used)
            epoch: Current epoch (not used)
            global_step: Current global step (not used)
            
        Returns:
            Tensor of uniform weights
        """
        # Assign equal weight to all examples
        uniform_weights = torch.ones(len(dataset)) / len(dataset)
        return uniform_weights
    
    def get_selection_sensitivity(self) -> float:
        """
        Calculate the selection sensitivity (Δ_Q) of uniform selection.
        For uniform selection, Δ_Q = 0 since the selection distribution
        does not depend on the dataset.
        
        Returns:
            Selection sensitivity (0 for uniform)
        """
        return 0.0