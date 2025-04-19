# selection_strategies/base_strategy.py

import torch
import numpy as np
import random
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseSelectionStrategy(ABC):
    """
    Base class for adaptive data selection strategies.
    All selection strategies should inherit from this class.
    """
    
    def __init__(self, seed=42):
        """
        Initialize the selection strategy.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._set_seed(seed)
    
    def _set_seed(self, seed):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    @abstractmethod
    def update_weights(self, dataset, model_outputs, epoch, global_step) -> torch.Tensor:
        """
        Update selection weights based on model outputs.
        
        Args:
            dataset: Dataset to update weights for
            model_outputs: Dictionary with model outputs
            epoch: Current epoch
            global_step: Current global step
            
        Returns:
            Tensor of updated weights for each example
        """
        pass
    
    def get_selection_sensitivity(self) -> float:
        """
        Calculate the selection sensitivity (Δ_Q) of this strategy.
        Δ_Q measures how much the selection distribution changes when 
        the dataset changes slightly.
        
        Returns:
            Selection sensitivity value
        """
        # Default implementation - should be overridden by strategies 
        # that can compute their theoretical sensitivity
        return float('inf')