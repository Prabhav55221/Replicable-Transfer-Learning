# selection_strategies/importance_weighting.py

import torch
import numpy as np
from collections import Counter
from .base_strategy import BaseSelectionStrategy

class ImportanceWeightingStrategy(BaseSelectionStrategy):
    """
    Importance weighting strategy based on domain features.
    Assigns weights based on genre distribution or other metadata.
    """
    
    def __init__(self, weight_by="genre", target_distribution=None, smoothing_factor=0.1, seed=42):
        """
        Initialize the importance weighting strategy.
        
        Args:
            weight_by: Feature to weight by (e.g., "genre")
            target_distribution: Target distribution to match (dict mapping feature values to probabilities)
            smoothing_factor: Smoothing factor for the weights
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)
        self.weight_by = weight_by
        self.target_distribution = target_distribution
        self.smoothing_factor = smoothing_factor
        self.initialized = False
        
    def _initialize_weights(self, dataset):
        """
        Initialize weights based on feature distribution.
        
        Args:
            dataset: Dataset to initialize weights for
        """
        # Extract features from dataset
        features = [item[self.weight_by] for item in dataset]
        
        # Count feature occurrences
        feature_counts = Counter(features)
        total_count = len(features)
        
        # Calculate source distribution
        source_distribution = {feature: count / total_count 
                              for feature, count in feature_counts.items()}
        
        # Use provided target distribution or default to uniform
        if self.target_distribution is None:
            self.target_distribution = {feature: 1.0 / len(feature_counts) 
                                      for feature in feature_counts.keys()}
        
        # Calculate importance weights
        self.feature_weights = {}
        for feature, source_prob in source_distribution.items():
            target_prob = self.target_distribution.get(feature, 0.0)
            
            # Apply smoothing to avoid extreme weights
            weight = (target_prob + self.smoothing_factor) / (source_prob + self.smoothing_factor)
            self.feature_weights[feature] = weight
            
        self.initialized = True
        
    def update_weights(self, dataset, model_outputs, epoch, global_step) -> torch.Tensor:
        """
        Update weights based on feature importance.
        
        Args:
            dataset: Dataset to update weights for
            model_outputs: Dictionary with model outputs (not used)
            epoch: Current epoch (not used)
            global_step: Current global step (not used)
            
        Returns:
            Tensor of updated weights
        """
        # Initialize weights if not done yet
        if not self.initialized:
            self._initialize_weights(dataset)
        
        # Assign weights based on features
        weights = []
        for i in range(len(dataset)):
            item = dataset[i]
            feature_value = item[self.weight_by]
            weight = self.feature_weights.get(feature_value, 1.0)
            weights.append(weight)
        
        # Convert to tensor and normalize
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        if weights_tensor.sum() > 0:
            weights_tensor = weights_tensor / weights_tensor.sum()
        else:
            weights_tensor = torch.ones_like(weights_tensor) / len(weights_tensor)
            
        return weights_tensor
    
    def get_selection_sensitivity(self) -> float:
        """
        Calculate the selection sensitivity (Δ_Q) of importance weighting.
        This is a rough approximation based on the variance of the weights.
        
        Returns:
            Selection sensitivity estimate
        """
        if not self.initialized or not self.feature_weights:
            return 0.1  # Default value before initialization
            
        # Approximation based on weight variance
        weights = list(self.feature_weights.values())
        return float(np.std(weights) * 2)