# experiments/configs.py

"""
Configuration definitions for replicability experiments.
"""

import json
from typing import Dict, Any, List, Optional
import os
import logging

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    # Experiment settings
    "experiment_name": "replicability_transfer_learning",
    "output_dir": "./results",
    "num_runs": 5,
    "epsilon": 0.01,
    "seed": 42,
    
    # Data settings
    "model_name": "roberta-base",
    "max_length": 128,
    "cache_dir": None,
    "source_sample_size": 5000,
    "target_sample_size": 2000,
    
    # Training settings
    "batch_size": 32,
    "num_epochs": 3,
    "learning_rate": 2e-5,
    "warmup_proportion": 0.1,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "evaluation_steps": 500,
    "logging_steps": 100,
    "save_steps": 1000,
    "pretrain_on_source": False,
    
    # Selection strategies
    "strategies": {
        "uniform": {
            "type": "uniform",
            "params": {}
        },
        "importance_weighting": {
            "type": "importance_weighting",
            "params": {
                "weight_by": "genre",
                "smoothing_factor": 0.1
            }
        },
        "confidence_sampling": {
            "type": "confidence_sampling",
            "params": {
                "temperature": 1.0,
                "min_weight": 0.1,
                "max_weight": 10.0
            }
        },
        "curriculum": {
            "type": "curriculum",
            "params": {
                "difficulty_metric": "loss",
                "start_ratio": 0.25,
                "end_ratio": 1.0,
                "epochs_to_max": 2,
                "pace": "linear"
            }
        }
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a file or use the default configuration.
    
    Args:
        config_path: Path to the configuration file (JSON)
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path is not None and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
            
            # Update config with user settings
            config.update(user_config)
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info("Using default configuration")
    
    return config

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save the configuration
    """
    try:
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving configuration to {output_path}: {e}")