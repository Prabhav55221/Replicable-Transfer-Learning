# main.py

"""
Main entry point for replicability experiments in transfer learning.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import os
import logging
import sys
import torch
import json
import random
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Import project modules
from data.data_loader import MultiNLIData, NLIDataset, set_seed
from models.model import RoBERTaForNLI
from selection_strategies.uniform_strategy import UniformStrategy
from selection_strategies.importance_weighting import ImportanceWeightingStrategy
from selection_strategies.confidence_sampling import ConfidenceSamplingStrategy
from selection_strategies.curriculum import CurriculumLearningStrategy
from evaluation.metrics import ReplicabilityMetrics
from experiments.experiment_runner import ExperimentRunner
from experiments.configs import load_config, save_config


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run replicability experiments for transfer learning")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--num_runs", type=int, default=None, help="Number of runs per strategy")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--only_viz", type=str, default=None, help="Do You Need Only Viz?")        
    
    args = parser.parse_args()

    if args.only_viz == 'True':
        logger.info("Creating experiment runner")
        experiment_name = 'replicability_transfer_learning_20250419_184158'

        # Load configuration
        config = load_config(args.config)

        if args.output_dir is not None:
            config["output_dir"] = args.output_dir
        if args.num_runs is not None:
            config["num_runs"] = args.num_runs
        if args.seed is not None:
            config["seed"] = args.seed

        # Initialize data handler
        logger.info("Initializing data handler")
        data_handler = MultiNLIData(
            model_name=config["model_name"],
            max_length=config["max_length"],
            cache_dir=config["cache_dir"]
        )
        data_handler.load_data()

        experiment_runner = ExperimentRunner(
            data_handler=data_handler,
            model_class=RoBERTaForNLI,
            strategies={},
            output_dir='/export/fs06/psingh54/Replicable-Transfer-Learning/results',
            num_runs=config["num_runs"],
            epsilon=config["epsilon"],
            metrics_class=ReplicabilityMetrics,
            seed=config["seed"]
        )
        
        # Extract training parameters
        training_kwargs = {
            "batch_size": config["batch_size"],
            "num_epochs": config["num_epochs"],
            "learning_rate": config["learning_rate"],
            "warmup_proportion": config["warmup_proportion"],
            "max_grad_norm": config["max_grad_norm"],
            "weight_decay": config["weight_decay"],
            "evaluation_steps": config["evaluation_steps"],
            "logging_steps": config["logging_steps"],
            "save_steps": config["save_steps"],
            "source_sample_size": config["source_sample_size"],
            "target_sample_size": config["target_sample_size"],
            "pretrain_on_source": config["pretrain_on_source"]
        }
        
        # Generate visualizations
        logger.info("Generating visualizations")
        experiment_runner.visualize_results()
        sys.exit()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.num_runs is not None:
        config["num_runs"] = args.num_runs
    if args.seed is not None:
        config["seed"] = args.seed
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['experiment_name']}_{timestamp}"
    output_dir = os.path.join(config["output_dir"], experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the configuration
    config_path = os.path.join(output_dir, "config.json")
    save_config(config, config_path)
    
    # Set global seed for reproducibility
    set_seed(config["seed"])
    
    # Initialize data handler
    logger.info("Initializing data handler")
    data_handler = MultiNLIData(
        model_name=config["model_name"],
        max_length=config["max_length"],
        cache_dir=config["cache_dir"]
    )
    data_handler.load_data()
    
    # Print dataset statistics
    stats = data_handler.get_data_stats()
    logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")
    
    # Generate dataset plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    data_handler.plot_genre_distribution(save_path=os.path.join(plots_dir, "genre_distribution.png"))
    
    # Initialize selection strategies based on config
    logger.info("Initializing selection strategies")
    strategies = {}
    
    for strategy_name, strategy_config in config["strategies"].items():
        strategy_type = strategy_config["type"]
        strategy_params = strategy_config["params"]
        
        if strategy_type == "uniform":
            strategies[strategy_name] = UniformStrategy(seed=config["seed"])
        elif strategy_type == "importance_weighting":
            strategies[strategy_name] = ImportanceWeightingStrategy(
                **strategy_params, seed=config["seed"]
            )
        elif strategy_type == "confidence_sampling":
            strategies[strategy_name] = ConfidenceSamplingStrategy(
                **strategy_params, seed=config["seed"]
            )
        elif strategy_type == "curriculum":
            strategies[strategy_name] = CurriculumLearningStrategy(
                **strategy_params, seed=config["seed"]
            )
        else:
            logger.warning(f"Unknown strategy type: {strategy_type}")
    
    # Create experiment runner
    logger.info("Creating experiment runner")
    experiment_runner = ExperimentRunner(
        data_handler=data_handler,
        model_class=RoBERTaForNLI,
        strategies=strategies,
        output_dir=output_dir,
        num_runs=config["num_runs"],
        epsilon=config["epsilon"],
        metrics_class=ReplicabilityMetrics,
        seed=config["seed"]
    )
    
    # Extract training parameters
    training_kwargs = {
        "batch_size": config["batch_size"],
        "num_epochs": config["num_epochs"],
        "learning_rate": config["learning_rate"],
        "warmup_proportion": config["warmup_proportion"],
        "max_grad_norm": config["max_grad_norm"],
        "weight_decay": config["weight_decay"],
        "evaluation_steps": config["evaluation_steps"],
        "logging_steps": config["logging_steps"],
        "save_steps": config["save_steps"],
        "source_sample_size": config["source_sample_size"],
        "target_sample_size": config["target_sample_size"],
        "pretrain_on_source": config["pretrain_on_source"]
    }
    
    # Run experiments
    logger.info("Starting experiments")
    experiment_runner.run_experiments(training_kwargs=training_kwargs)
    
    # Generate visualizations
    logger.info("Generating visualizations")
    experiment_runner.visualize_results()
    
    logger.info(f"Experiments completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()