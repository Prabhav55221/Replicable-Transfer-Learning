# run_single_experiment.py

"""
Script to run a single experiment for testing purposes.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import torch
import argparse
import json
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


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a single experiment for testing")
    parser.add_argument("--strategy", type=str, default="uniform", help="Selection strategy to use")
    parser.add_argument("--output_dir", type=str, default="./test_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--small", action="store_true", help="Use smaller dataset for quicker testing")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.strategy}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading data")
    data_handler = MultiNLIData(model_name="roberta-base", max_length=128)
    data_handler.load_data()
    
    # Sample sizes for testing
    source_sample_size = 1000 if args.small else 5000
    target_sample_size = 500 if args.small else 2000
    
    # Prepare data splits
    data_splits = data_handler.prepare_data_splits(
        source_sample_size=source_sample_size,
        target_sample_size=target_sample_size,
        seed=args.seed
    )
    
    # Create datasets
    source_dataset = NLIDataset(data_splits["source_train"])
    target_dataset = NLIDataset(data_splits["target_train"])
    
    # Initialize model
    logger.info("Initializing model")
    model = RoBERTaForNLI(model_name="roberta-base")
    
    # Initialize strategy
    logger.info(f"Using strategy: {args.strategy}")
    if args.strategy == "uniform":
        strategy = UniformStrategy(seed=args.seed)
    elif args.strategy == "importance_weighting":
        strategy = ImportanceWeightingStrategy(weight_by="genre", seed=args.seed)
    elif args.strategy == "confidence_sampling":
        strategy = ConfidenceSamplingStrategy(seed=args.seed)
    elif args.strategy == "curriculum":
        strategy = CurriculumLearningStrategy(seed=args.seed)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    # Set strategy
    model.set_selection_strategy(strategy)
    
    # Training parameters
    training_kwargs = {
        "batch_size": 16,
        "num_epochs": 2 if args.small else 3,
        "learning_rate": 2e-5,
        "evaluation_steps": 100 if args.small else 500,
        "logging_steps": 20 if args.small else 100
    }
    
    # Train model
    logger.info("Training model")
    training_history = model.train(
        train_dataset=target_dataset,
        eval_matched_dataset=data_splits["matched_dev"],
        eval_mismatched_dataset=data_splits["mismatched_dev"],
        output_dir=output_dir,
        **training_kwargs
    )
    
    # Evaluate model
    logger.info("Evaluating model")
    eval_results = model.evaluate(
        matched_dataset=data_splits["matched_dev"],
        mismatched_dataset=data_splits["mismatched_dev"],
        batch_size=training_kwargs["batch_size"]
    )
    
    # Save results
    results = {
        "strategy": args.strategy,
        "seed": args.seed,
        "eval_results": eval_results,
        "training_history": training_history
    }
    
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Visualize training history
    model.visualize_training_history(save_path=os.path.join(output_dir, "training_history.png"))
    
    if args.strategy != "uniform":
        model.visualize_selection_weights(save_path=os.path.join(output_dir, "selection_weights.png"))
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()