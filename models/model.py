"""
Model implementation for replicability experiments in transfer learning.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    RobertaForSequenceClassification, 
    RobertaConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
import logging
import json

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class RoBERTaForNLI:
    """
    RoBERTa model wrapper for Natural Language Inference tasks,
    with support for various adaptive selection strategies
    and replicability analysis.
    """
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 3,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        output_hidden_states: bool = True,
        seed: int = 42
        ):
        """
        Initialize the model.

        Args:
            model_name: Name of the pre-trained model to use
            num_labels: Number of labels for classification (3 for MultiNLI)
            device: Device to use (cpu or cuda)
            cache_dir: Directory to cache models
            output_hidden_states: Whether to output hidden states (needed for representation similarity)
            seed: Random seed for reproducibility
        """
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        self.model_name = model_name
        self.num_labels = num_labels
        self.cache_dir = cache_dir
        self.output_hidden_states = output_hidden_states
        
        # Determine device
        if device is None:
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current device: {torch.cuda.current_device()}")
                print(f"Device name: {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize the model
        self.config = RobertaConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=output_hidden_states,
            cache_dir=cache_dir
        )
        
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
            cache_dir=cache_dir
        )
        
        self.model.to(self.device)
        
        # Training history
        self.train_history = {
            "loss": [],
            "accuracy": [],
            "selection_weights": []
        }
        
        # Evaluation history
        self.eval_history = {
            "matched_accuracy": [],
            "mismatched_accuracy": [],
            "matched_loss": [],
            "mismatched_loss": []
        }
        
        # Selection strategy
        self.selection_strategy = None
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the model and its configuration.
        
        Args:
            output_dir: Directory to save the model
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"Saving model to {output_dir}")
        
        # Save model weights
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)
        
        # Save config
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f)
        
        # Save training history
        history_path = os.path.join(output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump({
                "train": self.train_history,
                "eval": self.eval_history
            }, f)
    
    def load_model(self, model_dir: str) -> None:
        """
        Load a saved model.
        
        Args:
            model_dir: Directory containing saved model
        """
        logger.info(f"Loading model from {model_dir}")
        
        # Load config
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            self.config = RobertaConfig.from_dict(config_dict)
        
        # Load model weights
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            self.model = RobertaForSequenceClassification(self.config)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
        
        # Load training history if available
        history_path = os.path.join(model_dir, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
                self.train_history = history.get("train", self.train_history)
                self.eval_history = history.get("eval", self.eval_history)
    
    def set_selection_strategy(self, strategy) -> None:
        """
        Set the adaptive selection strategy for training.
        
        Args:
            strategy: Selection strategy object
        """
        self.selection_strategy = strategy
    
    def train(
        self,
        train_dataset,
        eval_matched_dataset=None,
        eval_mismatched_dataset=None,
        batch_size: int = 32,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_proportion: float = 0.1,
        max_grad_norm: float = 1.0,
        weight_decay: float = 0.01,
        evaluation_steps: int = 500,
        logging_steps: int = 100,
        save_steps: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model using the specified selection strategy.
        
        Args:
            train_dataset: Training dataset (NLIDataset)
            eval_matched_dataset: Evaluation dataset for matched domain
            eval_mismatched_dataset: Evaluation dataset for mismatched domain
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_proportion: Proportion of steps for LR warmup
            max_grad_norm: Maximum gradient norm for clipping
            weight_decay: Weight decay for optimizer
            evaluation_steps: Evaluate every N steps
            logging_steps: Log every N steps
            save_steps: Save model every N steps (None to disable)
            output_dir: Directory to save model checkpoints
            
        Returns:
            Dictionary with training history
        """
        # Create optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create dataloader with initial uniform sampling
        if self.selection_strategy is None:
            # Use default uniform sampling
            dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            num_training_steps = len(dataloader) * num_epochs
        else:
            # Use weighted sampling based on selection strategy
            sampler = train_dataset.get_weighted_sampler()
            dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler
            )
            num_training_steps = len(dataloader) * num_epochs
        
        # Create learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_proportion * num_training_steps),
            num_training_steps=num_training_steps
        )
        
        # Training loop
        global_step = 0
        epoch_loss = 0.0
        self.model.train()
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            
            # Reset epoch stats
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            epoch_steps = 0
            
            for step, batch in enumerate(epoch_iterator):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "genre"}
                
                # Forward pass
                outputs = self.model(**{k: v for k, v in batch.items() 
                                     if k in ["input_ids", "attention_mask", "labels"]})
                loss = outputs.loss
                logits = outputs.logits
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update statistics
                global_step += 1
                epoch_loss += loss.item()
                
                # Calculate accuracy
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == batch["labels"]).float().mean().item()
                epoch_accuracy += accuracy
                epoch_steps += 1
                
                # Update progress bar
                epoch_iterator.set_postfix({
                    "loss": loss.item(),
                    "accuracy": accuracy
                })
                
                # Logging
                if global_step % logging_steps == 0:
                    logger.info(
                        f"Step {global_step}: loss = {loss.item():.4f}, "
                        f"accuracy = {accuracy:.4f}"
                    )
                    self.train_history["loss"].append(loss.item())
                    self.train_history["accuracy"].append(accuracy)
                
                # Evaluation
                if evaluation_steps > 0 and global_step % evaluation_steps == 0:
                    eval_results = self.evaluate(
                        eval_matched_dataset,
                        eval_mismatched_dataset,
                        batch_size=batch_size
                    )
                    self.model.train()  # Set back to training mode
                
                # Save checkpoint
                if save_steps is not None and output_dir is not None and global_step % save_steps == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    self.save_model(checkpoint_dir)
            
            # End of epoch
            avg_epoch_loss = epoch_loss / epoch_steps
            avg_epoch_accuracy = epoch_accuracy / epoch_steps
            
            logger.info(
                f"Epoch {epoch+1} completed: average loss = {avg_epoch_loss:.4f}, "
                f"average accuracy = {avg_epoch_accuracy:.4f}"
            )
            
            # Update selection weights if using adaptive strategy
            if self.selection_strategy is not None:
                logger.info("Updating selection weights using adaptive strategy")
                
                # Get model outputs for all examples for selection
                all_outputs = self._get_outputs_for_dataset(train_dataset, batch_size=batch_size)
                
                # Calculate new weights
                new_weights = self.selection_strategy.update_weights(
                    dataset=train_dataset,
                    model_outputs=all_outputs,
                    epoch=epoch,
                    global_step=global_step
                )
                
                # Store selection weight statistics
                weight_stats = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "mean": float(torch.mean(new_weights).item()),
                    "std": float(torch.std(new_weights).item()),
                    "min": float(torch.min(new_weights).item()),
                    "max": float(torch.max(new_weights).item()),
                    "entropy": float(self._compute_entropy(new_weights).item())
                }
                
                self.train_history["selection_weights"].append(weight_stats)
                
                # Update dataset weights
                train_dataset.update_weights(new_weights)
                
                # Update dataloader with new sampler
                sampler = train_dataset.get_weighted_sampler()
                dataloader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    sampler=sampler
                )
            
            # Evaluate at the end of each epoch
            if eval_matched_dataset is not None or eval_mismatched_dataset is not None:
                eval_results = self.evaluate(
                    eval_matched_dataset,
                    eval_mismatched_dataset,
                    batch_size=batch_size
                )
                self.model.train()  # Set back to training mode
        
        # Save final model
        if output_dir is not None:
            self.save_model(output_dir)
        
        return {
            "train": self.train_history,
            "eval": self.eval_history
        }
    
    def evaluate(
        self,
        matched_dataset=None,
        mismatched_dataset=None,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate the model on matched and/or mismatched datasets.
        
        Args:
            matched_dataset: Dataset for matched domain
            mismatched_dataset: Dataset for mismatched domain
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Evaluate on matched dataset
        if matched_dataset is not None:
            matched_results = self._evaluate_dataset(matched_dataset, batch_size, "matched")
            results.update(matched_results)
            
            # Add to history
            self.eval_history["matched_accuracy"].append(matched_results["matched_accuracy"])
            self.eval_history["matched_loss"].append(matched_results["matched_loss"])
        
        # Evaluate on mismatched dataset
        if mismatched_dataset is not None:
            mismatched_results = self._evaluate_dataset(mismatched_dataset, batch_size, "mismatched")
            results.update(mismatched_results)
            
            # Add to history
            self.eval_history["mismatched_accuracy"].append(mismatched_results["mismatched_accuracy"])
            self.eval_history["mismatched_loss"].append(mismatched_results["mismatched_loss"])
        
        # Log results
        log_str = "Evaluation results: "
        for key, value in results.items():
            log_str += f"{key} = {value:.4f}, "
        logger.info(log_str[:-2])  # Remove trailing comma and space
        
        return results
    
    def _evaluate_dataset(
        self,
        dataset,
        batch_size: int,
        prefix: str
    ) -> Dict[str, float]:
        """
        Evaluate the model on a single dataset.
        
        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size
            prefix: Prefix for metric names
            
        Returns:
            Dictionary with evaluation metrics
        """
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {prefix}"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "genre"}
                
                # Forward pass
                outputs = self.model(**{k: v for k, v in batch.items() 
                                      if k in ["input_ids", "attention_mask", "labels"]})
                loss = outputs.loss
                logits = outputs.logits
                
                # Update metrics
                total_loss += loss.item() * batch["input_ids"].size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        
        # Calculate metrics
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return {
            f"{prefix}_loss": avg_loss,
            f"{prefix}_accuracy": accuracy
        }
    
    def _get_outputs_for_dataset(
        self,
        dataset,
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """
        Get model outputs for all examples in a dataset.
        Used for adaptive selection strategies.
        
        Args:
            dataset: Dataset to get outputs for
            batch_size: Batch size
            
        Returns:
            Dictionary with model outputs for all examples
        """
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        all_logits = []
        all_labels = []
        all_loss = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "genre"}
                
                # Forward pass
                outputs = self.model(**{k: v for k, v in batch.items() 
                                      if k in ["input_ids", "attention_mask", "labels"]})
                loss = outputs.loss
                logits = outputs.logits
                
                # Collect outputs
                all_logits.append(logits.detach().cpu())
                all_labels.append(batch["labels"].detach().cpu())
                
                # Calculate per-example loss for later use
                per_example_loss = F.cross_entropy(
                    logits, 
                    batch["labels"], 
                    reduction="none"
                )
                all_loss.append(per_example_loss.detach().cpu())
        
        # Concatenate results
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_loss = torch.cat(all_loss, dim=0)
        
        # Calculate probabilities and confidence
        all_probs = F.softmax(all_logits, dim=1)
        all_confidence, all_preds = torch.max(all_probs, dim=1)
        
        return {
            "logits": all_logits,
            "probabilities": all_probs,
            "predictions": all_preds,
            "labels": all_labels,
            "loss": all_loss,
            "confidence": all_confidence
        }
    
    def extract_representations(
        self,
        dataset,
        batch_size: int = 32,
        layer_index: int = -1
    ) -> Dict[str, torch.Tensor]:
        """
        Extract model representations from a specific layer for replicability analysis.
        
        Args:
            dataset: Dataset to extract representations from
            batch_size: Batch size
            layer_index: Index of the layer to extract (-1 for last layer)
            
        Returns:
            Dictionary with model representations
        """
        if not self.output_hidden_states:
            raise ValueError("Model was initialized with output_hidden_states=False, "
                            "cannot extract representations")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        all_hidden_states = []
        all_labels = []
        all_preds = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting representations"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "genre"}
                
                # Forward pass
                outputs = self.model(**{k: v for k, v in batch.items() 
                                      if k in ["input_ids", "attention_mask"]})
                
                # Get hidden states
                hidden_states = outputs.hidden_states[layer_index]
                
                # Get CLS token representation
                cls_repr = hidden_states[:, 0, :]
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                # Collect outputs
                all_hidden_states.append(cls_repr.detach().cpu())
                all_labels.append(batch["labels"].detach().cpu())
                all_preds.append(preds.detach().cpu())
        
        # Concatenate results
        all_hidden_states = torch.cat(all_hidden_states, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        
        return {
            "representations": all_hidden_states,
            "labels": all_labels,
            "predictions": all_preds
        }
    
    def visualize_training_history(self, save_path=None):
        """
        Visualize training and evaluation history.
        
        Args:
            save_path: Path to save the plot (if None, display it)
        """
        if not self.train_history["loss"]:
            logger.warning("No training history to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(self.train_history["loss"])
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        
        # Training accuracy
        axes[0, 1].plot(self.train_history["accuracy"])
        axes[0, 1].set_title("Training Accuracy")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Accuracy")
        
        # Evaluation loss
        if self.eval_history["matched_loss"] or self.eval_history["mismatched_loss"]:
            if self.eval_history["matched_loss"]:
                axes[1, 0].plot(self.eval_history["matched_loss"], label="Matched")
            if self.eval_history["mismatched_loss"]:
                axes[1, 0].plot(self.eval_history["mismatched_loss"], label="Mismatched")
            axes[1, 0].set_title("Evaluation Loss")
            axes[1, 0].set_xlabel("Evaluation")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].legend()
        
        # Evaluation accuracy
        if self.eval_history["matched_accuracy"] or self.eval_history["mismatched_accuracy"]:
            if self.eval_history["matched_accuracy"]:
                axes[1, 1].plot(self.eval_history["matched_accuracy"], label="Matched")
            if self.eval_history["mismatched_accuracy"]:
                axes[1, 1].plot(self.eval_history["mismatched_accuracy"], label="Mismatched")
            axes[1, 1].set_title("Evaluation Accuracy")
            axes[1, 1].set_xlabel("Evaluation")
            axes[1, 1].set_ylabel("Accuracy")
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
    
    def visualize_selection_weights(self, save_path=None):
        """
        Visualize selection weight statistics over time.
        
        Args:
            save_path: Path to save the plot (if None, display it)
        """
        if not self.train_history["selection_weights"]:
            logger.warning("No selection weight history to visualize")
            return
        
        # Extract statistics
        epochs = [sw["epoch"] for sw in self.train_history["selection_weights"]]
        means = [sw["mean"] for sw in self.train_history["selection_weights"]]
        stds = [sw["std"] for sw in self.train_history["selection_weights"]]
        mins = [sw["min"] for sw in self.train_history["selection_weights"]]
        maxs = [sw["max"] for sw in self.train_history["selection_weights"]]
        entropies = [sw["entropy"] for sw in self.train_history["selection_weights"]]
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot weight statistics
        axes[0].plot(epochs, means, label="Mean")
        axes[0].fill_between(epochs, 
                           [m - s for m, s in zip(means, stds)], 
                           [m + s for m, s in zip(means, stds)], 
                           alpha=0.3)
        axes[0].plot(epochs, mins, label="Min", linestyle="--")
        axes[0].plot(epochs, maxs, label="Max", linestyle="--")
        axes[0].set_title("Selection Weight Statistics")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Weight Value")
        axes[0].legend()
        
        # Plot entropy
        axes[1].plot(epochs, entropies)
        axes[1].set_title("Selection Weight Entropy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Entropy")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _compute_entropy(self, weights):
        """
        Compute entropy of a weight distribution.
        
        Args:
            weights: Weight tensor
            
        Returns:
            Entropy value
        """
        # Normalize weights to sum to 1
        if torch.sum(weights) == 0:
            return torch.tensor(0.0)
        
        probs = weights / torch.sum(weights)
        
        # Filter out zeros to avoid log(0)
        mask = probs > 0
        masked_probs = probs[mask]
        
        # Compute entropy
        entropy = -torch.sum(masked_probs * torch.log(masked_probs))
        
        return entropy


if __name__ == "__main__":

    model = RoBERTaForNLI()
    print(f"Model initialized with {sum(p.numel() for p in model.model.parameters())} parameters")