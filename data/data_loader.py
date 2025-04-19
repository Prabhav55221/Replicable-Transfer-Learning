"""
Data loading and preprocessing utilities for the MultiNLI dataset.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import RobertaTokenizer
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MultiNLIData:
    """
    Handles loading, preprocessing and analysis of the MultiNLI dataset for 
    replicability experiments in transfer learning.
    """
    
    def __init__(self, model_name: str = "roberta-base", max_length: int = 128, 
                 cache_dir: Optional[str] = None):
        """
        Initialize the MultiNLI data handler.
        
        Args:
            model_name: Name of the pre-trained model to use for tokenization
            max_length: Maximum sequence length for tokenization
            cache_dir: Directory to cache the dataset
        """
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.dataset = None
        self.matched_train = None
        self.matched_dev = None
        self.mismatched_train = None
        self.mismatched_dev = None
        
    def load_data(self) -> None:
        """Load the MultiNLI dataset from HuggingFace datasets."""
        print("Loading MultiNLI dataset...")
        self.dataset = load_dataset("multi_nli", cache_dir=self.cache_dir)
        
        # Split into matched and mismatched sets
        # In MultiNLI, validation set is already split into matched and mismatched
        self.matched_dev = self.dataset["validation_matched"]
        self.mismatched_dev = self.dataset["validation_mismatched"]
        
        # Training data needs to be split based on genre for our transfer setup
        train_data = self.dataset["train"]
        
        # Get unique genres in the training set
        genres = list(set(train_data["genre"]))
        
        # For transfer learning setup, we'll use some genres as matched (source)
        # and others as mismatched (target) for the training set
        # This creates a domain shift for our transfer learning scenario
        matched_genres = genres[:len(genres)//2]
        mismatched_genres = genres[len(genres)//2:]
        
        print(f"Source domain genres: {matched_genres}")
        print(f"Target domain genres: {mismatched_genres}")
        
        # Split training data
        self.matched_train = train_data.filter(lambda x: x["genre"] in matched_genres)
        self.mismatched_train = train_data.filter(lambda x: x["genre"] in mismatched_genres)
        
        print(f"Dataset loaded: {len(self.matched_train)} matched training examples, "
              f"{len(self.mismatched_train)} mismatched training examples")
    
    def preprocess_dataset(self, dataset, is_training=True):
        """
        Tokenize and prepare a dataset for the model.
        
        Args:
            dataset: HuggingFace dataset to preprocess
            is_training: Whether this is training data
            
        Returns:
            Preprocessed dataset
        """
        def tokenize_function(examples):
            # Tokenize the premises and hypotheses
            tokenized_examples = self.tokenizer(
                examples["premise"],
                examples["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Add labels
            tokenized_examples["labels"] = examples["label"]
            
            # Add genre information for domain-based methods
            tokenized_examples["genre"] = examples["genre"]
            
            return tokenized_examples
        
        # Apply preprocessing
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["premise", "hypothesis"],
        )
        
        # Set format for PyTorch
        tokenized_dataset.set_format(
            type="torch", 
            columns=["input_ids", "attention_mask", "labels", "genre"]
        )
        
        return tokenized_dataset
    
    def prepare_data_splits(self, source_sample_size=None, target_sample_size=None, seed=42):
        """
        Prepare all data splits needed for the experiments.
        
        Args:
            source_sample_size: Sample size for source domain (None for all)
            target_sample_size: Sample size for target domain (None for all)
            seed: Random seed for sampling
            
        Returns:
            Dictionary containing all data splits
        """
        if self.dataset is None:
            self.load_data()
        
        # Set seed for reproducibility
        set_seed(seed)
        
        # Sample from source and target domains if requested
        source_train = self.matched_train
        target_train = self.mismatched_train
        
        if source_sample_size is not None and source_sample_size < len(source_train):
            source_train = source_train.shuffle(seed=seed).select(range(source_sample_size))
            
        if target_sample_size is not None and target_sample_size < len(target_train):
            target_train = target_train.shuffle(seed=seed).select(range(target_sample_size))
        
        # Preprocess all splits
        source_train_processed = self.preprocess_dataset(source_train)
        target_train_processed = self.preprocess_dataset(target_train)
        matched_dev_processed = self.preprocess_dataset(self.matched_dev, is_training=False)
        mismatched_dev_processed = self.preprocess_dataset(self.mismatched_dev, is_training=False)
        
        return {
            "source_train": source_train_processed,
            "target_train": target_train_processed,
            "matched_dev": matched_dev_processed,
            "mismatched_dev": mismatched_dev_processed
        }
    
    def get_data_stats(self):
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.dataset is None:
            self.load_data()
            
        stats = {}
        
        # Count examples per split
        stats["source_train_size"] = len(self.matched_train)
        stats["target_train_size"] = len(self.mismatched_train)
        stats["matched_dev_size"] = len(self.matched_dev)
        stats["mismatched_dev_size"] = len(self.mismatched_dev)
        
        # Label distribution
        source_labels = Counter(self.matched_train["label"])
        target_labels = Counter(self.mismatched_train["label"])
        
        stats["source_label_dist"] = {
            "entailment": source_labels[0] / len(self.matched_train),
            "neutral": source_labels[1] / len(self.matched_train),
            "contradiction": source_labels[2] / len(self.matched_train)
        }
        
        stats["target_label_dist"] = {
            "entailment": target_labels[0] / len(self.mismatched_train),
            "neutral": target_labels[1] / len(self.mismatched_train),
            "contradiction": target_labels[2] / len(self.mismatched_train)
        }
        
        # Genre distribution
        source_genres = Counter(self.matched_train["genre"])
        target_genres = Counter(self.mismatched_train["genre"])
        
        stats["source_genres"] = {genre: count / len(self.matched_train) 
                                 for genre, count in source_genres.items()}
        stats["target_genres"] = {genre: count / len(self.mismatched_train) 
                                  for genre, count in target_genres.items()}
        
        return stats
        
    def plot_genre_distribution(self, save_path=None):
        """
        Plot the distribution of genres in source and target domains.
        
        Args:
            save_path: Path to save the plot (if None, will display)
        """
        stats = self.get_data_stats()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot source genres
        genres = list(stats["source_genres"].keys())
        counts = list(stats["source_genres"].values())
        ax1.bar(genres, counts)
        ax1.set_title("Source Domain Genres")
        ax1.set_ylabel("Proportion")
        ax1.set_xticklabels(genres, rotation=45, ha="right")
        
        # Plot target genres
        genres = list(stats["target_genres"].keys())
        counts = list(stats["target_genres"].values())
        ax2.bar(genres, counts)
        ax2.set_title("Target Domain Genres")
        ax2.set_ylabel("Proportion")
        ax2.set_xticklabels(genres, rotation=45, ha="right")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()


class NLIDataset(Dataset):
    """
    Dataset class for MultiNLI with support for adaptive selection methods.
    """
    
    def __init__(self, dataset, selection_weights=None):
        """
        Initialize the dataset with optional selection weights.
        
        Args:
            dataset: HuggingFace dataset
            selection_weights: Optional weights for each example (for adaptive selection)
        """
        self.dataset = dataset
        self.selection_weights = selection_weights
        
        if selection_weights is None:
            # Initialize uniform weights
            self.selection_weights = torch.ones(len(dataset)) / len(dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item
    
    def update_weights(self, new_weights):
        """
        Update the selection weights.
        
        Args:
            new_weights: New weights for each example
        """
        self.selection_weights = new_weights
    
    def get_weighted_sampler(self):
        """
        Get a weighted sampler based on current selection weights.
        
        Returns:
            Weighted sampler for DataLoader
        """
        from torch.utils.data import WeightedRandomSampler
        
        # Convert weights to probabilities
        weights = self.selection_weights.clone()
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones_like(weights) / len(weights)
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        
        return sampler


if __name__ == "__main__":

    mnli_data = MultiNLIData()
    mnli_data.load_data()
    
    stats = mnli_data.get_data_stats()
    print("Dataset statistics:", stats)
    mnli_data.plot_genre_distribution(save_path="./plots/genre_distribution.png")
    
    data_splits = mnli_data.prepare_data_splits(
        source_sample_size=1000,
        target_sample_size=1000,
        seed=42
    )
    
    source_dataset = NLIDataset(data_splits["source_train"])
    target_dataset = NLIDataset(data_splits["target_train"])
    
    print(f"Source dataset size: {len(source_dataset)}")
    print(f"Target dataset size: {len(target_dataset)}")