# viz.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import matplotlib.gridspec as gridspec
import argparse
from collections import defaultdict
import torch

def load_experiment_results(json_path):
    """Load experiment results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path

def safe_get(d, keys, default=None):
    """Safely access nested dictionary keys."""
    if not isinstance(keys, list):
        keys = keys.split('.')
    
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d

# Performance Visualization Functions
def plot_accuracy_distributions(results, output_dir):
    """Create detailed plots of accuracy distributions across strategies."""
    # Extract accuracy data
    data = []
    for strategy, strategy_data in results['strategies'].items():
        for i, acc in enumerate(strategy_data['accuracies']):
            data.append({
                'Strategy': strategy,
                'Run': f'Run {i+1}',
                'Accuracy': acc,
                'Domain': 'Mismatched'
            })
        for i, acc in enumerate(strategy_data.get('matched_accuracies', [])):
            data.append({
                'Strategy': strategy,
                'Run': f'Run {i+1}',
                'Accuracy': acc,
                'Domain': 'Matched'
            })
    
    df = pd.DataFrame(data)
    
    # Create output directory
    vis_dir = create_dir(os.path.join(output_dir, 'accuracy_analysis'))
    
    # Box plot with individual points
    plt.figure(figsize=(12, 7))
    ax = sns.boxplot(x='Strategy', y='Accuracy', hue='Domain', data=df, palette='pastel')
    sns.stripplot(x='Strategy', y='Accuracy', hue='Domain', data=df, dodge=True, 
                 alpha=0.6, palette='dark', size=6)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title('Accuracy Distribution by Selection Strategy', fontsize=14)
    plt.legend(title='Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'accuracy_distribution.png'), dpi=300)
    plt.close()
    
    # Violin plot
    plt.figure(figsize=(12, 7))
    sns.violinplot(x='Strategy', y='Accuracy', hue='Domain', data=df, inner='quartile', split=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title('Accuracy Distribution (Violin Plot)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'accuracy_violin.png'), dpi=300)
    plt.close()
    
    # Performance stability metrics table
    plt.figure(figsize=(12, len(results['strategies']) * 0.5 + 1.5))
    plt.axis('tight')
    plt.axis('off')
    
    table_data = []
    headers = ['Strategy', 'Mean', 'Std Dev', 'Min', 'Max', 'Range', 'Failure Rate']
    
    for strategy, summary in results['summary'].items():
        metrics = summary['stability_metrics']
        row = [
            strategy,
            f"{metrics['mean']:.4f}",
            f"{metrics['std']:.4f}",
            f"{metrics['min']:.4f}",
            f"{metrics['max']:.4f}",
            f"{metrics['range']:.4f}",
            f"{summary['replicability_failure_rate']:.4f}"
        ]
        table_data.append(row)
    
    table = plt.table(cellText=table_data, colLabels=headers, loc='center', 
                     cellLoc='center', colColours=['#f0f0f0']*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title('Performance Stability Metrics', fontsize=14)
    plt.savefig(os.path.join(vis_dir, 'stability_metrics_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance-stability trade-off
    plt.figure(figsize=(10, 8))
    strategies = list(results['summary'].keys())
    means = [summary['stability_metrics']['mean'] for strategy, summary in results['summary'].items()]
    stds = [summary['stability_metrics']['std'] for strategy, summary in results['summary'].items()]
    
    # Create scatter plot with error bars
    plt.errorbar(stds, means, xerr=None, yerr=None, fmt='o', markersize=10, 
                capsize=0, elinewidth=2, markeredgewidth=2)
    
    # Annotate points
    for i, strategy in enumerate(strategies):
        plt.annotate(strategy, (stds[i], means[i]), xytext=(7, 0), 
                    textcoords='offset points', fontsize=12)
    
    plt.xlabel('Standard Deviation (Lower is Better)', fontsize=12)
    plt.ylabel('Mean Accuracy (Higher is Better)', fontsize=12)
    plt.title('Performance-Stability Trade-off', fontsize=14)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'performance_stability_tradeoff.png'), dpi=300)
    plt.close()

def plot_replicability_analysis(results, output_dir):
    """Generate replicability analysis visualizations."""
    vis_dir = create_dir(os.path.join(output_dir, 'replicability_analysis'))
    
    # Replicability failure rate bar chart
    plt.figure(figsize=(10, 6))
    strategies = list(results['summary'].keys())
    failure_rates = [summary['replicability_failure_rate'] for strategy, summary in results['summary'].items()]
    selection_sensitivities = [summary['avg_selection_sensitivity'] for strategy, summary in results['summary'].items()]
    
    bars = plt.bar(strategies, failure_rates, color=sns.color_palette("muted"))
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.ylim(0, max(failure_rates) * 1.2 or 0.1)  # Add headroom for text
    plt.ylabel('Replicability Failure Rate (ρ)', fontsize=12)
    plt.title('Replicability Failure Rate by Strategy', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'failure_rates.png'), dpi=300)
    plt.close()
    
    # Theoretical bound visualization
    plt.figure(figsize=(10, 6))
    
    # Calculate theoretical bound curve: ρ ≤ 2 * exp(-ε²n / Δ²Q)
    epsilon = results['metadata']['epsilon']
    n_values = np.arange(500, 10000, 100)
    
    plt.subplot(1, 2, 1)
    for strategy, sensitivity in zip(strategies, selection_sensitivities):
        if sensitivity > 0:
            bound_values = [2 * np.exp(-(epsilon**2 * n) / (sensitivity**2)) for n in n_values]
            plt.plot(n_values, bound_values, label=f"{strategy} (Δ={sensitivity:.2f})")

    # Add this line to set y-axis to log scale:
    plt.yscale('log')  # <-- Add this line

    # Adjust y-limits to focus on meaningful probability range (0,1]
    plt.ylim(1e-3, 2)  # Shows 0.001 to 2 range

    # Consider adding a horizontal line at y=1 to indicate the maximum probability
    plt.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='Max probability')
    
    plt.ylabel('Replicability Failure Rate (ρ)', fontsize=12)
    plt.xlabel('Sample Size (n)', fontsize=12)
    plt.title('Theoretical Bound vs. Sample Size', fontsize=14)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Sensitivity vs failure rate
    plt.subplot(1, 2, 2)
    plt.scatter(selection_sensitivities, failure_rates, s=100)
    
    for i, strategy in enumerate(strategies):
        plt.annotate(strategy, (selection_sensitivities[i], failure_rates[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=12)
    
    # Add trend line if we have enough points
    if len(strategies) > 1:
        z = np.polyfit(selection_sensitivities, failure_rates, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(selection_sensitivities), max(selection_sensitivities), 100)
        plt.plot(x_trend, p(x_trend), "--", color='red')
        
        # Calculate correlation
        corr, p_value = stats.pearsonr(selection_sensitivities, failure_rates)
        plt.annotate(f"Correlation: {corr:.3f}\np-value: {p_value:.3f}", 
                    xy=(0.05, 0.95), xycoords='axes fraction', 
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.xlabel('Selection Sensitivity (Δ_Q)', fontsize=12)
    plt.ylabel('Replicability Failure Rate (ρ)', fontsize=12)
    plt.title('Sensitivity vs. Replicability', fontsize=14)
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'replicability_analysis.png'), dpi=300)
    plt.close()
    
    # Pairwise accuracy differences distribution
    strategies_with_multiple_runs = [s for s, d in results['strategies'].items() 
                                if len(d.get('accuracies', [])) > 1]
    n_strategies = len(strategies_with_multiple_runs)

    if n_strategies > 0:
        # Calculate grid dimensions dynamically
        n_cols = min(3, n_strategies)  # At most 3 columns
        n_rows = (n_strategies + n_cols - 1) // n_cols  # Ceiling division
        
        plt.figure(figsize=(n_cols * 5, n_rows * 4))
        
        for i, strategy in enumerate(strategies_with_multiple_runs):
            data = results['strategies'][strategy]
            accuracies = data['accuracies']
            n_runs = len(accuracies)
            
            # Compute all pairwise differences
            deltas = []
            for j in range(n_runs):
                for k in range(j+1, n_runs):
                    deltas.append(abs(accuracies[j] - accuracies[k]))
            
            plt.subplot(n_rows, n_cols, i+1)
            if deltas:  # Only plot if we have data
                sns.histplot(deltas, bins=10, kde=True)
                plt.axvline(x=results['metadata']['epsilon'], color='red', linestyle='--')
                plt.text(results['metadata']['epsilon']*1.05, plt.ylim()[1]*0.9, 
                        f'ε={results["metadata"]["epsilon"]}', color='red', fontsize=12)
                
                # Calculate percentage above epsilon
                above_epsilon = sum(1 for d in deltas if d > results['metadata']['epsilon'])
                percentage = (above_epsilon / len(deltas)) * 100 if deltas else 0
                
                plt.xlabel('|Accuracy Difference|', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.title(f'{strategy}\n({percentage:.1f}% above ε)', fontsize=14)
                plt.grid(linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'pairwise_accuracy_differences.png'), dpi=300)
        plt.close()

def plot_representation_analysis(results, output_dir):
    """Generate visualizations for model representation analysis."""
    vis_dir = create_dir(os.path.join(output_dir, 'representation_analysis'))
    
    # Plot similarity matrices
    plt.figure(figsize=(15, 10))
    
    for i, (strategy, summary) in enumerate(results['summary'].items()):
        if 'representation_similarity_matrix' not in summary:
            continue
        
        plt.subplot(2, 2, i+1)
        similarity_matrix = np.array(summary['representation_similarity_matrix'])
        im = plt.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('CKA Similarity')
        
        # Add labels
        n_runs = similarity_matrix.shape[0]
        plt.xticks(range(n_runs), [f'Run {i+1}' for i in range(n_runs)])
        plt.yticks(range(n_runs), [f'Run {i+1}' for i in range(n_runs)])
        
        # Add title with average similarity
        avg_sim = summary['avg_representation_similarity']
        plt.title(f'{strategy}\nAvg. Similarity: {avg_sim:.3f}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'similarity_matrices.png'), dpi=300)
    plt.close()
    
    # MDS visualization of representation space
    strategies = list(results['summary'].keys())
    
    # Collect all similarity matrices
    all_matrices = []
    all_strategy_names = []
    
    for strategy in strategies:
        if 'representation_similarity_matrix' in results['summary'][strategy]:
            all_matrices.append(np.array(results['summary'][strategy]['representation_similarity_matrix']))
            all_strategy_names.append(strategy)
    
    if all_matrices:
        # Combine matrices into a block diagonal matrix
        n_runs = all_matrices[0].shape[0]
        n_strategies = len(all_matrices)
        combined_size = n_runs * n_strategies
        combined_matrix = np.zeros((combined_size, combined_size))
        
        # Fill diagonal blocks with similarity matrices
        for i, matrix in enumerate(all_matrices):
            start_idx = i * n_runs
            end_idx = (i + 1) * n_runs
            combined_matrix[start_idx:end_idx, start_idx:end_idx] = matrix
        
        # Fill off-diagonal blocks with cross-strategy similarities (if available)
        cross_similarities = results.get('cross_strategy', {}).get('cross_representation_similarity', {})
        for key, sim_value in cross_similarities.items():
            strat1, strat2 = key.split('_vs_')
            if strat1 in all_strategy_names and strat2 in all_strategy_names:
                i1 = all_strategy_names.index(strat1)
                i2 = all_strategy_names.index(strat2)
                start1, end1 = i1 * n_runs, (i1 + 1) * n_runs
                start2, end2 = i2 * n_runs, (i2 + 1) * n_runs
                combined_matrix[start1:end1, start2:end2] = sim_value
                combined_matrix[start2:end2, start1:end1] = sim_value
        
        # Convert to distance matrix
        distance_matrix = 1 - combined_matrix
        
        # Apply MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        points = mds.fit_transform(distance_matrix)
        
        # Plot MDS visualization
        plt.figure(figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_strategy_names)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']  # Different marker styles
        
        for i, strategy in enumerate(all_strategy_names):
            start_idx = i * n_runs
            end_idx = (i + 1) * n_runs
            
            plt.scatter(points[start_idx:end_idx, 0], points[start_idx:end_idx, 1], 
                       color=colors[i], marker=markers[i % len(markers)], s=100, label=strategy)
            
            # Draw convex hull or ellipse around points of same strategy
            if end_idx - start_idx > 2:  # Need at least 3 points
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points[start_idx:end_idx])
                for simplex in hull.simplices:
                    plt.plot(points[start_idx:end_idx, 0][simplex], points[start_idx:end_idx, 1][simplex], 
                            '-', color=colors[i], alpha=0.5)
        
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.title('MDS Projection of Model Representations', fontsize=14)
        plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'mds_projection.png'), dpi=300)
        plt.close()

def plot_training_dynamics(results, output_dir):
    """Visualize training dynamics and learning curves."""
    vis_dir = create_dir(os.path.join(output_dir, 'training_dynamics'))
    
    # Process training history for each strategy and run
    strategy_histories = defaultdict(list)
    
    for strategy, data in results['strategies'].items():
        for run in data['runs']:
            if 'training_history' in run:
                strategy_histories[strategy].append(run['training_history'])
    
    # Generate learning curve plots
    for strategy, histories in strategy_histories.items():
        if not histories:
            continue
            
        # Training loss curves
        if 'train' in histories[0] and 'loss' in histories[0]['train']:
            plt.figure(figsize=(12, 6))
            
            # Plot individual runs
            max_steps = max([len(h['train']['loss']) for h in histories])
            x_axis = list(range(max_steps))
            
            for i, history in enumerate(histories):
                losses = history['train']['loss']
                plt.plot(losses, alpha=0.3, label=f"Run {i+1}" if i == 0 else None)
            
            # Calculate and plot mean and confidence interval
            aligned_lengths = min([len(h['train']['loss']) for h in histories])
            if aligned_lengths > 0:
                aligned_losses = np.array([h['train']['loss'][:aligned_lengths] for h in histories])
                mean_loss = np.mean(aligned_losses, axis=0)
                std_loss = np.std(aligned_losses, axis=0)
                
                x = list(range(aligned_lengths))
                plt.plot(x, mean_loss, 'b-', linewidth=2, label='Mean')
                plt.fill_between(x, mean_loss - std_loss, mean_loss + std_loss, 
                                color='b', alpha=0.2, label='±1σ')
            
            plt.xlabel('Training Steps', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title(f'Training Loss - {strategy}', fontsize=14)
            plt.grid(linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{strategy}_training_loss.png'), dpi=300)
            plt.close()
        
        # Evaluation accuracy curves
        if 'eval' in histories[0] and 'mismatched_accuracy' in histories[0]['eval']:
            plt.figure(figsize=(12, 6))
            
            # Plot individual runs
            for i, history in enumerate(histories):
                steps = list(range(len(history['eval']['mismatched_accuracy'])))
                plt.plot(steps, history['eval']['mismatched_accuracy'], 
                        alpha=0.3, label=f"Run {i+1}" if i == 0 else None)
            
            # Calculate and plot mean and confidence interval
            aligned_lengths = min([len(h['eval']['mismatched_accuracy']) for h in histories])
            if aligned_lengths > 0:
                aligned_accs = np.array([h['eval']['mismatched_accuracy'][:aligned_lengths] for h in histories])
                mean_acc = np.mean(aligned_accs, axis=0)
                std_acc = np.std(aligned_accs, axis=0)
                
                x = list(range(aligned_lengths))
                plt.plot(x, mean_acc, 'g-', linewidth=2, label='Mean')
                plt.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, 
                                color='g', alpha=0.2, label='±1σ')
            
            plt.xlabel('Evaluation Steps', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title(f'Evaluation Accuracy (Mismatched) - {strategy}', fontsize=14)
            plt.grid(linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{strategy}_eval_accuracy.png'), dpi=300)
            plt.close()
    
    # Combined plot of final validation curves
    plt.figure(figsize=(12, 6))
    
    for strategy, histories in strategy_histories.items():
        if not histories or 'eval' not in histories[0] or 'mismatched_accuracy' not in histories[0]['eval']:
            continue
            
        # Calculate average accuracy at each evaluation point
        aligned_lengths = min([len(h['eval']['mismatched_accuracy']) for h in histories])
        if aligned_lengths > 0:
            aligned_accs = np.array([h['eval']['mismatched_accuracy'][:aligned_lengths] for h in histories])
            mean_acc = np.mean(aligned_accs, axis=0)
            steps = list(range(aligned_lengths))
            
            plt.plot(steps, mean_acc, linewidth=2, label=strategy)
    
    plt.xlabel('Evaluation Steps', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Validation Accuracy Comparison', fontsize=14)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'combined_eval_accuracy.png'), dpi=300)
    plt.close()

def plot_selection_weight_analysis(results, output_dir):
    """Analyze and visualize selection weight patterns."""
    vis_dir = create_dir(os.path.join(output_dir, 'selection_analysis'))
    
    # Process selection weight history for each strategy and run
    for strategy, data in results['strategies'].items():
        if strategy == 'uniform':  # Skip uniform strategy
            continue
            
        weight_stats = []
        
        for run_idx, run in enumerate(data['runs']):
            if 'training_history' not in run or 'train' not in run['training_history'] or 'selection_weights' not in run['training_history']['train']:
                continue
                
            selection_weights = run['training_history']['train']['selection_weights']
            
            if not selection_weights:
                continue
                
            # Extract metrics
            run_stats = {
                'run': run_idx,
                'epochs': [sw.get('epoch', i) for i, sw in enumerate(selection_weights)],
                'means': [sw.get('mean', 0) for sw in selection_weights],
                'stds': [sw.get('std', 0) for sw in selection_weights],
                'mins': [sw.get('min', 0) for sw in selection_weights],
                'maxs': [sw.get('max', 0) for sw in selection_weights],
                'entropies': [sw.get('entropy', 0) for sw in selection_weights]
            }
            
            weight_stats.append(run_stats)
        
        if not weight_stats:
            continue
            
        # Plot weight statistics over epochs
        plt.figure(figsize=(15, 10))
        
        # Plot mean and range for each run
        for i, stats in enumerate(weight_stats):
            plt.subplot(2, 2, 1)
            plt.plot(stats['epochs'], stats['means'], marker='o', label=f"Run {stats['run']+1}")
            
            plt.subplot(2, 2, 2)
            plt.plot(stats['epochs'], stats['stds'], marker='s', label=f"Run {stats['run']+1}")
            
            plt.subplot(2, 2, 3)
            plt.fill_between(stats['epochs'], stats['mins'], stats['maxs'], alpha=0.2)
            plt.plot(stats['epochs'], stats['mins'], marker='v', linestyle='--', alpha=0.7)
            plt.plot(stats['epochs'], stats['maxs'], marker='^', linestyle='--', alpha=0.7)
            
            plt.subplot(2, 2, 4)
            plt.plot(stats['epochs'], stats['entropies'], marker='D', label=f"Run {stats['run']+1}")
        
        # Add titles and labels
        plt.subplot(2, 2, 1)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Mean Weight', fontsize=12)
        plt.title('Mean Selection Weight', fontsize=14)
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Std Dev', fontsize=12)
        plt.title('Weight Standard Deviation', fontsize=14)
        plt.grid(linestyle='--', alpha=0.7)
        
        plt.subplot(2, 2, 3)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Weight Value', fontsize=12)
        plt.title('Min-Max Weight Range', fontsize=14)
        plt.grid(linestyle='--', alpha=0.7)
        
        plt.subplot(2, 2, 4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Entropy', fontsize=12)
        plt.title('Selection Weight Entropy', fontsize=14)
        plt.grid(linestyle='--', alpha=0.7)
        
        plt.suptitle(f'Selection Weight Analysis - {strategy}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(vis_dir, f'{strategy}_weight_analysis.png'), dpi=300)
        plt.close()

def create_replicability_dashboard(results, output_dir):
    """Create a comprehensive dashboard summarizing replicability findings."""
    # Create dashboard directory
    dashboard_dir = create_dir(os.path.join(output_dir, 'dashboard'))
    
    # Extract key metrics
    strategies = list(results['summary'].keys())
    mean_accuracies = [summary['stability_metrics']['mean'] for strategy, summary in results['summary'].items()]
    std_accuracies = [summary['stability_metrics']['std'] for strategy, summary in results['summary'].items()]
    failure_rates = [summary['replicability_failure_rate'] for strategy, summary in results['summary'].items()]
    rep_similarities = [summary.get('avg_representation_similarity', 0) for strategy, summary in results['summary'].items()]
    
    # Create dashboard figure
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # 1. Performance comparison
    ax1 = fig.add_subplot(gs[0, 0:2])
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, mean_accuracies, width, label='Mean Accuracy', 
                   color='skyblue', yerr=std_accuracies, capsize=5)
    bars2 = ax1.bar(x + width/2, failure_rates, width, label='Failure Rate', 
                   color='salmon')
    
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Performance and Replicability by Strategy', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Mean vs Std plot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(std_accuracies, mean_accuracies, s=100)
    
    # Add annotations
    for i, strategy in enumerate(strategies):
        ax2.annotate(strategy, (std_accuracies[i], mean_accuracies[i]), 
                    xytext=(7, 0), textcoords='offset points')
    
    ax2.set_xlabel('Standard Deviation (lower is better)', fontsize=12)
    ax2.set_ylabel('Mean Accuracy (higher is better)', fontsize=12)
    ax2.set_title('Performance-Stability Trade-off', fontsize=14)
    ax2.grid(linestyle='--', alpha=0.7)
    
    # 3. Theory vs. Practice plot
    ax3 = fig.add_subplot(gs[1, 0:2])
    sensitivities = [summary['avg_selection_sensitivity'] for strategy, summary in results['summary'].items()]
    
    # Create scatter plot
    ax3.scatter(sensitivities, failure_rates, s=120)
    
    # Add trend line
    if len(strategies) > 1:
        # Remove points with zero sensitivity for trend line calculation
        valid_points = [(s, f) for s, f in zip(sensitivities, failure_rates) if s > 0]
        if len(valid_points) > 1:
            valid_sensitivities, valid_failures = zip(*valid_points)
            z = np.polyfit(valid_sensitivities, valid_failures, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(valid_sensitivities), max(valid_sensitivities), 100)
            ax3.plot(x_trend, p(x_trend), "--", color='red')
    
    # Add annotations
    for i, strategy in enumerate(strategies):
        ax3.annotate(strategy, (sensitivities[i], failure_rates[i]), 
                    xytext=(7, 0), textcoords='offset points')
    
    # Add theoretical reference: ρ ≤ 2 * exp(-ε²n / Δ²Q)
    epsilon = results['metadata']['epsilon']
    n_samples = results.get('metadata', {}).get('target_sample_size', 2000)
    sensitivity_range = np.linspace(0.01, max(sensitivities) * 1.5, 100)
    theoretical_bound = [2 * np.exp(-(epsilon**2 * n_samples) / (s**2)) if s > 0 else 0 for s in sensitivity_range]
    
    ax3.plot(sensitivity_range, theoretical_bound, 'k--', alpha=0.5, label=f'Theoretical Bound (n={n_samples})')
    
    ax3.set_xlabel('Selection Sensitivity (Δ_Q)', fontsize=12)
    ax3.set_ylabel('Replicability Failure Rate (ρ)', fontsize=12)
    ax3.set_title('Selection Sensitivity vs. Replicability', fontsize=14)
    ax3.grid(linestyle='--', alpha=0.7)
    ax3.legend()
    
    # 4. Representation similarity comparison
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.bar(strategies, rep_similarities, color='lightgreen')
    
    ax4.set_ylabel('Similarity (CKA)', fontsize=12)
    ax4.set_title('Average Representation Similarity', fontsize=14)
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Summary table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')
    
    table_data = []
    for strategy in strategies:
        metrics = results['summary'][strategy]['stability_metrics']
        
        row = [
            strategy,
            f"{metrics['mean']:.4f}",
            f"{metrics['std']:.4f}",
            f"{metrics['min']:.4f}",
            f"{metrics['max']:.4f}",
            f"{metrics['range']:.4f}",
            f"{results['summary'][strategy]['replicability_failure_rate']:.4f}",
            f"{results['summary'][strategy].get('avg_representation_similarity', 'N/A')}",
            f"{results['summary'][strategy].get('avg_selection_sensitivity', 'N/A')}"
        ]
        table_data.append(row)
    
    columns = [
        'Strategy', 'Mean Acc', 'Std Dev', 'Min Acc', 'Max Acc', 'Range', 
        'Failure Rate', 'Rep. Similarity', 'Selection Sensitivity'
    ]
    
    table = ax5.table(cellText=table_data, colLabels=columns, loc='center', 
                     cellLoc='center', colColours=['#f0f0f0']*len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Add experiment metadata
    metadata = results.get('metadata', {})
    metadata_text = (
        f"Experiment: {metadata.get('timestamp', 'Unknown')}\n"
        f"Number of Runs: {metadata.get('num_runs', 'Unknown')}, "
        f"Sample Size: {metadata.get('target_sample_size', 'Unknown')}, "
        f"Epsilon: {metadata.get('epsilon', 'Unknown')}"
    )
    fig.text(0.5, 0.01, metadata_text, ha='center', fontsize=12)
    
    plt.suptitle('Replicability in Transfer Learning Dashboard', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    plt.savefig(os.path.join(dashboard_dir, 'replicability_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

def compare_experiments(experiment_paths, labels, output_dir):
    """Compare results from multiple experiments."""
    if len(experiment_paths) < 2:
        print("Need at least two experiments to compare.")
        return
        
    if len(labels) != len(experiment_paths):
        labels = [f"Experiment {i+1}" for i in range(len(experiment_paths))]
    
    # Load all experiment results
    experiments = []
    for path in experiment_paths:
        experiments.append(load_experiment_results(path))
    
    # Create comparison directory
    comp_dir = create_dir(os.path.join(output_dir, 'experiment_comparison'))
    
    # Get all unique strategies across experiments
    all_strategies = set()
    for exp in experiments:
        all_strategies.update(exp['strategies'].keys())
    
    # Performance comparison across experiments
    plt.figure(figsize=(15, 8))
    
    x = np.arange(len(all_strategies))
    width = 0.8 / len(experiments)
    
    for i, (exp, label) in enumerate(zip(experiments, labels)):
        means = []
        stds = []
        
        for strategy in all_strategies:
            if strategy in exp['summary']:
                means.append(exp['summary'][strategy]['stability_metrics']['mean'])
                stds.append(exp['summary'][strategy]['stability_metrics']['std'])
            else:
                means.append(0)
                stds.append(0)
        
        plt.bar(x - 0.4 + (i + 0.5) * width, means, width, yerr=stds, 
               capsize=5, label=label)
    
    plt.xlabel('Strategy', fontsize=12)
    plt.ylabel('Mean Accuracy', fontsize=12)
    plt.title('Performance Comparison Across Experiments', fontsize=14)
    plt.xticks(x, all_strategies)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(comp_dir, 'performance_comparison.png'), dpi=300)
    plt.close()
    
    # Replicability failure rate comparison
    plt.figure(figsize=(15, 8))
    
    for i, (exp, label) in enumerate(zip(experiments, labels)):
        failure_rates = []
        
        for strategy in all_strategies:
            if strategy in exp['summary']:
                failure_rates.append(exp['summary'][strategy]['replicability_failure_rate'])
            else:
                failure_rates.append(0)
        
        plt.bar(x - 0.4 + (i + 0.5) * width, failure_rates, width, label=label)
    
    plt.xlabel('Strategy', fontsize=12)
    plt.ylabel('Replicability Failure Rate', fontsize=12)
    plt.title('Replicability Comparison Across Experiments', fontsize=14)
    plt.xticks(x, all_strategies)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(comp_dir, 'replicability_comparison.png'), dpi=300)
    plt.close()
    
    # Create comparison table
    plt.figure(figsize=(15, 8))
    plt.axis('tight')
    plt.axis('off')
    
    # Prepare table data
    table_data = []
    for strategy in all_strategies:
        row = [strategy]
        
        for exp, label in zip(experiments, labels):
            if strategy in exp['summary']:
                metrics = exp['summary'][strategy]['stability_metrics']
                failure_rate = exp['summary'][strategy]['replicability_failure_rate']
                cell_text = f"Acc: {metrics['mean']:.4f} ± {metrics['std']:.4f}\nFail: {failure_rate:.4f}"
            else:
                cell_text = "N/A"
            
            row.append(cell_text)
        
        table_data.append(row)
    
    # Create table
    table_columns = ['Strategy'] + labels
    table = plt.table(cellText=table_data, colLabels=table_columns, loc='center', 
                     cellLoc='center', colColours=['#f0f0f0']*len(table_columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('Experiment Comparison Summary', fontsize=14)
    plt.savefig(os.path.join(comp_dir, 'comparison_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate enhanced visualizations for replicability experiments')
    parser.add_argument('--results', nargs='+', required=True, help='Path(s) to experiment results JSON file(s)')
    parser.add_argument('--labels', nargs='+', help='Labels for multiple experiments (for comparison)')
    parser.add_argument('--output_dir', default='./visualizations', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle comparison of multiple experiments
    if len(args.results) > 1:
        if not args.labels or len(args.labels) != len(args.results):
            args.labels = [f"Experiment-{i+1}" for i in range(len(args.results))]
        
        print(f"Comparing {len(args.results)} experiments...")
        compare_experiments(args.results, args.labels, args.output_dir)
    
    # Process each experiment
    for i, result_path in enumerate(args.results):
        print(f"Processing: {result_path}")
        
        # Load results
        results = load_experiment_results(result_path)
        
        # Create experiment-specific output directory
        if len(args.results) > 1:
            exp_dir = os.path.join(args.output_dir, args.labels[i])
        else:
            exp_dir = args.output_dir
        
        os.makedirs(exp_dir, exist_ok=True)
        
        # Generate visualizations
        print("  Generating accuracy analysis plots...")
        plot_accuracy_distributions(results, exp_dir)
        
        print("  Generating replicability analysis...")
        plot_replicability_analysis(results, exp_dir)
        
        print("  Generating representation analysis...")
        plot_representation_analysis(results, exp_dir)
        
        print("  Generating training dynamics plots...")
        plot_training_dynamics(results, exp_dir)
        
        print("  Generating selection weight analysis...")
        plot_selection_weight_analysis(results, exp_dir)
        
        print("  Creating replicability dashboard...")
        create_replicability_dashboard(results, exp_dir)
        
        print(f"Visualizations saved to: {exp_dir}")

if __name__ == "__main__":
    main()