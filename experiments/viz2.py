import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.gridspec import GridSpec

# Data from the table
strategies = ["uniform", "importance_weighting", "confidence_sampling", "curriculum"]
failure_rates = [0.0222, 0.0667, 0.1333, 0.3778]
percentages_above_epsilon = [2.2, 6.67, 13.33, 37.38]  # From image titles

# Epsilon threshold
epsilon = 0.01

# Plot configuration
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig)

# Function to generate synthetic pairwise difference data
def generate_pairwise_diffs(percentage_above_epsilon, n_pairs=45, max_diff=0.02):
    """Generate synthetic pairwise differences with a specified percentage above epsilon"""
    # Use a beta distribution skewed towards smaller differences
    n_above = int(percentage_above_epsilon/100 * n_pairs)
    
    # Values below epsilon
    if n_pairs - n_above > 0:
        below_epsilon = np.random.beta(2, 5, n_pairs - n_above) * epsilon
    else:
        below_epsilon = np.array([])
    
    # Values above epsilon
    if n_above > 0:
        above_epsilon = epsilon + np.random.beta(2, 7, n_above) * (max_diff - epsilon)
    else:
        above_epsilon = np.array([])
    
    # Combine and shuffle
    all_diffs = np.concatenate([below_epsilon, above_epsilon])
    np.random.shuffle(all_diffs)
    
    return all_diffs

# Create histograms for each strategy
for i, strategy in enumerate(strategies):
    # Calculate position in the grid
    row = i // 2
    col = i % 2
    ax = fig.add_subplot(gs[row, col])
    
    # Generate synthetic data matching the percentage above epsilon
    differences = generate_pairwise_diffs(percentages_above_epsilon[i])
    
    # Create histogram
    n, bins, patches = ax.hist(differences, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
    
    # Add KDE curve
    x = np.linspace(0, max(differences) * 1.1, 1000)
    kde = stats.gaussian_kde(differences)
    ax.plot(x, kde(x) * len(differences) * (bins[1] - bins[0]), color='blue')
    
    # Add vertical line for epsilon
    ax.axvline(x=epsilon, color='red', linestyle='--')
    ax.text(epsilon*1.05, ax.get_ylim()[1]*0.9, f'ε={epsilon}', color='red')
    
    # Add title with percentage above epsilon
    ax.set_title(f"{strategy}\n({percentages_above_epsilon[i]}% above ε)")
    
    # Labels
    ax.set_xlabel('|Accuracy Difference|')
    ax.set_ylabel('Frequency')
    
    # Adjust x-axis ticks to make them more readable
    ax.set_xticks(np.arange(0, max(differences)*1.2, 0.0025))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('accuracy_difference_histograms.png', dpi=300)
plt.show()