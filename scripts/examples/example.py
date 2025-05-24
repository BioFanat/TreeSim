#!/usr/bin/env python3
"""
Example script demonstrating programmatic usage of the Tree Correlation Analyzer.

This shows how to:
1. Use the library in your own Python code
2. Create custom statistics
3. Run batch analyses
4. Process results programmatically
"""

import numpy as np
import matplotlib.pyplot as plt
from ..tree_builder import (
    PreferentialAttachmentTree, UniformAttachmentTree,
    TreeSimulator, TestStatistic, TreeNode,
    SharedEdgesFraction, LargestCommonSubtree,
    plot_distributions
)
from typing import Dict


# Example 1: Basic programmatic usage
def example_basic_usage():
    """Basic example of using the tree correlation analyzer."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create a tree generator
    generator = PreferentialAttachmentTree()
    
    # Create simulator with a fixed seed
    simulator = TreeSimulator(generator, random_seed=42)
    
    # Add statistics
    simulator.add_statistic(SharedEdgesFraction())
    simulator.add_statistic(LargestCommonSubtree())
    
    # Run simulation
    results = simulator.run_simulation(n=500, rho=0.5, num_runs=100)
    
    # Process results
    for result in results:
        print(f"\n{result.statistic_name}:")
        print(f"  Effect size: {result.effect_size:.3f}")
        print(f"  P-value estimate: {estimate_pvalue(result):.4f}")


# Example 2: Custom test statistic
class DegreeSimilarity(TestStatistic):
    """Custom statistic: correlation of degree sequences."""
    
    @property
    def name(self) -> str:
        return "Degree Sequence Correlation"
    
    @property
    def description(self) -> str:
        return "Pearson correlation between degree sequences of the two trees"
    
    def compute(self, tree1: Dict[int, TreeNode], tree2: Dict[int, TreeNode]) -> float:
        # Compute degree sequences
        degrees1 = [len(node.children) + (1 if node.parent is not None else 0) 
                   for node in tree1.values()]
        degrees2 = [len(node.children) + (1 if node.parent is not None else 0) 
                   for node in tree2.values()]
        
        # Compute correlation
        correlation = np.corrcoef(degrees1, degrees2)[0, 1]
        
        # Map to [0, 1] range
        return (correlation + 1) / 2


def example_custom_statistic():
    """Example of using a custom test statistic."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Test Statistic")
    print("=" * 60)
    
    generator = PreferentialAttachmentTree()
    simulator = TreeSimulator(generator, random_seed=123)
    
    # Add our custom statistic
    custom_stat = DegreeSimilarity()
    simulator.add_statistic(custom_stat)
    
    # Run for different correlation values
    rho_values = [0.0, 0.3, 0.5, 0.7, 0.9]
    mean_values = []
    
    for rho in rho_values:
        results = simulator.run_simulation(n=1000, rho=rho, num_runs=50, progress=False)
        mean_values.append(results[0].alt_mean)
        print(f"ρ={rho:.1f}: mean degree correlation = {results[0].alt_mean:.3f}")
    
    # Plot relationship
    plt.figure(figsize=(8, 6))
    plt.plot(rho_values, mean_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Tree Correlation Parameter (ρ)')
    plt.ylabel('Mean Degree Sequence Correlation')
    plt.title('Degree Similarity vs Tree Correlation')
    plt.grid(True, alpha=0.3)
    plt.show()


# Example 3: Batch analysis
def example_batch_analysis():
    """Example of running batch analyses across parameter ranges."""
    print("\n" + "=" * 60)
    print("Example 3: Batch Analysis")
    print("=" * 60)
    
    # Parameter ranges to explore
    n_values = [100, 500, 1000]
    rho_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Storage for results
    effect_sizes = np.zeros((len(n_values), len(rho_values)))
    
    # Run simulations
    generator = PreferentialAttachmentTree()
    simulator = TreeSimulator(generator, random_seed=456)
    simulator.add_statistic(SharedEdgesFraction())
    
    for i, n in enumerate(n_values):
        for j, rho in enumerate(rho_values):
            print(f"Running n={n}, ρ={rho}...", end='', flush=True)
            results = simulator.run_simulation(n=n, rho=rho, num_runs=100, progress=False)
            effect_sizes[i, j] = results[0].effect_size
            print(f" Effect size: {effect_sizes[i, j]:.2f}")
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(effect_sizes, aspect='auto', cmap='viridis')
    plt.colorbar(label='Effect Size (Cohen\'s d)')
    plt.xlabel('Correlation Parameter (ρ)')
    plt.ylabel('Number of Nodes (n)')
    plt.xticks(range(len(rho_values)), rho_values)
    plt.yticks(range(len(n_values)), n_values)
    plt.title('Effect Size Heatmap: Shared Edges Statistic')
    
    # Add text annotations
    for i in range(len(n_values)):
        for j in range(len(rho_values)):
            plt.text(j, i, f'{effect_sizes[i, j]:.1f}', 
                    ha='center', va='center', color='white' if effect_sizes[i, j] > 5 else 'black')
    
    plt.tight_layout()
    plt.show()


# Example 4: Comparing tree models
def example_model_comparison():
    """Example comparing PA and UA tree models."""
    print("\n" + "=" * 60)
    print("Example 4: Comparing Tree Models")
    print("=" * 60)
    
    n = 1000
    rho = 0.5
    runs = 200
    
    # Run for both tree types
    models = {
        'Preferential Attachment': PreferentialAttachmentTree(),
        'Uniform Attachment': UniformAttachmentTree()
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (name, generator) in enumerate(models.items()):
        simulator = TreeSimulator(generator, random_seed=789)
        simulator.add_statistic(SharedEdgesFraction())
        
        results = simulator.run_simulation(n=n, rho=rho, num_runs=runs, progress=False)
        result = results[0]
        
        # Plot on subplot
        ax = axes[idx]
        ax.hist(result.null_values, bins=30, alpha=0.5, label='Null (ρ=0)', color='blue', density=True)
        ax.hist(result.alt_values, bins=30, alpha=0.5, label=f'Alt (ρ={rho})', color='red', density=True)
        ax.axvline(result.null_mean, color='blue', linestyle='--')
        ax.axvline(result.alt_mean, color='red', linestyle='--')
        ax.set_xlabel('Shared Edges Fraction')
        ax.set_ylabel('Density')
        ax.set_title(f'{name}\nEffect size: {result.effect_size:.2f}')
        ax.legend()
        
        print(f"{name}:")
        print(f"  Null mean: {result.null_mean:.4f}")
        print(f"  Alt mean:  {result.alt_mean:.4f}")
        print(f"  Effect size: {result.effect_size:.2f}")
    
    plt.tight_layout()
    plt.show()


# Helper functions
def estimate_pvalue(result):
    """Estimate p-value using permutation test logic."""
    # Count how many null values exceed the alternative mean
    null_exceeds = np.sum(np.array(result.null_values) >= result.alt_mean)
    return null_exceeds / len(result.null_values)


def generate_correlation_curve(generator, statistic, n=1000, runs=100):
    """Generate a curve showing statistic value vs correlation parameter."""
    rho_values = np.linspace(0, 1, 11)
    mean_values = []
    std_values = []
    
    simulator = TreeSimulator(generator)
    simulator.add_statistic(statistic)
    
    for rho in rho_values:
        results = simulator.run_simulation(n=n, rho=rho, num_runs=runs, progress=False)
        mean_values.append(results[0].alt_mean)
        std_values.append(np.std(results[0].alt_values))
    
    return rho_values, np.array(mean_values), np.array(std_values)


# Main execution
if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_custom_statistic()
    example_batch_analysis()
    example_model_comparison()
    
    # Bonus: Generate correlation curves for all statistics
    print("\n" + "=" * 60)
    print("Bonus: Correlation Curves for All Statistics")
    print("=" * 60)
    
    generator = PreferentialAttachmentTree()
    statistics = [
        SharedEdgesFraction(),
        LargestCommonSubtree(),
        DegreeSimilarity()
    ]
    
    plt.figure(figsize=(10, 6))
    
    for stat in statistics:
        print(f"Computing curve for {stat.name}...")
        rho_vals, means, stds = generate_correlation_curve(generator, stat, n=500, runs=50)
        plt.plot(rho_vals, means, 'o-', label=stat.name, linewidth=2)
        plt.fill_between(rho_vals, means - stds, means + stds, alpha=0.2)
    
    plt.xlabel('Correlation Parameter (ρ)')
    plt.ylabel('Mean Statistic Value')
    plt.title('Test Statistics vs Tree Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nAll examples completed!")