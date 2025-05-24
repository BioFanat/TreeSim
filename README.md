# TreeSim

A Python framework for generating pairs of correlated trees and analyzing their structural similarities through various test statistics. TreeSim provides an experimental platform for studying tree correlation, developing new similarity measures, and understanding how different tree generation models behave under correlation.

## Overview

TreeSim implements algorithms to generate correlated tree pairs using two fundamental attachment models:

- **Preferential Attachment (PA)**: New nodes attach by uniformly selecting an edge from the existing tree, then randomly choosing one of its endpoints
- **Uniform Attachment (UA)**: New nodes attach by uniformly selecting a vertex from the existing tree

The framework provides three test statistics to measure tree similarity:
1. **Shared Edges Fraction**: Proportion of parent-child pairs that appear in both trees
2. **Largest Common Subtree (LCS)**: Size of the largest common rooted subtree, normalized by tree size
3. **Rooted Subtree Similarity**: Average fraction of nodes with matching rooted subtree structures

## Mathematical Theory

### Tree Generation Models

#### Preferential Attachment (PA) Trees

The PA model generates trees where nodes with more connections are more likely to receive new connections. Starting with a root node (0), for each new node t ∈ {1, 2, ..., n-1}:

1. Sample an edge (u, v) uniformly at random from all edges in the current tree
2. With probability 0.5, attach node t to u; otherwise attach to v
3. Add the new edge (parent, t) to the tree

This process creates trees where high-degree nodes attract more connections, leading to hub-like structures.

#### Uniform Attachment (UA) Trees

The UA model generates trees through uniform random attachment. Starting with a root node (0), for each new node t ∈ {1, 2, ..., n-1}:

1. Sample a node u uniformly at random from {0, 1, ..., t-1}
2. Attach node t to u as its parent

This process creates more balanced trees compared to PA, as each existing node has equal probability of receiving new connections.

### Correlation Mechanism

To generate correlated tree pairs (T₁, T₂) with correlation parameter ρ ∈ [0, 1], TreeSim uses a coupling approach where for each new node t during tree construction, both trees select the same parent node with probability ρ; otherwise, both trees will independently select parent nodes with probability (1-ρ).

This coupling mechanism ensures that:
- When ρ = 0: Trees are generated independently
- When ρ = 1: Trees are identical (perfect correlation)
- When 0 < ρ < 1: Trees share some structural decisions, creating partial similarity

## Test Statistics

### 1. Shared Edges Fraction

Measures the proportion of edges that appear in both trees:

```
SEF(T₁, T₂) = |E(T₁) ∩ E(T₂)| / |E(T₁)|
```

Where E(T) denotes the edge set of tree T. This statistic directly captures how many parent-child relationships are preserved between trees.

### 2. Largest Common Subtree (LCS)

Computes the size of the largest common rooted subtree, normalized by tree size:

```
LCS(T₁, T₂) = |MaxCommonSubtree(T₁, T₂)| / n
```

The algorithm uses dynamic programming to find the maximum weighted common subsequence of children at each node, where weights are the sizes of matching subtrees.

### 3. Rooted Subtree Similarity

Measures the fraction of nodes whose rooted subtrees have identical structure:

```
RSS(T₁, T₂) = |{v ∈ V : Subtree(T₁, v) ≅ Subtree(T₂, v)}| / n
```

Where Subtree(T, v) denotes the rooted subtree at node v, and ≅ denotes structural isomorphism. The implementation uses canonical hashing to efficiently compare subtree structures.

## Usage

### Command Line Interface

TreeSim provides a comprehensive CLI for generating and analyzing correlated trees:

```bash
# Basic usage with PA trees
python scripts/tree_builder.py --n 1000 --rho 0.5 --tree-type pa

# Run with all statistics and save results
python scripts/tree_builder.py --n 500 --rho 0.3 --runs 5000 --all-stats --save-plot output.png

# Use specific statistics with UA trees
python scripts/tree_builder.py --n 1000 --rho 0.7 --tree-type ua --stats shared-edges lcs

# Set random seed for reproducibility
python scripts/tree_builder.py --n 1000 --rho 0.5 --seed 42 --json results.json
```

#### Key Parameters

- `--n`: Number of nodes in each tree (required)
- `--rho`: Correlation parameter, 0 ≤ ρ ≤ 1 (required)
- `--tree-type`: Choose `pa` (preferential attachment) or `ua` (uniform attachment)
- `--stats`: Select specific statistics: `shared-edges`, `lcs`, `subtree-sim`
- `--all-stats`: Compute all available statistics
- `--runs`: Number of simulation runs (default: 1000)
- `--seed`: Random seed for reproducibility
- `--save-plot`: Save distribution plots to file
- `--json`: Export results to JSON format

### Programmatic Usage

```python
from scripts.tree_builder import (
    PreferentialAttachmentTree, UniformAttachmentTree,
    TreeSimulator, SharedEdgesFraction, LargestCommonSubtree,
    RootedSubtreeSimilarity, plot_distributions
)

# Create a tree generator
generator = PreferentialAttachmentTree()

# Create simulator
simulator = TreeSimulator(generator, random_seed=42)

# Add statistics
simulator.add_statistic(SharedEdgesFraction())
simulator.add_statistic(LargestCommonSubtree())
simulator.add_statistic(RootedSubtreeSimilarity())

# Run simulation
results = simulator.run_simulation(n=1000, rho=0.5, num_runs=1000)

# Analyze results
for result in results:
    print(f"{result.statistic_name}:")
    print(f"  Null mean: {result.null_mean:.4f}")
    print(f"  Alt mean: {result.alt_mean:.4f}")
    print(f"  Effect size: {result.effect_size:.4f}")

# Visualize distributions
plot_distributions(results, save_path="analysis.png")
```

See `scripts/examples/` for more examples

### Future Directions

- Current: adding Numba support for increased performance in tree creation and analysis. 
- Current: integration of parallel analyses (i.e. direct profiler to compare results for given test statistics across different values of n and ρ)
- Ongoing: reading through literature for more efficient algorithms to calculate more complex statistics across large-n trees.
