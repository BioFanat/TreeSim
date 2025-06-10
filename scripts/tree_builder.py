#!/usr/bin/env python3

import argparse
import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


# ============================================================================
# Tree Data Structures
# ============================================================================

class TreeNode:
    """Represents a node in a tree structure."""
    
    def __init__(self, node_id: int):
        self.id = node_id
        self.children: Set[int] = set()
        self.parent: Optional[int] = None
    
    def add_child(self, child_id: int) -> None:
        """Add a child to this node."""
        self.children.add(child_id)


# ============================================================================
# Tree Generators
# ============================================================================

class TreeGenerator(ABC):
    """Abstract base class for tree generation algorithms."""
    
    @abstractmethod
    def generate(self, n: int, random_state: Optional[np.random.RandomState] = None) -> Dict[int, TreeNode]:
        """Generate a tree with n nodes."""
        pass
    
    @abstractmethod
    def generate_correlated_pair(self, n: int, rho: float, 
                                random_state: Optional[np.random.RandomState] = None) -> Tuple[Dict[int, TreeNode], Dict[int, TreeNode]]:
        """Generate a pair of correlated trees with correlation parameter rho."""
        pass


class PreferentialAttachmentTree(TreeGenerator):
    """
    Preferential Attachment (PA) tree generator.
    
    In this model, new nodes attach to existing nodes by:
    1. Uniformly sampling an edge from the current tree
    2. Randomly choosing one of the two endpoints as the attachment point
    """
    
    def generate(self, n: int, random_state: Optional[np.random.RandomState] = None) -> Dict[int, TreeNode]:
        """
        Generate a PA tree with n nodes.
        
        Args:
            n: Number of nodes in the tree
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary mapping node IDs to TreeNode objects
        """
        if n <= 0:
            return {}
        
        if random_state is None:
            random_state = np.random.RandomState()
        
        # Initialize tree with root
        tree = [TreeNode(0)]
        edge_list = []
        
        # Add first edge if n > 1
        if n > 1:
            tree.append(TreeNode(1))
            tree[1].parent = 0
            tree[0].add_child(1)
            edge_list.append((0, 1))
        
        # Generate remaining nodes
        for t in range(2, n):
            # Sample an edge uniformly
            edge_idx = random_state.randint(0, len(edge_list))
            edge = edge_list[edge_idx]
            
            # Choose endpoint with coin flip
            chosen = edge[0] if random_state.random() < 0.5 else edge[1]
            
            # Create new node and attach
            new_node = TreeNode(t)
            new_node.parent = chosen
            tree[chosen].add_child(t)
            tree.append(new_node)
            
            # Add new edge to list
            edge_list.append((chosen, t))
        
        return {node.id: node for node in tree}
    
    def generate_correlated_pair(self, n: int, rho: float, 
                                random_state: Optional[np.random.RandomState] = None) -> Tuple[Dict[int, TreeNode], Dict[int, TreeNode]]:
        """
        Generate a pair of correlated PA trees.
        
        Args:
            n: Number of nodes in each tree
            rho: Correlation parameter (0 <= rho <= 1)
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of two tree dictionaries
        """
        if random_state is None:
            random_state = np.random.RandomState()
        
        # Generate correlated random choices
        edge_indices1 = []
        edge_indices2 = []
        coins1 = []
        coins2 = []
        
        for t in range(1, n):
            # For edge selection
            if random_state.random() < rho:
                # Use same edge index
                idx = random_state.randint(0, max(1, t))
                edge_indices1.append(idx)
                edge_indices2.append(idx)
            else:
                # Independent edge indices
                edge_indices1.append(random_state.randint(0, max(1, t)))
                edge_indices2.append(random_state.randint(0, max(1, t)))
            
            # For endpoint selection
            if random_state.random() < rho:
                # Use same coin flip
                coin = random_state.random()
                coins1.append(coin)
                coins2.append(coin)
            else:
                # Independent coin flips
                coins1.append(random_state.random())
                coins2.append(random_state.random())
        
        # Generate trees using correlated choices
        tree1 = self._generate_with_choices(n, edge_indices1, coins1)
        tree2 = self._generate_with_choices(n, edge_indices2, coins2)
        
        return tree1, tree2
    
    def _generate_with_choices(self, n: int, edge_indices: List[int], coins: List[float]) -> Dict[int, TreeNode]:
        """Generate a PA tree using predetermined random choices."""
        if n <= 0:
            return {}
        
        tree = [TreeNode(0)]
        edge_list = []
        
        if n > 1:
            tree.append(TreeNode(1))
            tree[1].parent = 0
            tree[0].add_child(1)
            edge_list.append((0, 1))
        
        for t in range(2, n):
            # Use predetermined edge index (clamp to valid range)
            edge_idx = min(edge_indices[t-1], len(edge_list) - 1)
            edge = edge_list[edge_idx]
            
            # Use predetermined coin flip
            chosen = edge[0] if coins[t-1] < 0.5 else edge[1]
            
            new_node = TreeNode(t)
            new_node.parent = chosen
            tree[chosen].add_child(t)
            tree.append(new_node)
            
            edge_list.append((chosen, t))
        
        return {node.id: node for node in tree}


class UniformAttachmentTree(TreeGenerator):
    """
    Uniform Attachment (UA) tree generator.
    
    In this model, new nodes attach uniformly to any existing node.
    """
    
    def generate(self, n: int, random_state: Optional[np.random.RandomState] = None) -> Dict[int, TreeNode]:
        """
        Generate a UA tree with n nodes.
        
        Args:
            n: Number of nodes in the tree
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary mapping node IDs to TreeNode objects
        """
        if n <= 0:
            return {}
        
        if random_state is None:
            random_state = np.random.RandomState()
        
        tree = {0: TreeNode(0)}
        
        for t in range(1, n):
            # Choose attachment point uniformly from existing nodes
            attachment_point = random_state.randint(0, t)
            
            # Create new node and attach
            new_node = TreeNode(t)
            new_node.parent = attachment_point
            tree[attachment_point].add_child(t)
            tree[t] = new_node
        
        return tree
    
    def generate_correlated_pair(self, n: int, rho: float, 
                                random_state: Optional[np.random.RandomState] = None) -> Tuple[Dict[int, TreeNode], Dict[int, TreeNode]]:
        """
        Generate a pair of correlated UA trees.
        
        Args:
            n: Number of nodes in each tree
            rho: Correlation parameter (0 <= rho <= 1)
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of two tree dictionaries
        """
        if random_state is None:
            random_state = np.random.RandomState()
        
        # Generate correlated attachment points
        attachments1 = []
        attachments2 = []
        
        for t in range(1, n):
            if random_state.random() < rho:
                # Use same attachment point
                point = random_state.randint(0, t)
                attachments1.append(point)
                attachments2.append(point)
            else:
                # Independent attachment points
                attachments1.append(random_state.randint(0, t))
                attachments2.append(random_state.randint(0, t))
        
        # Generate trees
        tree1 = self._generate_with_attachments(n, attachments1)
        tree2 = self._generate_with_attachments(n, attachments2)
        
        return tree1, tree2
    
    def _generate_with_attachments(self, n: int, attachments: List[int]) -> Dict[int, TreeNode]:
        """Generate a UA tree using predetermined attachment points."""
        if n <= 0:
            return {}
        
        tree = {0: TreeNode(0)}
        
        for t in range(1, n):
            attachment_point = attachments[t-1]
            
            new_node = TreeNode(t)
            new_node.parent = attachment_point
            tree[attachment_point].add_child(t)
            tree[t] = new_node
        
        return tree


# ============================================================================
# Test Statistics
# ============================================================================

class TestStatistic(ABC):
    """Abstract base class for test statistics on pairs of trees."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the test statistic."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the statistic measures."""
        pass
    
    @abstractmethod
    def compute(self, tree1: Dict[int, TreeNode], tree2: Dict[int, TreeNode]) -> float:
        """Compute the test statistic for a pair of trees."""
        pass


class SharedEdgesFraction(TestStatistic):
    """Fraction of edges shared between two trees."""
    
    @property
    def name(self) -> str:
        return "Shared Edges Fraction"
    
    @property
    def description(self) -> str:
        return "Fraction of edges that appear in both trees (parent-child pairs)"
    
    def compute(self, tree1: Dict[int, TreeNode], tree2: Dict[int, TreeNode]) -> float:
        """
        Compute the fraction of shared edges between two trees.
        
        Args:
            tree1: First tree
            tree2: Second tree
            
        Returns:
            Fraction of shared edges (0 to 1)
        """
        edges1 = self._get_edges(tree1)
        edges2 = self._get_edges(tree2)
        
        if not edges1:  # Empty tree
            return 0.0
        
        shared = len(edges1.intersection(edges2))
        return shared / len(edges1)
    
    def _get_edges(self, tree: Dict[int, TreeNode]) -> Set[Tuple[int, int]]:
        """Extract all edges from a tree."""
        edges = set()
        for node_id, node in tree.items():
            if node.parent is not None:
                edges.add((node.parent, node_id))
        return edges


class LargestCommonSubtree(TestStatistic):
    """Normalized size of the largest common subtree."""
    
    @property
    def name(self) -> str:
        return "Largest Common Subtree (Normalized)"
    
    @property
    def description(self) -> str:
        return "Size of the largest common subtree normalized by tree size"
    
    def compute(self, tree1: Dict[int, TreeNode], tree2: Dict[int, TreeNode]) -> float:
        """
        Compute the normalized size of the largest common subtree.
        
        This uses a dynamic programming approach to find the largest common
        subtree when both trees are rooted at node 0.
        
        Args:
            tree1: First tree
            tree2: Second tree
            
        Returns:
            Normalized size (0 to 1)
        """
        if not tree1 or not tree2:
            return 0.0
        
        lcs_size = self._compute_lcs(tree1, tree2)
        return lcs_size / len(tree1)
    
    def _compute_lcs(self, tree1: Dict[int, TreeNode], tree2: Dict[int, TreeNode]) -> int:
        """Compute the size of the largest common subtree."""
        dp = {}
        
        def recurse(u: int, v: int) -> int:
            if (u, v) in dp:
                return dp[(u, v)]
            
            # Get sorted children lists
            children_u = sorted(list(tree1[u].children))
            children_v = sorted(list(tree2[v].children))
            m, n = len(children_u), len(children_v)
            
            # DP for weighted longest common subsequence
            dp_lcs = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    weight = recurse(children_u[i-1], children_v[j-1])
                    dp_lcs[i][j] = max(
                        dp_lcs[i-1][j],
                        dp_lcs[i][j-1],
                        dp_lcs[i-1][j-1] + weight
                    )
            
            # Include current node
            dp[(u, v)] = 1 + dp_lcs[m][n]
            return dp[(u, v)]
        
        return recurse(0, 0)  # Both trees rooted at 0


class RootedSubtreeSimilarity(TestStatistic):
    """Average similarity of all rooted subtrees."""
    
    @property
    def name(self) -> str:
        return "Rooted Subtree Similarity"
    
    @property
    def description(self) -> str:
        return "Average fraction of nodes with matching rooted subtree structures"
    
    def compute(self, tree1: Dict[int, TreeNode], tree2: Dict[int, TreeNode]) -> float:
        """
        Compute the average similarity of rooted subtrees.
        
        For each node that exists in both trees, check if their subtree
        structures match.
        
        Args:
            tree1: First tree
            tree2: Second tree
            
        Returns:
            Average similarity (0 to 1)
        """
        if not tree1 or not tree2:
            return 0.0
        
        # Get subtree hashes for each tree
        hashes1 = self._compute_subtree_hashes(tree1)
        hashes2 = self._compute_subtree_hashes(tree2)
        
        # Count matching subtrees
        matches = 0
        for node_id in tree1:
            if node_id in tree2 and hashes1[node_id] == hashes2[node_id]:
                matches += 1
        
        return matches / len(tree1)
    
    def _compute_subtree_hashes(self, tree: Dict[int, TreeNode]) -> Dict[int, str]:
        """Compute canonical hash for each rooted subtree."""
        hashes = {}
        
        def compute_hash(node_id: int) -> str:
            # Get hashes of children
            child_hashes = []
            for child_id in sorted(tree[node_id].children):
                child_hashes.append(compute_hash(child_id))
            
            # Create canonical representation
            canonical = f"({','.join(sorted(child_hashes))})"
            hashes[node_id] = canonical
            return canonical
        
        compute_hash(0)
        return hashes


# ============================================================================
# Simulation Engine
# ============================================================================

@dataclass
class SimulationResult:
    """Results from a simulation run."""
    n: int
    rho: float
    tree_type: str
    statistic_name: str
    null_values: List[float]
    alt_values: List[float]
    execution_time: float
    
    @property
    def null_mean(self) -> float:
        return np.mean(self.null_values)
    
    @property
    def alt_mean(self) -> float:
        return np.mean(self.alt_values)
    
    @property
    def effect_size(self) -> float:
        """Cohen's d effect size."""
        null_std = np.std(self.null_values)
        if null_std == 0:
            return 0.0
        return (self.alt_mean - self.null_mean) / null_std


class TreeSimulator:
    """Main simulation engine for tree correlation analysis."""
    
    def __init__(self, tree_generator: TreeGenerator, random_seed: Optional[int] = None):
        """
        Initialize the simulator.
        
        Args:
            tree_generator: Tree generation algorithm to use
            random_seed: Random seed for reproducibility
        """
        self.generator = tree_generator
        self.random_state = np.random.RandomState(random_seed)
        self.statistics: List[TestStatistic] = []
    
    def add_statistic(self, statistic: TestStatistic) -> None:
        """Add a test statistic to compute."""
        self.statistics.append(statistic)
    
    def run_simulation(self, n: int, rho: float, num_runs: int = 1000,
                      progress: bool = True) -> List[SimulationResult]:
        """
        Run simulation for given parameters.
        
        Args:
            n: Number of nodes in each tree
            rho: Correlation parameter
            num_runs: Number of simulation runs
            progress: Show progress bar
            
        Returns:
            List of SimulationResult objects (one per statistic)
        """
        results = []
        tree_type = type(self.generator).__name__
        
        for stat in self.statistics:
            print(f"\nComputing {stat.name}...")
            start_time = time.time()
            
            null_values = []
            alt_values = []
            
            iterator = tqdm(range(num_runs)) if progress else range(num_runs)
            
            for _ in iterator:
                # Generate null distribution (rho=0)
                tree1_null, tree2_null = self.generator.generate_correlated_pair(n, 0, self.random_state)
                null_values.append(stat.compute(tree1_null, tree2_null))
                
                # Generate alternative distribution
                tree1_alt, tree2_alt = self.generator.generate_correlated_pair(n, rho, self.random_state)
                alt_values.append(stat.compute(tree1_alt, tree2_alt))
            
            execution_time = time.time() - start_time
            
            result = SimulationResult(
                n=n,
                rho=rho,
                tree_type=tree_type,
                statistic_name=stat.name,
                null_values=null_values,
                alt_values=alt_values,
                execution_time=execution_time
            )
            
            results.append(result)
            
            # Print summary
            print(f"  Null distribution: mean={result.null_mean:.4f}, "
                  f"std={np.std(result.null_values):.4f}")
            print(f"  Alt distribution:  mean={result.alt_mean:.4f}, "
                  f"std={np.std(result.alt_values):.4f}")
            print(f"  Effect size: {result.effect_size:.4f}")
            print(f"  Execution time: {execution_time:.2f}s")
        
        return results


# ============================================================================
# Visualization
# ============================================================================

def plot_distributions(results: List[SimulationResult], save_path: Optional[str] = None) -> None:
    """
    Plot distribution comparisons for all statistics.
    
    Args:
        results: List of simulation results
        save_path: Path to save the plot (if None, display interactively)
    """
    n_stats = len(results)
    fig, axes = plt.subplots(1, n_stats, figsize=(6*n_stats, 6))
    
    if n_stats == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results):
        # Plot distributions
        sns.kdeplot(data=result.null_values, label=f'Null (ρ=0)', 
                   color='blue', fill=True, alpha=0.3, ax=ax)
        sns.kdeplot(data=result.alt_values, label=f'Alternative (ρ={result.rho})', 
                   color='red', fill=True, alpha=0.3, ax=ax)
        
        # Add vertical lines for means
        ax.axvline(result.null_mean, color='blue', linestyle='--', alpha=0.7)
        ax.axvline(result.alt_mean, color='red', linestyle='--', alpha=0.7)
        
        # Labels and title
        ax.set_xlabel(result.statistic_name)
        ax.set_ylabel('Density')
        ax.set_title(f'{result.statistic_name}\n(n={result.n}, effect size={result.effect_size:.3f})')
        ax.legend()
    
    plt.suptitle(f'{results[0].tree_type} Tree Correlation Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate correlated trees and compute similarity statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with PA trees
  %(prog)s --n 1000 --rho 0.5 --tree-type pa
  
  # Run with all statistics and save results
  %(prog)s --n 500 --rho 0.3 --runs 5000 --all-stats --save-plot output.png
  
  # Use specific statistics with UA trees
  %(prog)s --n 1000 --rho 0.7 --tree-type ua --stats shared-edges lcs
  
  # Set random seed for reproducibility
  %(prog)s --n 1000 --rho 0.5 --seed 42 --json results.json
        """
    )
    
    # Required arguments
    parser.add_argument('--n', type=int, required=True,
                       help='Number of nodes in each tree')
    parser.add_argument('--rho', type=float, required=True,
                       help='Correlation parameter (0 <= rho <= 1)')
    
    # Tree type
    parser.add_argument('--tree-type', type=str, choices=['pa', 'ua'], default='pa',
                       help='Tree type: preferential attachment (pa) or uniform attachment (ua)')
    
    # Statistics selection
    parser.add_argument('--stats', nargs='+', 
                       choices=['shared-edges', 'lcs', 'subtree-sim'],
                       help='Statistics to compute (default: shared-edges)')
    parser.add_argument('--all-stats', action='store_true',
                       help='Compute all available statistics')
    
    # Simulation parameters
    parser.add_argument('--runs', type=int, default=1000,
                       help='Number of simulation runs (default: 1000)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    
    # Output options
    parser.add_argument('--save-plot', type=str, help='Save plot to file')
    parser.add_argument('--json', type=str, help='Save results to JSON file')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 <= args.rho <= 1:
        parser.error("rho must be between 0 and 1")
    
    if args.n <= 0:
        parser.error("n must be positive")
    
    # Create tree generator
    if args.tree_type == 'pa':
        generator = PreferentialAttachmentTree()
    else:
        generator = UniformAttachmentTree()
    
    # Create simulator
    simulator = TreeSimulator(generator, random_seed=args.seed)
    
    # Add statistics
    available_stats = {
        'shared-edges': SharedEdgesFraction(),
        'lcs': LargestCommonSubtree(),
        'subtree-sim': RootedSubtreeSimilarity()
    }
    
    if args.all_stats:
        for stat in available_stats.values():
            simulator.add_statistic(stat)
    else:
        stats_to_use = args.stats if args.stats else ['shared-edges']
        for stat_name in stats_to_use:
            simulator.add_statistic(available_stats[stat_name])
    
    # Run simulation
    print(f"\nRunning {args.tree_type.upper()} tree simulation:")
    print(f"  Nodes: {args.n}")
    print(f"  Correlation: {args.rho}")
    print(f"  Runs: {args.runs}")
    print(f"  Statistics: {[s.name for s in simulator.statistics]}")
    
    results = simulator.run_simulation(
        n=args.n,
        rho=args.rho,
        num_runs=args.runs,
        progress=not args.no_progress
    )
    
    # Save JSON results if requested
    if args.json:
        json_data = []
        for result in results:
            json_data.append({
                'n': result.n,
                'rho': result.rho,
                'tree_type': result.tree_type,
                'statistic': result.statistic_name,
                'null_mean': result.null_mean,
                'null_std': float(np.std(result.null_values)),
                'alt_mean': result.alt_mean,
                'alt_std': float(np.std(result.alt_values)),
                'effect_size': result.effect_size,
                'execution_time': result.execution_time
            })
        
        with open(args.json, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"\nResults saved to: {args.json}")
    
    # Plot results
    if not args.no_plot:
        plot_distributions(results, save_path=args.save_plot)


if __name__ == "__main__":
    main()