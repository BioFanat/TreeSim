#!/bin/bash
#SBATCH --mem=12G
#SBATCH --output=script_out/example.out
#SBATCH --error=script_out/example.err

# Example SLURM script for running k-mer extraction
# Adjust paths as needed

python tree_correlation_analyzer.py --n 1000 --rho 0.5 --all-stats --save-plot output.png --json results.json
