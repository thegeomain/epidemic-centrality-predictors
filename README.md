# Epidemic Centrality Predictors

This repository contains the code used in the study

**Comparing the Effectiveness of Degree, Spectral, k-Core, and Hybrid Structural Predictors at Predicting Epidemic Outbreak Size**

The code in this repository generates synthetic networks, runs SIR simulations, computes node centrality measures, and calculates correlations between centrality metrics and epidemic outbreak size. The scripts also include the values used to produce the figures in the manuscript.

## Usage
Running experiments.py prints the experimental results for the correlations of Perron eigenvector entries with epidemic outbreak size.
Running experiments_kcore.py does the same with the k-core metric. Running experiments_degrees.py does the same with the degree metric. Running experiments_hybrid.py does the same with the hybrid metric. Running real_world_experiments.py does the correlations for real-world networks, and running net_visual.py givea a sample visualization of the synthetic networks used. Running figures.py returns some of the figures used in the paper, where correlation values are hardcoded for convenience.

## Data Availability
All data in this study arise from stochastic simulations (random graph generation, SIR dynamics, and permutation tests). Exact numerical results cannot be reproduced because random seeds were not preserved, but statistically equivalent datasets can be regenerated using the scripts in this repository.

## License
This code is released for transparency and academic reproducibility. No warranty is provided.
