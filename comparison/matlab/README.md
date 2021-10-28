# Comparison with MATLAB codes

This directory contains micro-benchmarks for functions for which equivalent functionality is available in MATLAB codes provided
by corresponding researchers. We only provide micro-benchmarks for our implementation in `CR-Sparse` here. Benchmarking
for corresponding MATLAB code was done separately.

## Lanczos Bidiagonalization with Partial Reorthogonalization

This routine is called in short as `lanbpro`.  MATLAB/C implementation is available in 
[PROPACK](https://github.com/andrewssobral/PROPACK).

- [lanbpro on a very large matrix with small number of non-zero singular values](lanbpro/lanbpro_large_matrix_test.ipynb) [open in nbviewer](https://nbviewer.org/github/carnotresearch/cr-sparse-companion/blob/main/comparison/matlab/lanbpro/lanbpro_large_matrix_test.ipynb)