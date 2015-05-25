# What is this repository for?
This repository contains python implementation of rational Krylov methods (ALR, KPIK, RKSM) for obtaining approximation of solution of continuous-time Lyapunov equation. The main computational costs for all methods are linear system solutions, so algebraic multigrid techniques can be applyed for methods acceleration instead of scipy direct solver. 

KPIK method is described in the paper [V. Simoncini, A new iterative method for solving large-scale Lyapunov matrix equations](http://dx.doi.org/10.1137/06066120X)

RKSM method is described in the paper [V. Druskin, C. Lieberman, M. Zaslavsky, adaptive choice of shifts in rational Krylov subspace reduction of evolutionary problems](http://dx.doi.org/10.1137/090774082)

The Python code for KPIK and RKMS is based on the publicly available MATLAB implementation http://www.dm.unibo.it/~simoncin/software.html

# Citing 
ALR method is described in paper 

    @article{kolesnikov2014low,
      title={From low-rank approximation to an efficient rational Krylov subspace method for the Lyapunov equation},
      author={Kolesnikov, D. A. and Oseledets, I. V.},
      journal={arXiv preprint arXiv:1410.3335},
      year={2014}
    }

# Documentation
will be available soon.

# Requirements
You need only numpy and scipy to run example, since there are pure python functions.
For algebraic multigrid usage [PyAMG library](https://github.com/pyamg/pyamg/) is required.

# Python version support
Current implementation was succesfully tested with numpy 1.9.2, scipy 0.15.1 and ipython-notebook 3.1.0 for Python 2.7.9

Main Contributor
Denis Kolesnikov d4kolesnikov@yandex.ru
