# Cross-Entropy Bayesian Updating

MATLAB and Python 3 software for Bayesian inference of engineering models using the cross-entropy Bayesian updating (CEBU) method. The CEBU method employs the cross-entropy method to determine an effective importance sampling estimate of the target posterior distribution. Herein, intermediate distributions defined by some tempering parameters that gradually approach the target posterior distribution are introduced to efficiently solve the CE optimization problem. The tempering parameters are adaptively selected for fast convergence. The Gaussian mixture (GM) and the von-Mises-Fischer-Nakagami mixture families are implemented as parametric distributions for the CE method. 

Required input:
* Number of samples per level
* Log-likelihood function
* Maximum number of iterations
* Prior distribution defined as an object of the ERANataf class
* Target correlation of variation of weights
* Initial number of marginals in the mixture model
* Flag indicating the return of samples
Optional:
* Number of iid samples to generate in the last level

The software returns:
* Samples from the posterior distribution in standard space
* Samples from the posterior distribution in original space
* Intermediate distribution parameters
* Intermediate values of the tempering parameter
* Final number of marginals in the mixture
* An estimate of the model evidence
* Normed weights of the last level samples
* Iid samples of the last level


## Requirements

MATLAB, incl. Statistical toolbox, ERADist and ERANataf probability distribution classes

Python 3


## Documentation & background

Engel, M., Kanjilal, O., Papaioannou, I., & Straub, D. (2023): Bayesian Updating and Marginal Likelihood Estimation
by Cross Entropy based Importance Sampling. Journal of Computational Physics, 473: 111746.



## Version

Last change: 10/2024


