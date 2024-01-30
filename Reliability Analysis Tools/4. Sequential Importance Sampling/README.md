# Sequential Importance Sampling

A MATLAB and Python 3 software which computes the probability of failure for engineering models using sequential importance sampling (SIS). SIS is an adaptive importance sampling method that generates samples from a sequence of distributions that gradually approach the optimal importance sampling density. The distribution sequence is constructed through introducing a smooth approximation of the indicator function of the failure domain in the expression for the optimal importance sampling density. Once this optimal density is approached sufficiently, the method estimates the probability of failure through importance sampling with the final importance sampling density.

Sampling from the distribution sequence is performed through a resample-move scheme. In the current implementation, the move step is based on either of two Markov chain Monte Carlo (MCMC) samplers: (1) An adaptive conditional sampling (aCS) algorithm that is efficient for application in high dimensional spaces (2) An independent Metropolis-Hastings algorithm with proposal density obtained by fittig a Gaussian mixture model with the current weighted samples using a modified expectation maximization algorithm.

Optionally, a reliability sensitivity analysis can be run after convergence based on the obtained failure sample. In particular, the method returns variance-based sensitivity indices of the limit-state function.
Required input:

- Number of samples per subset level
- Quantile value to select samples for parameter update
- Limit state function passed as a handle to a user-defined MATLAB function
- Distribution of the random variables defined as an object of the ERANataf class
- Burn-in period
- Target coefficient of variation of the weights
- Option for which samples to return (0 - none, 1 - final sample, 2 - all samples)
- Logical variable indicating whether to return reliability sensitivities

The software returns:

- An estimate of the failure probability of the system
- Number of intermediate levels
- Optional: samples in the standard Gaussian and original probability space
- Optional: reliability sensitivities

## Requirements

MATLAB, incl. Statistical toolbox, ERADist and ERANataf probability distribution classes

Python 3, incl. numpy, scipy, matplotlib, ERADist and ERANataf probability distribution classes

## Documentation & background

Papaioannou I., Papadimitriou, C., Straub D. (2016): Sequential importance sampling for structural reliability analysis. Structural Safety 62: 66-75.

Papaioannou I., Betz W., Zwirglmaier K., Straub D. (2015): MCMC algorithms for subset simulation. Probabilistic Engineering Mechanics 41: 89-103.

Li L., Papaioannou I., Straub D. (2019): Global reliability sensitivity estimation based on failure samples. Structural Safety 81: 101871.

Straub, D., Ehre, M., & Papaioannou, I. (2022). Decision-theoretic reliability sensitivity. Reliability Engineering & System Safety, 221, 108215.


## Version

Last change: 05/2023

