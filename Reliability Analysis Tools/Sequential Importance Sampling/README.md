# Subset Simulation

A MATLAB software for the computation of small failure probabilities using Subset Simulation (SuS) with the adaptive Conditional Sampling (aCS) approach. The SuS method is an adaptive simulation procedure for efficiently computing small failure probabilities in engineering systems. The failure probability is expressed as a product of large values of conditional probabilities by introducing several intermediate failure events. Conditional samples on each intermediate failure event are simulated using Markov Chain Monte Carlo (MCMC) methods. In the current implementation, MCMC is performed with the adaptive Conditional Sampling (aCS) algorithm. The performance of aCS is independent of the dimension of the random variable space and is hence suitable for reliability assessment in high dimensions. Moreover aCS allows accounting for the relative influence of each random variable, based on the statistics of the seeds of the Markov chains at each failure level.

Optionally, a reliability sensitivity analysis can be run after convergence based on the obtained failure sample. In particular, the method returns variance-based sensitivity indices of the limit-state function.

Required input:

- Number of samples per subset level
- Intermediate conditional probability
- Limit state function passed as a handle to a user-defined MATLAB function
- Distribution of the random variables defined as an object of the ERANataf class
- Option for which samples to return (0 - none, 1 - final sample, 2 - all samples)
- Logical variable indicating whether to return reliability sensitivities

The software returns:

- An estimate of the failure probability of the system
- An estimate of the coefficient of variation of the probability estimate
- Probability estimates for intermediate levels
- Optional: samples in the standard Gaussian and original probability space
- Optional: reliability sensitivities

## Requirements

MATLAB, incl. Statistical toolbox, ERADist and ERANataf probability distribution classes

Python 3, incl. numpy, scipy, matplotlib, ERADist and ERANataf probability distribution classes

## Documentation & background

Au S.-K., Beck J.L. (2001): Estimation of small failure probabilities in high dimensions by subset simulation. Probabilistic Engineering Mechanics 16: 263-277.

Papaioannou I., Betz W., Zwirglmaier K., Straub D. (2015): MCMC algorithms for subset simulation. Probabilistic Engineering Mechanics 41: 89-103.

Li L., Papaioannou I., Straub D. (2019): Global reliability sensitivity estimation based on failure samples. Structural Safety 81: 101871.


## Version

Last change: 05/2023

