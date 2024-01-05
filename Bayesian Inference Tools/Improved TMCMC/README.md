# Improved Transitional MCMC for Bayesian Inference

MATLAB and Python 3 software for Bayesian inference of engineering models using the improved Transitional Markov Chain Monte Carlo (iTMCMC) method. The TMCMC method samples a sequence of intermediate distributions defined by some tempering parameters that gradually approach the target posterior distribution. Based on the original TMCMC method, the iTMCMC algorithm implements the following modifications: (1) Adjust the sample weights after each MCMC step in order to reduce the average bias of the estimation of the model evidence; (2) Apply a burn-in period in the MCMC step in order to improve the posterior approximation; and (3) Adaptive selection of the scale of the proposal distribution of the MCMC algorithm in order to achieve a near-optimal acceptance rate (this requires performing the Bayesian inference in an underlying standard Gaussian space).

Required input:
* Number of samples per level
* Burn-in period
* Logarithm of the likelihood function passed as a handle to a user-defined MATLAB function
* Prior distribution defined as an object of the ERANataf class

The software returns:
* Samples from the posterior distribution in standard space
* Samples from the posterior distribution in original space
* Intermediate values of the tempering parameter
* An estimate of the model evidence



## Requirements

MATLAB, incl. Statistical toolbox, ERADist and ERANataf probability distribution classes

Python 3


## Documentation & background

Ching. J, Chen Y.-C. (2007): Transitional Markov chain Monte Carlo method for Bayesian model updating,model class selection, and model averaging. Journal of Engineering Mechanics, ASCE, 133(7): 816-832.

Betz W., Papaioannou I., Straub D. (2016): Transitional Markov chain Monte Carlo: Observations and improvements. Journal of Engineering Mechanics, ASCE, 142(5): 04016016.


## Version

Last change: 03/2021


