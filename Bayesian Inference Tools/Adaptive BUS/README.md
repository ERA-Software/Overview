# Bayesian Updating with Subset Simulation

MATLAB and Python 3 software for Bayesian inference of engineering models using an adaptive version of the BUS approach (aBUS). In the standard version of BUS the choice of a constant c is required such that 1/c is not smaller than the maximum of the likelihood function. The aBUS method combines the BUS approach with the SuS method in such a way that the constant c is no longer required as input. The aBUS approach is well-suited for problems with many uncertain parameters and for problems where it is computationally demanding to evaluate the likelihood function.

Required input:
* Number of samples per subset level
* Intermediate conditional probability
* Natural logarithm of the likelihood function passed as a handle to a user-defined MATLAB function
* Prior distribution defined as an object of the ERANataf class

The software returns:
* Samples from the posterior distribution in the standard space
* Samples from the posterior distribution in the original space
* An estimate of the model evidence
* Constant c
* Scaling of the proposal at the last level



## Requirements

MATLAB, incl. Statistical toolbox, ERADist and ERANataf probability distribution classes

Python 3


## Documentation & background

Betz W., Papaioannou I., Beck J.L., Straub D. (2018): Bayesian inference with subset simulation: Strategies and improvements. Computer Methods in Applied Mechanics and Engineering 331, 72-93.

Straub D., Papaioannou I. (2015): Bayesian updating with structural reliability methods. Journal of Engineering Mechanics, 141(3), 04014134.

Au S.-K., Beck J.L. (2001): Estimation of small failure probabilities in high dimensions by subset simulation. Probabilistic Engineering Mechanics, ASCE, 16: 263-277.

Papaioannou I., Betz W., Zwirglmaier K., Straub D. (2015): MCMC algorithms for subset simulation. Probabilistic Engineering Mechanics, 41: 89-103.


## Version

Last change: 03/2021


