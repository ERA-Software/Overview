# Bayesian Updating with Subset Simulation

MATLAB and Python 3 software for Bayesian inference of engineering models using Bayesian updating with structural reliability (BUS) methods. The BUS approach is based on an equivalent formulation of the Bayesian updating problem in terms of a structural reliability problem. By using this method, a failure event (in the context of structural reliability) can be related to an observation event (in the context of Bayesian updating), and thus, algorithms developed for structural reliability can be applied to obtain samples from the posterior distribution. BUS requires the choice of a constant c such that 1/c is not smaller than the maximum of the likelihood function. In the current implementation BUS is combined with Subset Simulation (SuS), which is an adaptive Monte Carlo method that performs efficiently in high-dimensional problems.

</br></br> 

Required input: 
* Number of samples per subset level
* Intermediate conditional probability
* Constant c
* Likelihood function passed as a handle to a user-defined MATLAB function
* Prior distribution defined as an object of the ERANataf class

The software returns:
* Intermediate subset levels
* Samples from the posterior distribution in the standard space
* Samples from the posterior distribution in the original space
* An estimate of the model evidence
* Scaling of the proposal at the last level



## Requirements

MATLAB, incl. Statistical toolbox, ERADist and ERANataf probability distribution classes

Python 3


## Documentation & background

Straub D., Papaioannou I. (2015): Bayesian updating with structural reliability methods. Journal of Engineering Mechanics, 141(3), 04014134.

Au S.-K., Beck J.L. (2001): Estimation of small failure probabilities in high dimensions by subset simulation. Probabilistic Engineering Mechanics, ASCE, 16: 263-277.

Papaioannou I., Betz W., Zwirglmaier K., Straub D. (2015): MCMC algorithms for subset simulation. Probabilistic Engineering Mechanics, 41: 89-103.


## Version

Last change: 03/2021


