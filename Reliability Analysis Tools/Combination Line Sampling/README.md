# Combination line sampling

A MATLAB and Python 3 software for the computation of rare event probabilities for reliability assessment using the combination line sampling (CLS) approach. The CLS method estimates the probability of failure as a combination of line sampling estimators corresponding to important directions chosen adaptively throughout the simulation. Sampling is performed on a hyperplane perpendicular to the important directions and the contribution of each sample is evaluated through computing the distance of the sample to the limit state surface.

If a good initial direction is chosen, the method performs similarly to the classical line sampling method, and can be efficiently applied to moderately nonlinear high-dimensional problems. However, the adaptation of a poorly chosen initial direction is better suited for problems with low to moderate dimensions; the potential of improving the initial direction through an adaptive process decreases as the dimension of the space increases.

For each sample, the distance to the limit-state surface is found either by: (1) a line search algorithm which alternates between secant and regula falsi updates, or (2) using a quadratic response surface approximation of the limit state function (LSF) along the sampled line fitted to a fixed number of LSF evaluations.

Required input:

- Point in U-space that defines the initial direction
- Desired number of samples
- Limit state function passed as a handle to a user-defined MATLAB function
- Input distribution

The software returns:

- Pf_LS:  estimate of the failure probability
- cv_LS:  estimate of the coefficient of variation of the probability estimate
- u_new:  final selected point in U-space (estimate for the FORM design point)
- x_new:  same point in X-space
- n_eval: the total number of LSF-evaluations

## Requirements

MATLAB, incl. Statistical toolbox, ERADist and ERANataf probability distribution classes

Python 3, incl. numpy, scipy, ERADist and ERANataf probability distribution classes

## Documentation & background

Papaioannou I. and Straub D. (2021): Combination line sampling for structural reliability analysis. Structural Safety 88:102025.


## Version

Last change: 06/2021

