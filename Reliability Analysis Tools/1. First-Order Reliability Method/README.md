# First-Order Reliability Method

A MATLAB and Python 3 software for reliability analysis using the First-Order Reliability Method (FORM). After transforming the random variables to an equivalent standard Gaussian space (transformed space), FORM approximates the limit state function by a linear (first-order polynomial) using a Taylor series expansion at the so-called design point. The design point is the point of the failure domain in the transformed space with highest probability density value. It is found by solving an optimization problem. Particularly, the Hasofer-Lind-Rackwitz-Fiessler (HLRF) algorithm is an iterative gradient-based optimization approach adapted to this type of reliability problems. It has been shown to be effective in many situations even if its convergence is not assured in all cases. Once the design point is computed, variance-based sensitivity indices (Sobol' and total Sobol' indices) of the linear LSF approximation underlying the FORM approach can be computed analytically.

Required input:

- Limit state function passed as a handle to a user-defined MATLAB function
- Gradient of the limit state function passed as a handle to a user-defined MATLAB function
- Distribution of the random variables defined as an object of the ERANataf class
- Logical variable indicating whether to return reliability sensitivities of the FORM solution

Optional input:

- Parameters of the design point search algorithm: initial value, step size, termination tolerance

The software returns:

- Design point in the standard space
- Design point in the original space
- Reliability index
- Probability of failure estimate
- First-order Sobol' and total-effect Sobol' indices if sensitivity analysis is toggled


## Requirements

MATLAB, incl. Statistical toolbox, ERADist and ERANataf probability distribution classes (optional: Deep Learning Toolbox)

Python 3, incl. numpy, scipy, matplotlib, ERADist and ERANataf probability distribution classes (optional autograd)

## Documentation & background

Rackwitz, R., and B. Fiessler (1979): Structural reliability under combined random load sequences. Computers and Structures 9(5): 489-494.

Papaioannou I., Straub D. (2021): Variance-based reliability sensitivity analysis and the FORM α-factors. Reliability Engineering and System Safety 210: 107496.


## Version

Last change: 05/2023

