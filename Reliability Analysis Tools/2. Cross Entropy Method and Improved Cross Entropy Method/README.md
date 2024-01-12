# Cross Entropy Method and Improved Cross Entropy Method

A MATLAB and Python 3 software for the computation of rare event probabilities using the cross entropy method (CE) or improved cross entropy method (iCE) with different distribution families employed as parametric importance sampling (IS) densities. The cross entropy method is an adaptive sampling approach that determines the sampling density through minimizing the Kullback-Leibler divergence between the theoretically optimal importance sampling density and a chosen parametric family of distributions. The improved cross entropy method introduces a smooth transition to make better use of the samples from intermediate sampling levels for fitting the sought IS density.

Three different distribution families are implemented in the code: (1) The multivariate normal distribution (single Gaussian (SG)); (2) A mixture of several multivariate normal distributions (Gaussian mixture (GM)). These two distribution families work only in low dimensions, an application in dimensions larger than 20 is not recommended. (3) A newly developed flexible mixture model which couples the von Mises-Fisher distribution and the Nakagami distribution (von Mises-Fisher-Nakagami mixture (vMFNM)). This model is based on a polar coordinate transformation of the standard normal space and works well in low to moderately high dimensions. For the parameter updating of the Gaussian mixture and the von Mises-Fisher-Nakagami mixture, a modified version of the expectation-maximization algorithm is used that works with weighted samples.

Optionally, a reliability sensitivity analysis can be run after convergence based on the obtained failure sample. In particular, the method returns variance-based sensitivity indices of the limit-state function.

Required input:

- Number of samples per level
- Quantile value to select samples for parameter update (only required when using CE)
- Distribution of the random variables defined as an object of the ERAdist or ERANataf classes
- Limit state function passed as a handle to a user-defined MATLAB function
- Initial number of components in the mixture model (when using GM or vMFNM)
- Option for which samples to return (0 - none, 1 - final sample, 2 - all samples)
- Logical variable indicating whether to return reliability sensitivities

The software returns:

- An estimate of the failure probability of the system
- Number of intermediate levels
- Total number of samples
- Intermediate levels
- Final number of components in the mixture (when using GM or vMFNM)
- Optional: samples in the standard Gaussian space
- Optional: samples in the original space
- Optional: reliability sensitivities


## Requirements

MATLAB, incl. Statistical toolbox, ERADist and ERANataf probability distribution classes

Python 3, incl. numpy, scipy, matplotlib, ERADist and ERANataf probability distribution classes


## Documentation & background

Geyer S., Papaioannou I., Straub D. (2019): Cross entropy-based importance sampling using Gaussian densities revisited. Structural Safety 76: 15-27.

Papaioannou, Geyer S., Straub D. (2019): Improved cross entropy-based importance sampling with a flexible mixture model. Reliability Engineering and System Safety 191: 106564.

Li L., Papaioannou I., Straub D. (2019): Global reliability sensitivity estimation based on failure samples. Structural Safety 81: 101871.


## Version

Last change: 05/2023

