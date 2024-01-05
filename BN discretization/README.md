# Bayesian network discretization of reliability problems

A MATLAB based software prototype performing efficient static discretization for structural reliability problems, such that they can be updated in a discrete Bayesian network framework. The user defines his/her reliability problem through a limit state function and a statistical model of the corresponding basic random variables. Furthermore a couple of parameters corresponding to the discretization and the BN structure have to be provided.

The software:
* Comes up with an efficient discretization scheme, applying the heuristics developed in (Zwirglmaier & Straub, 2015)
* Establishes the CPTs of the reliability Bayesian network.
* Outputs the BN model in the form of a xdsl-file, which can be read and modified, e.g., using the (free) BN software [GeNIe](https://download.bayesfusion.com/files.html?category=Academia#GeNIe)



## Requirements

Matlab, incl. Statistical toolbox and Optimization toolbox

The Bayesian network is outputed in the format of the Genie BN software, which can be downloaded for free here: [https://download.bayesfusion.com/files.html?](https://download.bayesfusion.com/files.html?)


## Documentation & background

Zwirglmaier, K., & Straub, D. (2016). A discretization procedure for rare events in Bayesian networks. Reliability Engineering & System Safety, 153, 96-109.


## Version

Last change: 03/2021


