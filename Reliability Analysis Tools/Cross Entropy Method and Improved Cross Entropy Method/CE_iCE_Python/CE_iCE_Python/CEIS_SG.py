import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from Sim_Sobol_indices import Sim_Sobol_indices
np.seterr(all='ignore')

"""
---------------------------------------------------------------------------
Cross entropy-based importance sampling with Single Gaussian distribution
---------------------------------------------------------------------------
Created by:
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Matthias Willer
Peter Kaplan
Luca Sardi
Daniel Koutas

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2022-04:
* inlusion of sensitivity analysis
---------------------------------------------------------------------------
Comments:
* The LSF must be coded to accept the full set of samples and no one by one
  (see line 99)
* The CE method in combination with a Gaussian Mixture model can only be
  applied for low-dimensional problems, since its accuracy decreases
  dramatically in high dimensions.
* General convergence issues can be observed with linear LSFs.
---------------------------------------------------------------------------
Input:
* N                     : number of samples per level
* p                     : quantile value to select samples for parameter update
* g_fun                 : limit state function
* distr                 : Nataf distribution object or
                          marginal distribution object of the input variables
* sensitivity_analysis  : implementation of sensitivity analysis: 1 - perform, 0 - not perform
* samples_return        : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr        : probability of failure
* lv        : total number of levels
* N_tot     : total number of samples
* gamma_hat : intermediate levels
* samplesU  : object with the samples in the standard normal space
* samplesX  : object with the samples in the original space
* k_fin     : final number of Gaussians in the mixture
* S_F1      : vector of first order Sobol' indices
---------------------------------------------------------------------------
Based on:
1."Cross entropy-based importance sampling using Gaussian densities revisited"
   Geyer et al.
   To appear in Structural Safety
2."A new flexible mixture model for cross entropy based importance sampling".
   Papaioannou et al. (2018)
   In preparation.
---------------------------------------------------------------------------
"""


def CEIS_SG(N, p, g_fun, distr, sensitivity_analysis, samples_return):
    if (N * p != np.fix(N * p)) or (1 / p != np.fix(1 / p)):
        raise RuntimeError(
            "N*p and 1/p must be positive integers. Adjust N and p accordingly"
        )

    # initial check if there exists a Nataf object
    if isinstance(distr, ERANataf):  # use Nataf transform (dependence)
        dim = len(distr.Marginals)  # number of random variables (dimension)
        u2x = lambda u: distr.U2X(u)  # from u to x

    elif isinstance(
        distr[0], ERADist
    ):  # use distribution information for the transformation (independence)
        # Here we are assuming that all the parameters have the same distribution !!!
        # Adjust accordingly otherwise
        dim = len(distr)  # number of random variables (dimension)
        u2x = lambda u: distr[0].icdf(sp.stats.norm.cdf(u))  # from u to x
    else:
        raise RuntimeError(
            "Incorrect distribution. Please create an ERADist/Nataf object!"
        )

    # LSF in standard space
    G_LSF = lambda u: g_fun(u2x(u))

    # Initialization of variables and storage
    j = 0  # initial level
    max_it = 50  # estimated number of iterations
    N_tot = 0  # total number of samples

    # Definition of parameters of the random variables (uncorrelated standard normal)
    mu_init = np.zeros(dim)
    Si_init = np.eye(dim)
    gamma_hat = np.zeros((max_it + 1))
    samplesU = list()

    # CE Procedure
    # Initializing parameters
    gamma_hat[j] = 1
    mu_hat = mu_init
    Si_hat = Si_init

    # Iteration
    for j in range(max_it):
        # Generate samples and save them
        X = sp.stats.multivariate_normal.rvs(mean=mu_hat, cov=Si_hat, size=N).reshape(
            -1, dim)

        # Count generated samples
        N_tot += N

        # Evaluation of the limit state function
        geval = G_LSF(X)

        # Calculating h for the likelihood ratio
        h = sp.stats.multivariate_normal.pdf(X, mu_hat, Si_hat)

        # Samples return - all / all by default
        if samples_return not in [0, 1]:
            samplesU.append(X)

        # check convergence
        if gamma_hat[j] == 0:
            # Samples return - last
            if samples_return == 1 or (samples_return == 0 and sensitivity_analysis == 1):
                samplesU.append(X)
            break

        # obtaining estimator gamma
        gamma_hat[j + 1] = np.maximum(0, np.nanpercentile(geval, p * 100))
        print("\nIntermediate failure threshold: ", gamma_hat[j + 1])

        # Indicator function
        I = geval <= gamma_hat[j + 1]

        # Likelihood ratio
        W = (
            sp.stats.multivariate_normal.pdf(X, mean=np.zeros((dim)), cov=np.eye((dim)))
            / h
        )

        # Parameter update: Closed-form update
        prod = np.matmul(W[I], X[I, :])
        summ = np.sum(W[I])
        mu_hat = (prod) / summ
        Xtmp = X[I, :] - mu_hat
        Xo = (Xtmp) * np.tile(np.sqrt(W[I]), (dim, 1)).T
        Si_hat = np.matmul(Xo.T, Xo) / np.sum(W[I]) + 1e-6 * np.eye(dim)

    # Samples return - all by default message
    if samples_return not in [0, 1, 2]:
        print("\n-Invalid input for samples return, all samples are returned by default")

    # needed steps
    lv = j
    gamma_hat = gamma_hat[: lv + 1]

    # Calculation of the Probability of failure
    W_final = (
        sp.stats.multivariate_normal.pdf(X, mean=np.zeros(dim), cov=np.eye((dim))) / h
    )
    I_final = geval <= 0
    Pr = 1 / N * sum(I_final * W_final)
    k_fin = 1

    # transform the samples to the physical/original space
    samplesX = list()
    if samples_return != 0 or (samples_return == 0 and sensitivity_analysis == 1):
        for i in range(len(samplesU)):
            samplesX.append(u2x(samplesU[i][:, :]))

    # %% sensitivity analysis
    if sensitivity_analysis == 1:
        # resample 10000 failure samples with final weights W
        weight_id = np.random.choice(list(np.nonzero(I_final))[0],10000,list(W_final[I_final]))
        f_s = samplesX[-1][weight_id,:]
        S_F1 = []

        if len(f_s) == 0:
            print("\n-Sensitivity analysis could not be performed, because no failure samples are available")
            S_F1 = []
        else:
            S_F1, exitflag, errormsg = Sim_Sobol_indices(f_s, Pr, distr)
            if exitflag == 1:
                print("\n-First order indices: \n", S_F1)
            else:
                print('\n-Sensitivity analysis could not be performed, because: \n', errormsg)
        if samples_return == 0:
            samplesU = list()  # empty return samples U
            samplesX = list()  # and X
    else:
        S_F1 = []

    # Convergence is not achieved message
    if j == max_it:
        print("\n-Exit with no convergence at max iterations \n")
        
    return Pr, lv, N_tot, gamma_hat, samplesU, samplesX, k_fin, S_F1
