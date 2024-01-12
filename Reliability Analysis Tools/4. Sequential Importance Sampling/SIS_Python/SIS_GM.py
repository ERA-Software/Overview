import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from EMGM import EMGM
from Sim_Sobol_indices import Sim_Sobol_indices
"""
---------------------------------------------------------------------------
Sequential importance sampling with Gaussian mixture
---------------------------------------------------------------------------
Created by:
Sebastian Geyer
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
Version 2022-04
* Inclusion of sensitivity analysis
---------------------------------------------------------------------------
Comments:
* The SIS method in combination with a Gaussian Mixture model can only be
  applied for low-dimensional problems, since its accuracy decreases
  dramatically in high dimensions.
---------------------------------------------------------------------------
Input:
* N                    : Number of samples per level
* p                    : N/number of chains per level
* g_fun                : limit state function
* distr                : Nataf distribution object or
                         marginal distribution object of the input variables
* k_init               : initial number of Gaussians in the mixture model
* burn                 : burn-in period
* tarCoV               : target coefficient of variation of the weights
* sensitivity_analysis : implementation of sensitivity analysis: 1 - perform, 0 - not perform
* samples_return       : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr       : probability of failure
* l        : total number of levels
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* k_fin    : final number of Gaussians in the mixture
* S_F1     : vector of first order Sobol' indices
---------------------------------------------------------------------------
Based on:
1. "Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
---------------------------------------------------------------------------
"""

def SIS_GM(N, p, g_fun, distr, k_init, burn, tarCoV, sensitivity_analysis, samples_return):
    if (N*p != np.fix(N*p)) or (1/p != np.fix(1/p)):
        raise RuntimeError('N*p and 1/p must be positive integers. Adjust N and p accordingly')

    # initial check if there exists a Nataf object
    if isinstance(distr, ERANataf):   # use Nataf transform (dependence)
        dim = len(distr.Marginals)     # number of random variables (dimension)
        u2x = lambda u: distr.U2X(u)   # from u to x

    elif isinstance(distr[0], ERADist):   # use distribution information for the transformation (independence)
        # Here we are assuming that all the parameters have the same distribution !!!
        # Adjust accordingly otherwise or create an ERANataf object
        dim = len(distr)                    # number of random variables (dimension)
        u2x = lambda u: distr[0].icdf(sp.stats.norm.cdf(u))   # from u to x
    else:
        raise RuntimeError('Incorrect distribution. Please create an ERADist/Nataf object!')

    # LSF in standard space
    g = lambda u: g_fun(np.array(u2x(u),ndmin=2))

    # Initialization of variables and storage
    max_it   = 100              # estimated number of iterations
    m        = 0                # counter for number of levels
    samplesU = []               # space for samples in U-space

    # Properties of SIS
    nsamlev  = N                    # number of samples
    nchain   = int(nsamlev*p)       # number Markov chains
    lenchain = int(nsamlev/nchain)  # number of samples per Markov chain

    # initialize samples
    accrate = np.zeros([max_it])          # space for acceptance rate
    Sk      = np.ones([max_it])           # space for expected weights
    sigmak  = np.zeros([max_it])          # space for sigmak

    # Step 1
    # Perform the first Monte Carlo simulation
    uk = sp.stats.norm.rvs(size=(nsamlev,dim))    # initial samples
    gk = g(uk)                                  # evaluations of g

    # save samples
    samplesU.append(uk)

    # set initial subset and failure level
    gmu = np.mean(gk)

    # Iteration
    for m in range(max_it):
        # Step 2 and 3
        # compute sigma and weights
        if m == 0:
            func        = lambda x: abs(np.std(sp.stats.norm.cdf(-gk/x))/ \
                                        np.mean(sp.stats.norm.cdf(-gk/x))-tarCoV)
            sigma2      = sp.optimize.fminbound(func, 0, 10.0*gmu)
            sigmak[m+1] = sigma2
            wk          = sp.stats.norm.cdf(-gk/sigmak[m+1])
        else:
            func        = lambda x: abs(np.std(sp.stats.norm.cdf(-gk/x)/sp.stats.norm.cdf(-gk/sigmak[m]))/ \
                                        np.mean(sp.stats.norm.cdf(-gk/x)/sp.stats.norm.cdf(-gk/sigmak[m]))-tarCoV)
            sigma2      = sp.optimize.fminbound(func, 0, sigmak[m])
            sigmak[m+1] = sigma2
            wk          = sp.stats.norm.cdf(-gk/sigmak[m+1])/sp.stats.norm.cdf(-gk/sigmak[m])

        # Step 4
        # compute estimate of expected w
        Sk[m] = np.mean(wk)
        # Exit algorithm if no convergence is achieved
        if Sk[m] == 0:
            break
        wnork = wk/Sk[m]/nsamlev     # compute normalized weights

        # fit Gaussian Mixture
        [mu, si, ww] = EMGM(uk.T,wnork.T,k_init)

        # Step 5: resample
        # seeds for chains
        ind = np.random.choice(range(nsamlev),nchain,True,wnork)
        gk0 = gk[ind]
        uk0 = uk[ind,:]

        # Step 6: perform MH
        count  = 0
        alphak = np.zeros([nchain])        # initialize chain acceptance rate
        gk     = np.zeros([nsamlev])       # delete previous samples
        uk     = np.zeros([nsamlev,dim])   # delete previous samples
        for k in range(nchain):
          # set seed for chain
          u0 = uk0[k,:]
          g0 = gk0[k]

          for i in range(lenchain+burn):
              if i == burn:
                  count = count-burn

              # get candidate sample from conditional normal distribution
              indw  = np.random.choice(np.arange(len(ww)), 1, True, ww)[0]
              ucand = sp.stats.multivariate_normal.rvs(mean=mu[:,indw], cov=si[:,:,indw])

              # Evaluate limit-state function
              gcand = g(ucand)

              # compute acceptance probability
              pdfn = 0
              pdfd = 0
              for ii in range(len(ww)):
                  pdfn = pdfn+ww[ii]*sp.stats.multivariate_normal.pdf(u0.T,mu[:,ii],si[:,:,ii])
                  pdfd = pdfd+ww[ii]*sp.stats.multivariate_normal.pdf(ucand.T,mu[:,ii],si[:,:,ii])

              alpha     = min(1,sp.stats.norm.cdf(-gcand/sigmak[m+1])*np.prod(sp.stats.norm.pdf(ucand))*pdfn/ \
                                sp.stats.norm.cdf(-g0/sigmak[m+1])/np.prod(sp.stats.norm.pdf(u0))/pdfd)
              alphak[k] = alphak[k] + alpha/(lenchain+burn)

              # check if sample is accepted
              uhelp = sp.stats.uniform.rvs()
              if uhelp <= alpha:
                  uk[count,:] = ucand
                  gk[count]   = gcand
                  u0 = ucand
                  g0 = gcand
              else:
                  uk[count,:] = u0
                  gk[count]   = g0

              count = count+1

        uk = uk[:nsamlev,:]
        gk = gk[:nsamlev]

        # samples return - all / all by default
        if samples_return not in [0, 1]:
            samplesU.append(uk)

        # compute mean acceptance rate of all chains in level m
        accrate[m] = np.mean(alphak)

        if sigmak[m+1] == 0:
            COV_Sl = np.nan
        else:
            COV_Sl = np.std( (gk < 0)/sp.stats.norm.cdf(-gk/sigmak[m+1]))/ \
                     np.mean((gk < 0)/sp.stats.norm.cdf(-gk/sigmak[m+1]))

        print('\nCOV_Sl =', COV_Sl)
        print('\t*MH-GM accrate =', accrate[m])
        if COV_Sl < tarCoV:
            # samples return - last
            if samples_return == 1 or (samples_return == 0 and sensitivity_analysis == 1):
                samplesU[0] = uk
            break

    # Samples return - all by default message
    if samples_return not in [0, 1, 2]:
        print("\n-Invalid input for samples return, all samples are returned by default")

    # needed steps
    k_fin = len(ww)
    l     = m + 2

    # Calculation of the Probability of failure
    # accfin = accrate[m]
    const = np.prod(Sk)
    #tmp1  = (gk < 0)
    #tmp2  = -gk/sigmak[m+1]
    #tmp3  = sp.stats.norm.cdf(tmp2)
    #tmp4  = tmp1/tmp3
    #Pr    = np.mean(tmp4)*const
    I_final = gk <= 0
    W_final = 1/sp.stats.norm.cdf(-gk/sigmak[m+1])
    Pr    = const * np.mean(I_final * W_final)

    # transform the samples to the physical/original space
    samplesX = list()
    if samples_return != 0 or (samples_return == 0 and sensitivity_analysis == 1):
        for i in range(len(samplesU)):
            samplesX.append( u2x(samplesU[i][:,:]) )

    ## sensitivity analysis
    if  sensitivity_analysis == 1:
        # resample 10000 failure samples with final weights W
        weight_id = np.random.choice(list(np.nonzero(I_final))[0], 10000, list(W_final[I_final]))
        f_s = samplesX[-1][weight_id, :]

        [S_F1, exitflag, errormsg] = Sim_Sobol_indices(f_s, Pr, distr)
        if exitflag == 1:
            print("\n-First order indices: ")
            print(S_F1)
        else:
            print('\n-Sensitivity analysis could not be performed, because: ')
            print(errormsg)
    else:
        S_F1 = []

    if samples_return == 0:
        samplesU = list()  # empty return samples U
        samplesX = list()  # and X

    # Convergence is not achieved message
    if m == max_it:
        print("\n-Exit with no convergence at max iterations \n")
        
    return Pr, l, samplesU, samplesX, k_fin, S_F1
