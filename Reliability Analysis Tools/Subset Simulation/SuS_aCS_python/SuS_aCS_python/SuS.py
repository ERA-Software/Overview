import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from aCS import aCS
from corr_factor import corr_factor
from Sim_Sobol_indices import Sim_Sobol_indices
"""
---------------------------------------------------------------------------
Subset Simulation function (standard Gaussian space)
---------------------------------------------------------------------------
Created by:
Created by:
Matthias Willer
Felipe Uribe
Luca Sardi
Daniel Koutas

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Current version 2022-04
* Inclusion of sensitivity analysis
---------------------------------------------------------------------------
Comments:
*Express the failure probability as a product of larger conditional failure
 probabilities by introducing intermediate failure events.
*Use MCMC based on the modified Metropolis-Hastings algorithm for the
 estimation of conditional probabilities.
*p0, the prob of each subset, is chosen 'adaptively' to be in [0.1,0.3]
---------------------------------------------------------------------------
Input:
* N                    : Number of samples per level
* p0                   : Conditional probability of each subset
* g_fun                : limit state function
* distr                : Nataf distribution object or
                         marginal distribution object of the input variables
* sensitivity_analysis : implementation of sensitivity analysis: 1 - perform, 0 - not perform
* samples_return       : return of samples: 0 - none, 1 - final sample, 2 - all samples          
---------------------------------------------------------------------------
Output:
* Pf_SuS    : failure probability estimate of subset simulation
* delta_SuS : coefficient of variation estimate of subset simulation
* b         : intermediate failure levels
* Pf        : intermediate failure probabilities
* b_line    : limit state function values
* Pf_line   : failure probabilities corresponding to b_line
* samplesU  : samples in the Gaussian standard space for each level
* samplesX  : samples in the physical/original space for each level
* S_F1      : vector of first order Sobol' indices
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SubSim"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
2. "MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
---------------------------------------------------------------------------
"""

def SuS(N, p0, g_fun, distr, sensitivity_analysis, samples_return):
    if (N*p0 != np.fix(N*p0)) or (1/p0 != np.fix(1/p0)):
        raise RuntimeError('N*p0 and 1/p0 must be positive integers. Adjust N and p0 accordingly')

    # %% initial check if there exists a Nataf object
    if isinstance(distr, ERANataf):   # use Nataf transform (dependence)
        n   = len(distr.Marginals)     # number of random variables (dimension)
        u2x = lambda u: distr.U2X(u)   # from u to x

    elif isinstance(distr[0], ERADist):   # use distribution information for the transformation (independence)
        # Here we are assuming that all the parameters have the same distribution !!!
        # Adjust accordingly otherwise
        n   = len(distr)                    # number of random variables (dimension)
        u2x = lambda u: distr[0].icdf(sp.stats.norm.cdf(u))   # from u to x
    else:
        raise RuntimeError('Incorrect distribution. Please create an ERADist/Nataf object!')

    # %% LSF in standard space
    G_LSF = lambda u: g_fun(np.array(u2x(u),ndmin=2))

    # %% Initialization of variables and storage
    j        = 0                # initial conditional level
    Nc       = int(N*p0)        # number of markov chains
    Ns       = int(1/p0)        # number of samples simulated from each Markov chain
    lam      = 0.6              # recommended initial value for lambda
    max_it   = 50               # estimated number of iterations
    samplesU = {'seeds': list(),
                'total': list()}
    #
    geval = np.zeros(N)            # space for the LSF evaluations
    gsort = np.zeros([max_it,N])   # space for the sorted LSF evaluations
    delta = np.zeros(max_it)       # space for the coefficient of variation
    nF    = np.zeros(max_it)       # space for the number of failure point per level
    prob  = np.zeros(max_it)       # space for the failure probability at each level
    b     = np.zeros(max_it)       # space for the intermediate levels

    # %% SuS procedure
    # initial MCS stage
    print('Evaluating performance function:\t', end='')
    u_j = sp.stats.norm.rvs(size=(N,n))     # samples in the standard space
    for i in range(N):
        geval[i] = G_LSF(u_j[i,:])
    print('OK!')

    # SuS stage
    while True:
        # sort values in ascending total
        idx        = np.argsort(geval)
        gsort[j,:] = geval[idx]

        # total the samples according to idx
        u_j_sort = u_j[idx,:]

        # intermediate level
        b[j] = np.percentile(geval, p0*100)

        # number of failure points in the next level
        nF[j] = sum(geval <= max(b[j],0))

        # assign conditional probability to the level
        if b[j] <= 0:
            b[j]    = 0
            prob[j] = nF[j]/N
        else:
            prob[j] = p0
        print('\n-Threshold intermediate level ', j, ' = ', b[j])

        # compute coefficient of variation
        if j == 0:
            delta[j] = np.sqrt(((1-p0)/(N*p0)))   # cov for p(1): MCS (Ref. 2 Eq. 8)
        else:
            I_Fj     = np.reshape(geval <= b[j],(Ns,Nc))       # indicator function for the failure samples
            p_j      = (1/N)*np.sum(I_Fj[:])                   # ~=p0, sample conditional probability
            gamma    = corr_factor(I_Fj,p_j,Ns,Nc)             # corr factor (Ref. 2 Eq. 10)
            delta[j] = np.sqrt( ((1-p_j)/(N*p_j))*(1+gamma) )  # coeff of variation(Ref. 2 Eq. 9)

        # select seeds
        ord_seeds = u_j_sort[:np.int(nF[j]),:]  # ordered level seeds

        # randomize the totaling of the samples (to avoid bias)
        idx_rnd   = np.random.permutation(np.int(nF[j]))
        rnd_seeds = ord_seeds[idx_rnd,:]     # non-ordered seeds

        # Samples return - all / all by default
        if samples_return not in [0, 1]:
            samplesU['total'].append(u_j_sort)      # store the ordered samples
            samplesU['seeds'].append(ord_seeds)     # store the ordered level seeds

        # sampling process using adaptive conditional sampling
        [u_j, geval, lam, sigma, accrate] = aCS(N, lam, b[j], rnd_seeds, G_LSF)
        print('\t*aCS lambda =', lam, '\t*aCS sigma =', sigma[0], '\t*aCS accrate =', accrate)

        # next level
        j = j+1

        if b[j-1] <= 0 or j == max_it:
            break

    m = j

    # Final failure samples (non-ordered)
    if samples_return != 0 or (samples_return == 0 and sensitivity_analysis == 1):
        samplesU['total'].append(u_j)

    # Samples return - all by default message
    if samples_return not in [0, 1, 2]:
        print("\n-Invalid input for samples return, all samples are returned by default")

    # delete unnecessary data
    if m < max_it:
        gsort = gsort[:m,:]
        prob  = prob[:m]
        b     = b[:m]
        delta = delta[:m]

    # %% probability of failure
    # failure probability estimate
    Pf_SuS = np.prod(prob)   # or p0^(m-1)*(Nf(m)/N)

    # coefficient of variation estimate
    delta_SuS = np.sqrt(np.sum(delta**2))   # (Ref. 2 Eq. 12)

    # %% Pf evolution
    Pf           = np.zeros(m)
    Pf_line      = np.zeros((m,Nc))
    b_line       = np.zeros((m,Nc))
    Pf[0]        = p0
    Pf_line[0,:] = np.linspace(p0,1,Nc)
    b_line[0,:]  = np.percentile(gsort[0,:],Pf_line[0,:]*100)
    for i in range(1,m):
        Pf[i]        = Pf[i-1]*p0
        Pf_line[i,:] = Pf_line[i-1,:]*p0
        b_line[i,:]  = np.percentile(gsort[i,:],Pf_line[0,:]*100)

    Pf_line = np.sort(Pf_line.reshape(-1))
    b_line  = np.sort(b_line.reshape(-1))

    # %% transform the samples to the physical/original space
    samplesX = list()
    if samples_return != 0 or (samples_return == 0 and sensitivity_analysis == 1):
        for i in range(len(samplesU['total'])):
            samplesX.append( u2x(samplesU['total'][i][:,:]) )

    if sensitivity_analysis == 1:
        # resample 10000 failure samples
        I_final = geval <= 0
        id = np.random.choice(list(np.nonzero(I_final))[0], 10000)
        f_s = samplesX[-1][id, :]

        S_F1, exitflag, errormsg = Sim_Sobol_indices(f_s, Pf_SuS, distr)

        if exitflag == 1:
            print("\n-First order indices: \n", S_F1)
        else:
            print('\n-Sensitivity analysis could not be performed, because: \n', errormsg)
        if samples_return == 0:
            samplesU['total'] = list()  # empty return samples U
            samplesX = list()           # and X
    else:
        S_F1 = []

    # Convergence is not achieved message
    if j - 1 == max_it:
        print("\n-Exit with no convergence at max iterations \n")

    return Pf_SuS, delta_SuS, b, Pf, b_line, Pf_line, samplesU, samplesX, S_F1
# %%END
