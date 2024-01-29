import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from EMGM import EMGM
from EMvMFNM import EMvMFNM
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
Ivan Olarte Rodriguez

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Current version 2023-04
* Modification of Sensitivity Analysis Calls
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
* samples_return       : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr       : probability of failure
* l        : total number of levels
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* k_fin    : final number of Gaussians in the mixture
* W_final  : final weights
* f_s_iid  : Independent Identically Distributed samples generated from last step
---------------------------------------------------------------------------
Based on:
1. "Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
---------------------------------------------------------------------------
"""

def SIS_vMFNM(N, p, g_fun, distr, k_init, burn, tarCoV, samples_return):
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
    max_it = 100;    # estimated number of iterations
    m      = 0;      # counter for number of levels

    # initialize output arrays
    sigma_t   = np.zeros((max_it,1)) # squared difference between COV and target COV
    samplesU  = []          # space for the samples in the standard space

    # Properties of SIS
    nsamlev  = N               # number of samples
    nchain   = nsamlev*p       # number Markov chains
    lenchain = nsamlev/nchain  # number of samples per Markov chain

    # initialize samples
    accrate = np.zeros([max_it])          # space for acceptance rate
    Sk      = np.ones([max_it])           # space for expected weights
    sigmak  = np.zeros([max_it])          # space for sigmak

    # Step 1
    # Perform the first Monte Carlo simulation
    uk = sp.stats.norm.rvs(size=(nsamlev,dim)).reshape(-1, dim)    # initial samples
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
            func        = lambda x: abs(np.std(approx_normCDF(-gk/x))/ \
                                        np.mean(approx_normCDF(-gk/x))-tarCoV)
            sigma2      = sp.optimize.fminbound(func, 0, 10.0*gmu)
            sigmak[m+1] = sigma2
            wk          = approx_normCDF(-gk/sigmak[m+1])
        else:
            func        = lambda x: abs(np.std(approx_normCDF(-gk/x)/approx_normCDF(-gk/sigmak[m]))/ \
                                        np.mean(approx_normCDF(-gk/x)/approx_normCDF(-gk/sigmak[m]))-tarCoV)
            sigma2      = sp.optimize.fminbound(func, 0, sigmak[m])
            sigmak[m+1] = sigma2
            wk          = np.divide(approx_normCDF(-gk/sigmak[m+1]),approx_normCDF(-gk/sigmak[m]))

        # Step 4
        # compute estimate of expected w
        Sk[m] = np.mean(wk)
        # Exit algorithm if no convergence is achieved
        if Sk[m] == 0:
            break
        wnork = np.divide(wk,Sk[m])/nsamlev     # compute normalized weights


        # fit Gaussian Mixture
        #[mu, si, ww] = EMGM(uk.T,wnork.T,k_init)

        # fit vMFN Mixture
        [mu, kappa, mshape, omega, ww] = EMvMFNM(np.transpose(uk), wnork, k_init)

        # remove unnecessary components
        if np.min(ww) <= 0.01:
            ind = ww > 0.01
            mu = mu[:,ind]
            kappa  = kappa[ind]
            mshape = mshape[:,ind]
            omega  = omega[:,ind]
            ww     = ww[ind]

        
        # Step 5: resample
        # seeds for chains
        ind = np.random.choice(range(nsamlev),int(nchain),True,wnork)
        gk0 = gk[ind]
        uk0 = uk[ind,:]

        # Step 6: perform MH
        count  = 0
        alphak = np.zeros([int(nchain)])        # initialize chain acceptance rate
        gk     = np.zeros([nsamlev])       # delete previous samples
        uk     = np.zeros([nsamlev,dim])   # delete previous samples
        for k in range(int(nchain)):
          # set seed for chain
          u0 = uk0[k,:]
          g0 = gk0[k]

          for i in range(int(lenchain+burn)):
              if i == burn:
                  count = count-burn

              # get candidate sample from conditional normal distribution
              ucand = vMFNM_sample(mu.T,kappa,omega,mshape,ww,1).reshape(-1, dim)

              # Evaluate limit-state function
              gcand = g(ucand)

              # compute acceptance probability
              W_log_cand = likelihood_ratio_log(np.reshape(ucand,(-1,dim)),mu.T,kappa,omega,mshape,ww)
              W_log_0 = likelihood_ratio_log(np.reshape(u0,(-1,dim)),mu.T,kappa,omega,mshape,ww)
              # (X, mu, kappa, omega, m, alpha)

              alpha     = np.minimum(1,np.exp(W_log_cand - W_log_0)*sp.stats.norm.cdf(-gcand/sigmak[m+1])/(sp.stats.norm.cdf(-g0/sigmak[m+1])));
              alphak[k] = alphak[k] + alpha/[int(lenchain+burn)]

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
            COV_Sl = np.std( (gk < 0)/approx_normCDF(-gk/sigmak[m+1]))/ \
                     np.mean((gk < 0)/approx_normCDF(-gk/sigmak[m+1]))

        print('\nCOV_Sl =', COV_Sl)
        print('\t*MH-GM accrate =', accrate[m])
        if COV_Sl < tarCoV:
            # samples return - last
            if samples_return == 1:
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
    I_final = gk <= 0
    W_final = 1/sp.stats.norm.cdf(-gk/sigmak[m+1])
    Pr    = const * np.mean(I_final * W_final)

    # transform the samples to the physical/original space
    samplesX = list()
    if samples_return != 0:
        for i in range(len(samplesU)):
            samplesX.append( u2x(samplesU[i][:,:]) )
    
    # resample 10000 failure samples with final weights W
    weight_id = np.random.choice(list(np.nonzero(I_final))[0], 10000, list(W_final[I_final]))
    f_s_iid = samplesX[-1][weight_id, :]

    if samples_return == 0:
        samplesU = list()  # empty return samples U
        samplesX = list()  # and X

    # Convergence is not achieved message
    if m == max_it:
        print("\n-Exit with no convergence at max iterations \n")
        
    return Pr, l, samplesU, samplesX, k_fin, W_final, f_s_iid

# ===========================================================================
# =============================AUX FUNCTIONS=================================
# ===========================================================================
# --------------------------------------------------------------------------
# Returns uniformly distributed samples from the surface of an
# n-dimensional hypersphere
# --------------------------------------------------------------------------
# N: # samples
# n: # dimensions
# R: radius of hypersphere
# --------------------------------------------------------------------------
def hs_sample(N, n, R):

    Y = sp.stats.norm.rvs(size=(n, N))  # randn(n,N)
    Y = Y.T
    norm = np.tile(np.sqrt(np.sum(Y ** 2, axis=1)), [1, n])
    X = Y / norm * R  # X = np.matmul(Y/norm,R)

    return X


# ===========================================================================
# --------------------------------------------------------------------------
# Returns samples from the von Mises-Fisher-Nakagami mixture
# --------------------------------------------------------------------------
def vMFNM_sample(mu, kappa, omega, m, alpha, N):

    [k, dim] = np.shape(mu)
    if k == 1:
        # sampling the radius
        #     pd=makedist('Nakagami','mu',m,'omega',omega)
        #     R=pd.random(N,1)
        R = np.sqrt(sp.stats.gamma.rvs(a=m, scale=omega / m, size=[N, 1]))

        # sampling on unit hypersphere
        X_norm = vsamp(mu.T, kappa, N)

    else:
        # Holder for the probabilities adjustment
        if np.sum(alpha)!=1.0:
            alpha = np.divide(alpha,np.sum(alpha))
        # Determine number of samples from each distribution
        z = np.sum(dummyvar(np.random.choice(np.arange(k), N, True, alpha)), axis=0)
        k = len(z)

        # Generation of samples
        R = np.zeros([N, 1])
        R_last = 0
        X_norm = np.zeros([N, dim])
        X_last = 0

        for p in range(k):
            # sampling the radius
            R[R_last : R_last + z[p], :] = np.sqrt(
                sp.stats.gamma.rvs(
                    a=m[:, p], scale=omega[:, p] / m[:, p], size=[z[p], 1]
                )
            )
            R_last = R_last + z[p]

            # sampling on unit hypersphere
            X_norm[X_last : X_last + z[p], :] = vsamp(mu[p, :].T, kappa[p], z[p])
            X_last = X_last + z[p]

            # clear pd

    # Assign sample vector
    X = R * X_norm  # bsxfun(@times,R,X_norm)

    return X


# ===========================================================================
# --------------------------------------------------------------------------
# Returns samples from the von Mises-Fisher distribution
# --------------------------------------------------------------------------
def vsamp(center, kappa, n):

    d = np.size(center, axis=0)  # Dimensionality
    l = kappa  # shorthand
    t1 = np.sqrt(4 * l * l + (d - 1) * (d - 1))
    b = (-2 * l + t1) / (d - 1)
    x0 = (1 - b) / (1 + b)
    X = np.zeros([n, d])
    m = (d - 1) / 2
    c = l * x0 + (d - 1) * np.log(1 - x0 * x0)

    for i in range(n):
        t = -1000
        u = 1
        while t < np.log(u):
            z = sp.stats.beta.rvs(m, m)  # z is a beta rand var
            u = sp.stats.uniform.rvs()  # u is unif rand var
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            t = l * w + (d - 1) * np.log(1 - x0 * w) - c

        v = hs_sample(1, d - 1, 1)
        X[i, : d - 1] = (
            np.sqrt(1 - w * w) * v
        )  # X[i,:d-1] = np.matmul(np.sqrt(1-w*w),v.T)
        X[i, d - 1] = w

    [v, b] = house(center)
    Q = np.eye(d) - b * np.matmul(v, v.T)
    for i in range(n):
        tmpv = np.matmul(Q, X[i, :].T)
        X[i, :] = tmpv.T

    return X


# ===========================================================================
# --------------------------------------------------------------------------
# X,mu,kappa
# Returns the von Mises-Fisher mixture log pdf on the unit hypersphere
# --------------------------------------------------------------------------
def vMF_logpdf(X, mu, kappa):

    d = np.size(X, axis=0)
    n = np.size(X, axis=1)

    if kappa == 0:
        A = np.log(d) + np.log(np.pi ** (d / 2)) - sp.special.gammaln(d / 2 + 1)
        y = -A * np.ones([1, n])
    elif kappa > 0:
        c = (
            (d / 2 - 1) * np.log(kappa)
            - (d / 2) * np.log(2 * np.pi)
            - logbesseli(d / 2 - 1, kappa)
        )
        q = np.matmul((mu * kappa).T, X)  # bsxfun(@times,mu,kappa)'*X
        y = q + c.T  # bsxfun(@plus,q,c')
    else:
        raise ValueError("Kappa must not be negative!")

    return y


# ===========================================================================
# --------------------------------------------------------------------------
# Returns the value of the log-nakagami-pdf
# --------------------------------------------------------------------------
def nakagami_logpdf(X, m, om):

    y = (
        np.log(2)
        + m * (np.log(m) - np.log(om) - X ** 2 / om)
        + np.log(X) * (2 * m - 1)
        - sp.special.gammaln(m)
    )

    return y


# ===========================================================================
# --------------------------------------------------------------------------
# likelihood_ratio_log()
# --------------------------------------------------------------------------
def likelihood_ratio_log(X, mu, kappa, omega, m, alpha):

    k = len(alpha)
    [N, dim] = np.shape(X)
    R = np.sqrt(np.sum(X * X, axis=1)).reshape(-1, 1)
    if k == 1:
        # log pdf of vMF distribution
        logpdf_vMF = vMF_logpdf((X / R).T, mu.T, kappa).T
        # log pdf of Nakagami distribution
        logpdf_N = nakagami_logpdf(R, m, omega)
        # log pdf of weighted combined distribution
        h_log = logpdf_vMF + logpdf_N
    else:
        logpdf_vMF = np.zeros([N, k])
        logpdf_N = np.zeros([N, k])
        h_log = np.zeros([N, k])

        # log pdf of distributions in the mixture
        for p in range(k):
            # log pdf of vMF distribution
            logpdf_vMF[:, p] = vMF_logpdf((X / R).T, mu[p, :].T, kappa[p]).squeeze()
            # log pdf of Nakagami distribution
            logpdf_N[:, p] = nakagami_logpdf(R, m[:, p], omega[:, p]).squeeze()
            # log pdf of weighted combined distribution
            h_log[:, p] = logpdf_vMF[:, p] + logpdf_N[:, p] + np.log(alpha[p])

        # mixture log pdf
        h_log = sp.special.logsumexp(h_log, axis=1)

    # unit hypersphere uniform log pdf
    A = np.log(dim) + np.log(np.pi ** (dim / 2)) - sp.special.gammaln(dim / 2 + 1)
    f_u = -A

    # chi log pdf
    f_chi = (
        np.log(2) * (1 - dim / 2)
        + np.log(R) * (dim - 1)
        - 0.5 * R ** 2
        - sp.special.gammaln(dim / 2)
    )

    # logpdf of the standard distribution (uniform combined with chi distribution)
    f_log = f_u + f_chi
    W_log = f_log - h_log

    return W_log


# ===========================================================================
# --------------------------------------------------------------------------
# HOUSE Returns the householder transf to reduce x to b*e_n
#
# [V,B] = HOUSE(X)  Returns vector v and multiplier b so that
# H = eye(n)-b*v*v' is the householder matrix that will transform
# Hx ==> [0 0 0 ... ||x||], where  is a constant.
# --------------------------------------------------------------------------
def house(x):

    x = x.squeeze()
    n = len(x)
    s = np.matmul(x[: n - 1].T, x[: n - 1])
    v = np.concatenate([x[: n - 1], np.array([1.0])]).squeeze()
    if s == 0:
        b = 0
    else:
        m = np.sqrt(x[n - 1] * x[n - 1] + s)

        if x[n - 1] <= 0:
            v[n - 1] = x[n - 1] - m
        else:
            v[n - 1] = -s / (x[n - 1] + m)

        b = 2 * v[n - 1] * v[n - 1] / (s + v[n - 1] * v[n - 1])
        v = v / v[n - 1]

    v = v.reshape(-1, 1)

    return [v, b]


# ===========================================================================
# --------------------------------------------------------------------------
# log of the Bessel function, extended for large nu and x
# approximation from Eqn 9.7.7 of Abramowitz and Stegun
# http://www.math.sfu.ca/~cbm/aands/page_378.htm
# --------------------------------------------------------------------------
def logbesseli(nu, x):

    if nu == 0:  # special case when nu=0
        logb = np.log(sp.special.iv(nu, x))  # besseli
    else:  # normal case
        # n    = np.size(x, axis=0)
        n = 1  # since x is always scalar here
        frac = x / nu
        square = np.ones(n) + frac ** 2
        root = np.sqrt(square)
        eta = root + np.log(frac) - np.log(np.ones(n) + root)
        logb = -np.log(np.sqrt(2 * np.pi * nu)) + nu * eta - 0.25 * np.log(square)

    return logb

# ===========================================================================
# --------------------------------------------------------------------------
# Translation of the Matlab-function "dummyvar()" to Python
# --------------------------------------------------------------------------
def dummyvar(idx):

    n = np.max(idx) + 1
    d = np.zeros([len(idx), n], int)
    for i in range(len(idx)):
        d[i, idx[i]] = 1

    return d


def approx_normCDF(x):
    # Returns an approximation for the standard normal CDF based on a polynomial fit of degree 9

    erfun = np.zeros(len(x))

    idpos = x > 0
    idneg = x < 0

    t = (1 + 0.5 * abs(x / np.sqrt(2))) ** -1

    tau = t * np.exp(
        -((x / np.sqrt(2)) ** 2)
        - 1.26551223
        + 1.0000236 * t
        + 0.37409196 * (t ** 2)
        + 0.09678418 * (t ** 3)
        - 0.18628806 * (t ** 4)
        + 0.27886807 * (t ** 5)
        - 1.13520398 * (t ** 6)
        + 1.48851587 * (t ** 7)
        - 0.82215223 * (t ** 8)
        + 0.17087277 * (t ** 9)
    )
    erfun[idpos] = 1 - tau[idpos]
    erfun[idneg] = tau[idneg] - 1

    p = 0.5 * (1 + erfun)

    return p
