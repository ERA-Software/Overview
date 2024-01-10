import numpy as np
import scipy.integrate as integrate

def FORM_Sobol_indices(alpha, beta, Pf):
    ## Compute Sobol' and total-effect indices from FORM parameters

    """
    ---------------------------------------------------------------------------
    Created by:
    Daniel Koutas
    Engineering Risk Analysis Group
    Technische Universitat Munchen
    www.bgu.tum.de/era
       ---------------------------------------------------------------------------
    First version: 2021-06
    ---------------------------------------------------------------------------
    Based on:
    1."Variance-based reliability sensitivity analysis and the FORM a-factors"
       Iason Papaioannou & Daniel Straub.
       Reliability Engineering and System Safety 210 (2021) 107496.
    ---------------------------------------------------------------------------
    Comment:
    * The results apply directly to the first order and total-effect indices
      with respect to the original variables X for the case where X consists of
      statistically independent components.
    ---------------------------------------------------------------------------
    Input:
    * alpha : normalized negative gradient vector at design point
    * beta  : reliability index
    * Pf    : probability of failure
    ---------------------------------------------------------------------------
    Output:
    * S_F1      : vector of first order indices
    * S_F1_T    : vector of total-effect indices
    * exitflag  : flag whether method was successful or not
    * errormsg  : error message describing what went wrong
    ---------------------------------------------------------------------------
    """

    S_F1 = []
    S_F1_T = []

    ##Initial check for valid inputs
    if alpha == [] or not np.prod(np.isfinite(alpha)):
        exitflag = 0
        errormsg = 'Invalid function input: alpha either infinite, NaN or empty.'
        return [S_F1, S_F1_T, exitflag, errormsg]

    elif beta == [] or not np.prod(np.isfinite(beta)):
        exitflag = 0
        errormsg = 'Invalid function input: beta either infinite, NaN or empty.';
        return [S_F1, S_F1_T, exitflag, errormsg]

    elif Pf == [] or not np.prod(np.isfinite(Pf)):
        exitflag = 0
        errormsg = 'Invalid function input: Pf either infinite, NaN or empty.';
        return [S_F1, S_F1_T, exitflag, errormsg]

    elif (Pf <0) or (Pf>1):
        exitflag = 0
        errormsg = 'Invalid function input: Pf must be between 0 and 1.'
        return [S_F1, S_F1_T, exitflag, errormsg]

    # function to integrate
    fun = lambda r: 1/(2*np.pi) * 1/np.sqrt(1-r**2) * np.exp(-(beta**2)/(1+r))

    # Compute first order and total-effect indices for each dimension
    for k in range(len(alpha)):
        S_F1.append(1/(Pf*(1-Pf)) * integrate.quad(fun, 0, alpha[k]**2)[0])
        S_F1_T.append(1/(Pf*(1-Pf)) * integrate.quad(fun, 1-alpha[k]**2, 1)[0])

    S_F1 = np.asarray([S_F1]).squeeze()
    S_F1_T = np.asarray([S_F1_T]).squeeze()


    # check if computed indices are valid
    if (not np.prod(np.isreal(S_F1))) or (not np.prod(np.isreal(S_F1_T))):
        errormsg = 'Integration was not succesful, at least one index is complex.'
        exitflag = 0
    elif np.any(S_F1 > 1) or np.any(S_F1_T > 1):
        errormsg = 'Integration was not succesful, at least one index is greater than 1.'
        exitflag = 0
    elif np.any(S_F1 < 0) or np.any(S_F1_T < 0):
        errormsg = 'Integration was not succesful, at least one index is smaller than 0.'
        exitflag = 0
    elif (not np.prod(np.isfinite(S_F1))) or (not np.prod(np.isfinite(S_F1_T))):
        errormsg = 'Integration was not succesful, at least one index is NaN.'
        exitflag = 0
    else:
        errormsg = []
        exitflag = 1


    return [S_F1, S_F1_T, exitflag, errormsg]
