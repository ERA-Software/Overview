import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist

def FORM_EVPPI(alpha:np.ndarray, beta:float, pf:float, distr:ERANataf, 
               c_R:float, c_F:float, normalization:str) -> list:

    """
    Created by:
    Daniel Koutas
    Max Ehre
    Ivan Olarte-Rodriguez
    Engineering Risk Analysis Group   
    Technische Universitat Munchen
    www.bgu.tum.de/era
    ---------------------------------------------------------------------------
    First version: 2023-06
    ---------------------------------------------------------------------------
    Changelog 

    ---------------------------------------------------------------------------
    Comment:
    * The EVPPI computation uses the FORM indices as input. 
    WARNING: Only valid for independent marginal distributions
    ---------------------------------------------------------------------------
    Input:
    * alpha         : vector with the values of FORM indices
    * beta          : reliability index
    * pf            : probaiblity of failure
    * distr         : ERANataf or ERARosen distribution object
    * c_R           : cost of replacement (scalar)
    * c_F           : cost of Failure
    * normalization : normalization output of EVPPI
    ---------------------------------------------------------------------------
    Output:
    * EVPPI         : Vector with computation of EVPPI per each input variable
    ---------------------------------------------------------------------------
    """

    ## Precomputations
    d:int = len(distr.Marginals) # number of dimensions

    # Array to store the EVPPI values
    crude_EVPPI:np.ndarray = np.zeros((d,))

    # Preset the error message and exit flag
    exitflag:int = 1
    errormsg:str = ""


    # Compute the threshold of the boundaries of the cost of repair vs the
    # cost of failure

    PF_thres:float = c_R/c_F

    if c_R > c_F:
        EVPPI:np.ndarray = np.repeat(np.array(np.NaN),d,axis=0)
        exitflag:int = 0
        errormsg:str = "The cost of replacement is greater than the cost of failure \n \n"
        return np.NaN, exitflag, errormsg
    else:
        # Show the cost of replacement and cost of failure
        print(f"\n-cost of replacement: {c_R}")
        print(f"-cost of failure: {c_F}")

    ## COMPUTATION OF CRUDE EVPPI
    # WARNING! -> The procedure is meant only for independent marginal distributions

    u_i_thres:np.ndarray = np.zeros((d,),dtype=float)

    # Lambda function to compute the threshold
    u_i_thres_fun = lambda u: np.divide(1,u)*(np.add(np.multiply(np.sqrt(1-np.power(u,2)),sp.special.ndtri(PF_thres)),beta))

    # Array to store the s_i
    s_i = np.zeros_like(u_i_thres)

    for ii in range(d):
        # Compute the threshold
        u_i_thres[ii] = u_i_thres_fun(alpha[ii])

        # Compute the s_i
        if (pf-PF_thres)*alpha[ii] !=0:
            s_i[ii] = np.sign((pf-PF_thres)*alpha[ii])
        else:
            s_i[ii] = -1

        # Compute the EVPPI
        # Generate auxiliary mean array for bivariate standard normal cdf
        aux_mean:np.ndarray = np.zeros((2,))
        # Generate auxiliary sigma array for bivariate standard normal cdf
        aux_sigma:np.ndarray = np.array([[1,-np.squeeze(s_i[ii]*alpha[ii])],[-np.squeeze(s_i[ii]*alpha[ii]),1]])
        mvn = sp.stats.multivariate_normal(mean = aux_mean,cov = aux_sigma)
        crude_EVPPI[ii] = np.abs(c_F*mvn.cdf(np.array([-beta,np.squeeze(s_i[ii]*u_i_thres[ii])]))-
                                                                  c_R*sp.stats.norm.cdf(np.squeeze(s_i[ii]*u_i_thres[ii])))

    ## OUTPUT
    # Modify the output depending on the output mode set by user
    if normalization.find("crude")>=0:
        EVPPI:np.ndarray = crude_EVPPI
    elif normalization.find("normalized")>=0:
        EVPPI:np.ndarray = np.zeros_like(crude_EVPPI,dtype=float)
        for ii in range(d):
            EVPPI[ii] = np.divide(crude_EVPPI[ii],np.sum(crude_EVPPI,axis=None))
    elif normalization.find("relative")>=0:
        # Compute EVPI
        if pf <= PF_thres:
            EVPI:float = pf*(c_F-c_R)
        else:
            EVPI:float = c_R*(1-pf)

        EVPPI = np.divide(crude_EVPPI,EVPI)
    
    # Return values
    return EVPPI, exitflag, errormsg

