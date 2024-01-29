import numpy as np
from ERANataf import ERANataf
from ERADist import ERADist
from Sim_Sobol_indices import Sim_Sobol_indices
from Sim_EVPPI import Sim_EVPPI


def Sim_Sensitivity(f_s_iid:np.ndarray, pf:float, distr:ERANataf, comp_Sobol:bool=False,
                    comp_EVPPI:bool=False, c_R:float= 1e05, c_F:float=1e08, 
                    normalization:str="normalized"):
    '''
    Compute the Sobol indices and EVPPI from Samples 
    
    ---------------------------------------------------------------------------
    Created by:
    Daniel Koutas
    Ivan Olarte-Rodriguez
    Engineering Risk Analysis Group   
    Technische Universitat Munchen
    www.bgu.tum.de/era
    ---------------------------------------------------------------------------
    First version: 2022-04
    2023-07 (Ivan Olarte-Rodriguez): 
    Splitted the Sensitivity Computations from main functions
    ---------------------------------------------------------------------------
    Based on:
    1. "Global reliability sensitivity estimation based on failure samples"
        Luyi Li, Iason Papaioannou & Daniel Straub.
        Structural Safety 81 (2019) 101871.
    2. "Kernel Estimator and Bandwidth Selection for Density and its
        Derivatives"
        Arsalane Chouaib Guidoum.
        Department of Probabilities and Statistics, University of Science and 
        Technology, Houari Boumediene, Algeria (2015)
    }
    ---------------------------------------------------------------------------
    Comments: 
    * The upper bound of fminbnd is set to a multiple of the maximum distance 
      between the failure samples, because Inf is not handled well.
    * Significantly dominates computation time at higher number of samples
    * User can trigger plot of posterior kernel density estimations as well as
      maximum likelihood cross validation dependent on the bandwidth (optimal
      bandwidth marked as star)
    ---------------------------------------------------------------------------
    Input:
    - Required
    * f_s_iid:          Independent and identically distributed failure samples 
    * pf:               estimated failure probability
    * distr:            ERADist or ERANataf object containing the infos about 
                        the random variables.
    * comp_Sobol:       boolean variable to indicate the computation of the
                        sensitivity metrics based on Sobol Indices.
    * comp_EVPPI:       boolean variable to indicate the computation of EVPPI
                        indices

    - Optional
    * c_R:              Cost of replacement
    * c_F:              Cost of Failure
    * normalization:    Normalization options for EVPPI calculation
    ---------------------------------------------------------------------------
    Output:
    * S_F1:             vector of first order Sobol indices
    * S_EVPPI:          vector of EVPPI measures for each variable
    ---------------------------------------------------------------------------
    '''
    #  Validate the required inputs
    assert (isinstance(pf,float) and pf>0.0 and pf<=1.0)
    assert (isinstance(distr,ERANataf) or isinstance(distr[0],ERADist))
    assert (isinstance(f_s_iid,list) or isinstance(f_s_iid,np.ndarray))

    # Transform the f_s_iid values if this is given as a list
    if isinstance(f_s_iid,list):
        f_s_iid:np.ndarray = np.array(f_s_iid)

    assert (isinstance(comp_EVPPI,bool))
    assert (isinstance(comp_Sobol,bool))

    # Validate the optional inputs

    assert((isinstance(c_R,float) or isinstance(c_R,int)) and c_R > 0)
    assert((isinstance(c_F,float) or isinstance(c_F,int)) and c_F > 0)
    
    expectedEVPPIOutputType:tuple = ("crude","normalized","relative")

    assert(normalization in expectedEVPPIOutputType)

    # Compute the Sensitivity Indices
    S_F1:np.ndarray = np.empty_like(f_s_iid[0,:])
    w_opt:np.ndarray = []

    if comp_Sobol:
        print("\n\nComputing Sobol Sensitivity Indices")
        [S_F1, exitflag, errormsg, w_opt] = Sim_Sobol_indices(f_s_iid, pf, distr)
        if exitflag == 1:
            print("\n-First order indices:")
            print(S_F1)
        else:
            print('\n-Sensitivity analysis could not be performed, because:')
            print(errormsg)

    # Compute the EVPPI based on input parameters
    S_EVPPI:np.ndarray = np.empty_like(f_s_iid[0,:])

    if isinstance(distr,ERANataf):
        marginal_list = distr.Marginals
    else:
        marginal_list = distr

    # Compute only if the outputs are different than the default
    if comp_EVPPI:
        print("\n\nComputing EVPPI Sensitivity Indices")
        [S_EVPPI, exitflag, errormsg] = Sim_EVPPI(f_s_iid, marginal_list,pf, c_R, c_F, 
                                                  normalization, w_opt=w_opt)
            
        if exitflag == 1:
            # Show the computation of EVPPI given the parameters
            print("-EVPPI normalized as: ", normalization)
            print("\n-EVPPI indices:")
            print(S_EVPPI)
        else:
            print('\n-Sensitivity analysis could not be performed, because:')
            print(errormsg)
    
    return S_F1, S_EVPPI
 
 