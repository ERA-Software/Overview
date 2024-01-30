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
    exit_msg = ""
    #  Validate the required inputs
    if not ((isinstance(pf,float) or isinstance(pf,int)) and pf>0.0 and pf<=1.0):
        exit_msg += "pf must be a scalar and in range [0,1]! "
    if not (isinstance(distr,ERANataf) or isinstance(distr[0],ERADist)):
        exit_msg += "distribution object not ERADist or ERANataf instance! "
    if not (isinstance(f_s_iid,list) or isinstance(f_s_iid,np.ndarray)):
        exit_msg += "failure samples not provided as list or numpy array! "

    # Transform the f_s_iid values if this is given as a list
    if isinstance(f_s_iid,list):
        f_s_iid:np.ndarray = np.array(f_s_iid)
    
    # check if sample array is empty
    if not (f_s_iid.size != 0):
        exit_msg += "failure samples list/array f_s_iid is empty! Check e.g. if samples_return > 0. "

    if not (isinstance(comp_EVPPI,bool) and isinstance(comp_Sobol,bool)):
        exit_msg += "comp_Sobol and comp_EVPPI have to be boolean! "

    # Validate the optional inputs
    if not ((isinstance(c_R,float) or isinstance(c_R,int)) and 
            (isinstance(c_F,float) or isinstance(c_F,int))):
        exit_msg += "c_R and c_F have to be of type float or int! "
    if not (c_R > 0 and c_F > 0):
        exit_msg += "c_R and c_F have to be larger than 0! "
    if not (normalization in ["crude", "normalized", "relative"]):
        exit_msg += "normalization has to be 'crude', 'normalized', or 'relative! "
    
    if exit_msg != "":
        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Sensitivity computation aborted due to: \n", exit_msg)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return [], []

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
 
 