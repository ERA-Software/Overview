import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from FORM_Sobol_indices import FORM_Sobol_indices
from FORM_EVPPI import FORM_EVPPI


def FORM_Sensitivity(pf:float, distr:ERANataf, beta:float, alpha:np.ndarray, comp_Sobol:bool=False, 
                     comp_EVPPI:bool=False, c_R:float= 1e05, c_F:float=1e08, 
                     normalization:str="normalized") -> list:
    
    """
    #### Compute the Sensitivity Indices and EVPPI from Samples 

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
    2.  "Kernel Estimator and Bandwidth Selection for Density and its
        Derivatives"
        Arsalane Chouaib Guidoum.
        Department of Probabilities and Statistics, University of Science and 
        Technology, Houari Boumediene, Algeria (2015)
    ---------------------------------------------------------------------------
    Comments: 
    * Calls to FORM_Sobol are splitted from FORM main functions
    ---------------------------------------------------------------------------
    Inputs:
    - Required
    * pf:           estimated failure probability
    * distr:        ERANataf or ERARosen object containing the infos about 
                    the random variables
    * beta:         reliability index
    * alpha:        vector with the values of FORM indices
    * comp_Sobol : boolean variable to indicate the computation of the
                        sensitivity metrics based on Sobol Indices.
    * comp_EVPPI        : boolean variable to indicate the computation of EVPPI
                        indices

    - Optional
    * c_R           : Cost of replacement
    * c_F           : Cost of Failure
    * normalization : Normalization options for EVPPI calculation
    ---------------------------------------------------------------------------
    Output:
    * S_F1      : vector of first order sensitivity indices
    * S_F1_T    : vector of total-effect indices
    * S_EVPPI   : vector of EVPPI measures for each variable
    ---------------------------------------------------------------------------
    """
    
    exit_msg = ""
    #  Validate the required inputs
    if not (isinstance(pf,float) and pf>0.0 and pf<=1.0):
        exit_msg += "pf not in allowed range [0,1]! "
    if not (isinstance(distr,ERANataf) or isinstance(distr[0],ERADist)):
        exit_msg += "distribution object not ERADist or ERANataf instance! "
    if not ((isinstance(beta,float) or isinstance(beta,int)) and (beta>0)):
        exit_msg += "beta has to be a float/int and bigger than 0! "
    if not (isinstance(alpha,list) or isinstance(alpha,np.ndarray)):
        exit_msg += "alpha not provided as list or numpy array! "

    # Transform the alpha values if this is given as a list
    if isinstance(alpha,list):
        alpha:np.ndarray = np.array(alpha)


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

    # Generate empty variables to store the sensitivity indices
    S_F1:np.ndarray = np.empty_like(alpha)
    S_F1_T:np.ndarray = np.empty_like(alpha)

    # Proceed to compute Sobol Indices should the handle is activated
    if comp_Sobol:
        print("\n\nComputing Sobol Sensitivity Indices")
        [S_F1, S_F1_T, exitflag, errormsg] = FORM_Sobol_indices(alpha, beta, pf)
        
        # Print error messages and flags
        if hasattr(distr, 'Marginals'):
            if not (distr.Rho_X == np.eye(len(distr.Marginals))).all():
                print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING: !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Results of sensitivity analysis do not apply for dependent inputs.")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")

        if exitflag == 1:
            print("\n-First order indices:")
            print(' ', S_F1)
            print("\n-Total-effect indices:")
            print(' ', S_F1_T)
        else:
            print('\n-Sensitivity analysis could not be performed, because:')
            print(errormsg)

    
    # Start the output as an empty set
    S_EVPPI:np.ndarray = np.empty_like(alpha)
    #  Proceed to compute EVPPI should the handle is activated
    if comp_EVPPI:
        print("\n\nComputing EVPPI Sensitivty Indices")
        [S_EVPPI,exitflag, errormsg] = FORM_EVPPI(alpha,beta,pf,distr,c_R,
                                                  c_F,normalization)
        
        # Print error messages and flags
        if hasattr(distr, 'Marginals'):
            if not (distr.Rho_X == np.eye(len(distr.Marginals))).all():
                print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING: !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Results of EVPPI do not apply for dependent inputs.")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")

        if exitflag == 1:
            #Show the computation of EVPPI given the parameters
            print("-EVPPI normalized as: ", normalization)
            print("\n-EVPPI indices:")
            print(S_EVPPI)
        else:
            print('\n-Sensitivity analysis could not be performed, because:')
            print(errormsg)

    # Return the values            
    return S_F1, S_F1_T, S_EVPPI


