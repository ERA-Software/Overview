import numpy as np
import scipy as sp
from ERADist import ERADist
from Sim_Sobol_indices import w_opt_finder, kde


def Sim_EVPPI(f_s_iid:np.ndarray, X_marg:ERADist, pf:float, c_R:float= 1e05, 
              c_F:float=1e08, normalization:str="normalized", w_opt:np.ndarray=[],
              integration_points:int=2000):
    '''
    EVPPI CALCULATION FROM Samples
    ---------------------------------------------------------------------------
    Created by:
    Ivan Olarte-Rodriguez
    Engineering Risk Analysis Group   
    Technische Universitat Munchen
    www.bgu.tum.de/era
    ---------------------------------------------------------------------------
    First version: 2023-06
    ---------------------------------------------------------------------------
    Changelog 
    
    ---------------------------------------------------------------------------

    Inputs:
    - X_marg            : list of d marginal input PDFs (ERADist object)
    - pf                : unconditional system failure probability
    - f_s_iid           : failure samples containing n points (d x n matrix)
    - c_R               : cost of repair
    - c_F               : cost of failure
    - normalization     : string with the output mode of the function. The
                          normalization can take three modes: 'crude',
                          'normalized' and 'relative'. This input is treated
                          as optional within this function as the default
                          mode is "normalized".
    Optional:
    - w_opt             : optimal bandwidths of the sampled distributions
    - integration_points: number of integration points to define the integral 
                          function to compute the EVPPI.

    Outputs:
	- S_EVPPI           : Results of the calculation of EVPPI
	- exitflag          : Integer with indication of possible error
	- errormsg          : String with error message (in case thereof)
    '''

    #  Validate the required inputs
    assert (isinstance(pf,float) and pf>0.0 and pf <=1.0)
    assert (isinstance(f_s_iid,list) or isinstance(f_s_iid,np.ndarray))

    # Transform the f_s_iid values if this is given as a list
    if isinstance(f_s_iid,list):
        f_s_iid:np.ndarray = np.array(f_s_iid)
    
    # Ensure all the objects contained in X_marg are of the class 'ERADist'
    for ii,i_marg in enumerate(X_marg):
        try:
            assert(isinstance(i_marg,ERADist))
        except AssertionError:
            raise("At the position {}, the object is not of the class 'ERADist'".format(ii))

    # Validate the optional inputs
    assert((isinstance(c_R,float) or isinstance(c_R,int)) and c_R > 0)
    assert((isinstance(c_F,float) or isinstance(c_F,int)) and c_F > 0)

    
    expectedEVPPIOutputType:tuple = ("crude", "normalized", "relative")

    assert(normalization in expectedEVPPIOutputType)

    assert(isinstance(w_opt,list) or isinstance(w_opt,np.ndarray))

    

    """
    PRECOMPUTATIONS - PART 2: 
    """
    
    # Get number of samples
    N:int = f_s_iid.shape[0]
    # Dimensionality (d)
    d:int = f_s_iid.shape[1]

    # Change the variable type of w_opt if the input is a list
    if isinstance(w_opt,list) and len(w_opt)>=1:
        w_opt:np.ndarray = np.array(w_opt)
        

    # Generate Empty arrays to store objects (set memory in advance)
    crude_EVPPI:np.ndarray = np.zeros(d,)
    S_EVPPI:np.ndarray = np.empty_like(crude_EVPPI)

    # Preset the error message and exit flag
    exitflag:int = 1
    errormsg:str = ""

    # Compute the threshold of the boundaries of the cost of repair vs the
    # cost of failure
    PF_thres:float = c_R/c_F

    # if the bandwidth array is empty, compute it
    if len(w_opt)<1:

        # Set a zeros array to store the bandwidths
        w_opt:np.ndarray = np.zeros((d,))

        # remove multiple and Inf failure samples for optimization
        f_s_iid = f_s_iid[~(~np.isfinite(f_s_iid)).any(axis=1), :]
        _, idcs_x, _, counts = np.unique(f_s_iid, axis=0, return_index=True, return_inverse=True, return_counts=True )

        # Solver parameters
        Maxiter = 500       # default of fminbnd is 500
        Tolx    = 1e-4      # default of fminbnd is 1e-4
        lb      = 0         # lower boundary for search interval on w

        # heuristic upper boundary for search interval on w; Inf not well handled
        #ub      = [int(round(k)) for k in 5*abs(samplesX.max(axis=0)-samplesX.min(axis=0))]
        ub      = [k for k in 5*abs(f_s_iid.max(axis=0)-f_s_iid.min(axis=0))]
        #ub = [int(round(k)) for k in ub]

        # Silverman's rule of thumb for starting points for band width estimation
        # w0_vec = 1.06*np.std(samplesX, axis=0)*N**(-1/5)
        # heuristic starting points (global minimum is often closer to 0)
        w0_vec = 1/10*abs(f_s_iid.max(axis=0)-f_s_iid.min(axis=0))

        for u in range(d):
            w_opt_handle = lambda w: w_opt_finder(w, N, f_s_iid[:,u], idcs_x, counts)

            optim_opts:dict = {'maxiter':Maxiter,'disp':False}
            opt_res:sp.optimize.OptimizeResult = sp.optimize.minimize_scalar(w_opt_handle,
                                                                             bounds=(lb, ub[u]),
                                                                             options=optim_opts,
                                                                             method='bounded')
            
            w_opt[u] = opt_res.x
            ex_flag = opt_res.status
            # if optimal w not found, try fmincon, success if ex_flag == 0
            if ex_flag != 0:
                print('Fminbnd was not succesful, now trying scipy.optimize.minimize\n')
                res = sp.optimize.minimize(w_opt_handle, w0_vec[u], bounds=[(lb, ub[u])], options={'gtol': Tolx})
                w_opt[u] = res.x
                ex_flag = res.success

                # exitflag handling, success if ex_flag == 1
                if ex_flag:
                    exitflag = 1
                    errormsg = ""
                else:
                    exitflag = 0
                    S_EVPPI = np.array([])
                    errormsg = res.message
                    return S_EVPPI, exitflag, errormsg
        
    print("\n-Optimal bandwidths:")
    print(w_opt)


    #Print values of Cost of Replacement and Cost of Failure
    if c_R > c_F:
        exitflag = 0
        errormsg = "The cost of replacement is greater than " + "the cost of failure \n \n"
        return S_EVPPI, exitflag, errormsg
    else:
        # Show the cost of replacement and cost of failure
        print("\n-cost of replacement: {:.2f} ".format( c_R))
        print("-cost of failure: {:.2f}".format( c_F))
    
    '''
    OUTPUT COMPUTATIONS:
    EVPPI: expected value of partial perfect information of each of the d
    input variables of a system regarding a binary repair (repair/do nothing)
    decision defined by the inputs
    '''

    nx:int = integration_points

    # discretize input r.v.:
    for ii in range(d):
        xmin:float = X_marg[ii].mean() - 8* X_marg[ii].std()
        xmax:float = X_marg[ii].mean() + 8* X_marg[ii].std()
        dxi:float = (xmax-xmin)/nx
        xi:np.ndarray  = np.linspace(xmin+dxi/2,xmax-dxi/2,nx,endpoint=True)

        kde_eval = lambda x: np.amin(kde(f_s_iid[:,ii], x, w_opt[ii]))

        '''
        Compute conditional probability of failure, CVPPI and EVPPI:
        '''
        #conditional probability of failure
        #Set the storage size with empty array first
        PF_xi = np.zeros(nx,)
        for jj,xi_i in enumerate(xi):
            # Ignore division by zero
            with np.errstate(divide='ignore'):
                PF_xi[jj] =  np.divide(kde_eval(xi_i),X_marg[ii].pdf(xi_i))*pf

        # Slice the values when Inf to compute CVPPI
        with np.errstate(all='ignore'):
            if pf>PF_thres:
                heaviside_res:np.ndarray = np.heaviside(PF_thres-PF_xi,1)
                CVPPI_xi = np.multiply(heaviside_res,
                                    (c_R-c_F*PF_xi))
            else:
                heaviside_res:np.ndarray = np.heaviside(PF_xi-PF_thres,1)
                CVPPI_xi = np.multiply(heaviside_res,
                                    (c_F*PF_xi-c_R))

        # Replace Inf or NaN with zeros
        CVPPI_xi[np.bitwise_or(np.isnan(CVPPI_xi),np.isinf(CVPPI_xi))] = 0.0
       
        crude_EVPPI[ii] = np.sum(np.multiply(CVPPI_xi,X_marg[ii].pdf(xi)),axis=0)*dxi


    """
    Modify the output depending on the output mode set by user
    """

    if normalization.find("crude")>=0:
        S_EVPPI = crude_EVPPI
    elif normalization.find("normalized")>=0:
        for ii in range(d):
            S_EVPPI[ii] = crude_EVPPI[ii]/np.sum(crude_EVPPI)
    elif normalization.find("relative")>=0:
        # Compute EVPI
        if pf <= PF_thres:
            EVPI = pf*(c_F-c_R)
        else:
            EVPI = c_R*(1-pf)

        S_EVPPI = np.divide(crude_EVPPI,EVPI)


    return S_EVPPI, exitflag, errormsg

