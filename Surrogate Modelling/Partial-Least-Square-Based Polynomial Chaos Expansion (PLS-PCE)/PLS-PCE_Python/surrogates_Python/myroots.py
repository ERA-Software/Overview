import numpy as np
import scipy as sp

def myroots(lam, M):
    # lam: given correlation length
    # M  : the number of roots we want

    # set up function. 
    c = 1/lam # So c>0.
    g = lambda x: (np.tan(x) - (2*c*x) / (x**2-c**2))
    
    # initialise.
    m = 0
    answer = np.empty(M)

    # for all the intervals.
    for i in range(M):
        
        # lower and upper bounds.
        wmin = (i-0.499)*np.pi
        wmax = (i+0.499)*np.pi

        # interval with two solutions, this is the one containing c.
        if ((wmin <= c) and (wmax >= c)):
            # If it is not the first interval, look for a solution towards the
            #left boundary.
            if (wmin > 0):
                answer[m] = sp.optimize.fsolve(g, (c+wmin)/2)
                m=m+1           
            
            # Always look for a solution towards right boundary.
            answer[m] = sp.optimize.fsolve(g, (c+wmax)/2)
            m = m+1      
            
        # other intervals
        elif (wmin > 0):
            answer[m] = sp.optimize.fsolve(g, wmin)
            m = m+1  
    
    answer = answer[0:M]

    return answer

# We exclude the root zero since this results in the zero eigenfunction.