# Find the root(s) of a function by fitting a polynomial
"""
---------------------------------------------------------------------------
Created by:
Iason Papaioannou
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Muenchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Current version 2021-05
---------------------------------------------------------------------------
Input:
* g_1D: desired function for which to find the root(s)
* u_0: starting point
* N: number of evaluations of the g_1D function
---------------------------------------------------------------------------
Output:
* u_root: found root (either the found value or empty [])
* exitflag: flag whether a root was found (1) or not (0)
"""

import numpy as np
import math

def find_root_RS(g_1D, u_0, N):

    if np.mod(N, 2) == 1:
        k = (N+1)/2
    else:
        k = N/2

    # selection of points for LSF evaluations
    u = np.zeros(N)
    y = np.zeros(N)

    for i in range(N):
        u[i] = u_0 + (i+1-k)*0.6
        y[i] = g_1D(u[i])

    idx1 = [math.isfinite(v) for v in y]
    u = u[idx1]
    y = y[idx1]

    if len(u) < 3:
        u_root = []
        exitflag = 0
    else:
        p = np.polyfit(u, y, 2)

        # check if any polynomial coefficient is nan or inf -> invalid fit
        if np.sum([not math.isnan(v) for v in p]) + np.sum(np.isfinite(p)) < len(p):
            u_root = []
            exitflag = 0
        else:
            r = np.roots(p)

            # check if there is any root which is not nan or inf
            # -> one is sufficient here
            if np.sum(np.isfinite(r)) > 0:
                idx = np.where(abs(r) == min(abs(r)))[0][0]
                u_root = r[idx]
                exitflag = 1
            else:
                u_root = []
                exitflag = 0

        # filter out complex roots
        if np.iscomplex(u_root):
            u_root = []
            exitflag = 0

    return [u_root, exitflag]
