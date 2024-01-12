## Find the root(s) of a function by secant and regula falsi updates
"""
---------------------------------------------------------------------------
Created by:
Iason Papaioannou
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Current version 2021-05
---------------------------------------------------------------------------
Input:
* g_1D: desired function for which to find the root(s)
* u_0: starting point
---------------------------------------------------------------------------
Output:
* u_root: found root (either the found value or empty [])
* n_eval: number of evaluations it needed to find the root(s) of the g_1D
* exitflag: flag whether a root was found (1) or not (0)
"""

import numpy as np
import math

def find_root(g_1D, u_0):

    n_eval = 0

    # Parameters for algorithm to find beta_i
    maxiter = 10
    hfactor = 0.1
    g_0 = g_1D(u_0)
    n_eval += 1

    if not math.isfinite(g_0):
        # Decrease the distance to origin
        for j in range(1, 6):
            u_0 = u_0*((1/2)**j)

            g_0 = g_1D(u_0)
            n_eval += 1

            if math.isfinite(g_0):
                break

    # set tolerance for stopping criterion
    if not math.isfinite(g_0):
        tolg = 1e-4*g_0
    else:
        tolg = 1e-4
    tolu = 1e-4

    # initial step

    u_new = (1-hfactor)*u_0
    u_old = u_0
    g_old = g_0
    du = u_new-u_0

    iter = 1

    g_new = g_1D(u_new)
    n_eval += 1

    if not math.isfinite(g_new):
        # Decrease the distance to origin
        for j in range(1, 6):

            u_new = (1-hfactor**(j+1))*beta_first
            du = u_new-u_0

            g_new = g_1D(u_new)
            n_eval += 1

            if math.isfinite(g_new):
                break

    first_rf = 0

    # Begin line search iterations
    while True:

        iter += 1

        if first_rf == 0:
            if g_new*g_old < 0:
                # first regula falsi step
                first_rf = 1
                if g_new > 0:
                    [g_new, g_old] = [g_old, g_new]
                    [u_new, u_old] = [u_old, u_new]

        # secant/regula falsi step
        du = (u_old-u_new)*g_new/(g_new-g_old)

        # Update u
        u_latest = u_new + du

        g_latest = g_1D(u_latest)
        n_eval += 1

        if not math.isfinite(g_latest):
            # Decrease the distance to origin
            for j in range(1, 6):
                u_latest = u_new + du*((1/2)**j)
                g_latest = g_1D(u_latest)
                n_eval += 1

                if math.isfinite(g_latest):
                    break

        if first_rf == 1:
            if g_latest < 0:
                u_new = u_latest
                g_new = g_latest
            else:
                u_old = u_latest
                g_old = g_latest
                du = u_latest - u_old

        else:
            [u_old, u_new] = [u_new, u_latest]
            [g_old, g_new] = [g_new, g_latest]

        # Compute errors
        erru = abs(du)
        errg = abs(g_latest)

        if (erru < tolu) and (errg < tolg):
            u_root = u_latest
            exitflag = 1
            break

        elif iter > maxiter:
            u_root = []
            exitflag = 0
            break

    return [u_root, n_eval, exitflag]
