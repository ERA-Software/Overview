# import of modules
import numpy as np
import scipy as sp
from scipy import optimize, stats, special
import warnings

"""
---------------------------------------------------------------------------
Generation of distribution objects
---------------------------------------------------------------------------
Developed by:
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Luca Sardi
Fong-Lin Wu
Alexander von Ramm
Matthias Willer
Peter Kaplan

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Version 2021-03:
* General update to match the current MATLAB version
* Implementation of all the missing distribution types for the option
  'DATA'. Now all distribution types are available for all three options
  'MOM','PAR' and 'DATA'
*  Introduction of the truncated normal distribution  
Version 2020-05:
* Fixing bugs in exponential constructor ( remove val=val[0] )
Version 2019-11:
* Fixing bug with Weibull distribution parameters
Version 2019-01:
* Automatic import of required scipy subpackages
* Fixing of bugs in the lognormal and exponential distribution
* Optimization and fixing of minor bugs
---------------------------------------------------------------------------
This software generates distribution objects according to the parameters
and definitions used in the distribution table of the ERA Group of TUM.
They can be defined either by their parameters, the first and second
moment or by data, given as a vector.
---------------------------------------------------------------------------
"""


class ERADist(object):
    """
    Generation of marginal distribution objects.
    Construction of the distribution object with 
      
            Obj = ERADist(name,opt,val)
    or      Obj = ERADist(name,opt,val,ID)
    
    The second option is only useful when using the ERADist object within
    the scope of an ERARosen object.
    
    
    The following distribution types are available:

      opt = "PAR", specification of the distribution by its parameters:
      Beta:                       Obj = ERADist('beta','PAR',[r,s,a,b])
      Binomial:                   Obj = ERADist('binomial','PAR',[n,p])
      Chi-squared:                Obj = ERADist('chisquare','PAR',[k])
      Exponential:                Obj = ERADist('exponential','PAR',[lambda])
      Fréchet:                    Obj = ERADist('frechet','PAR',[a_n,k])
      Gamma:                      Obj = ERADist('gamma','PAR',[lambda,k])
      Geometric:                  Obj = ERADist('geometric','PAR',[p])
      GEV (to model maxima):      Obj = ERADist('GEV','PAR',[beta,alpha,epsilon])
      GEV (to model minima):      Obj = ERADist('GEVMin','PAR',[beta,alpha,epsilon])
      Gumbel (to model maxima):   Obj = ERADist('gumbel','PAR',[a_n,b_n])
      Gumbel (to model minima):   Obj = ERADist('gumbelMin','PAR',[a_n,b_n])
      Log-normal:                 Obj = ERADist('lognormal','PAR',[mu_lnx,sig_lnx])
      Negative binomial:          Obj = ERADist('negativebinomial','PAR',[k,p])
      Normal:                     Obj = ERADist('normal','PAR',[mean,std])
      Pareto:                     Obj = ERADist('pareto','PAR',[x_m,alpha])
      Poisson:                    Obj = ERADist('poisson','PAR',[v,t])
                               or Obj = ERADist('poisson','PAR',[lambda])
      Rayleigh:                   Obj = ERADist('rayleigh','PAR',[alpha])
      Standard normal:            Obj = ERADist('standardnormal','PAR',[])
      Truncated normal:           Obj = ERADist('truncatednormal','PAR',[mu_n,sigma_n,a,b])
      Uniform:                    Obj = ERADist('uniform','PAR',[lower,upper])
      Weibull:                    Obj = ERADist('weibull','PAR',[a_n,k])
      
      
      opt = "MOM", specification of the distribution by its moments:
      Beta:                       Obj = ERADist('beta','MOM',[mean,std,a,b])
      Binomial:                   Obj = ERADist('binomial','MOM',[mean,std])
      Chi-squared:                Obj = ERADist('chisquare','MOM',[mean])
      Exponential:                Obj = ERADist('exponential','MOM',[mean])
      Fréchet:                    Obj = ERADist('frechet','MOM',[mean,std])
      Gamma:                      Obj = ERADist('gamma','MOM',[mean,std])
      Geometric:                  Obj = ERADist('geometric','MOM',[mean])
      GEV (to model maxima):      Obj = ERADist('GEV','MOM',[mean,std,epsilon])
      GEV (to model minima):      Obj = ERADist('GEVMin','MOM',[mean,std,epsilon])
      Gumbel (to model maxima):   Obj = ERADist('gumbel','MOM',[mean,std])
      Gumbel (to model minima):   Obj = ERADist('gumbelMin','MOM',[mean,std])
      Log-normal:                 Obj = ERADist('lognormal','MOM',[mean,std])
      Negative binomial:          Obj = ERADist('negativebinomial','MOM',[mean,std])
      Normal:                     Obj = ERADist('normal','MOM',[mean,std])
      Pareto:                     Obj = ERADist('pareto','MOM',[mean,std])
      Poisson:                    Obj = ERADist('poisson','MOM',[mean,t])
                              or  Obj = ERADist('poisson','MOM',[mean])
      Rayleigh:                   Obj = ERADist('rayleigh','MOM',[mean])
      Standard normal:            Obj = ERADist('standardnormal','MOM',[])
      Truncated normal:           Obj = ERADist('truncatednormal','MOM',[mean,std,a,b])
      Uniform:                    Obj = ERADist('uniform','MOM',[mean,std])
      Weibull:                    Obj = ERADist('weibull','MOM',[mean,std])
      
      
      opt = "DATA", specification of the distribution by data given as a vector:
      Beta:                       Obj = ERADist('beta','DATA',[[X],[a,b]])
      Binomial:                   Obj = ERADist('binomial','DATA',[[X],n])
      Chi-squared:                Obj = ERADist('chisquare','DATA',[X])
      Exponential:                Obj = ERADist('exponential','DATA',[X])
      Frechet:                    Obj = ERADist('frechet','DATA',[X])
      Gamma:                      Obj = ERADist('gamma','DATA',[X])
      Geometric:                  Obj = ERADist('geometric','DATA',[X])
      GEV (to model maxima):      Obj = ERADist('GEV','DATA',[X])
      GEV (to model minima):      Obj = ERADist('GEVMin','DATA',[X])
      Gumbel (to model maxima):   Obj = ERADist('gumbel','DATA',[X])
      Gumbel (to model minima):   Obj = ERADist('gumbelMin','DATA',[X])
      Log-normal:                 Obj = ERADist('lognormal','DATA',[X])
      Negative binomial:          Obj = ERADist('negativebinomial','DATA',[X])
      Normal:                     Obj = ERADist('normal','DATA',[X])
      Pareto:                     Obj = ERADist('pareto','DATA',[X])
      Poisson:                    Obj = ERADist('poisson','DATA',[[X],t])
                              or  Obj = ERADist('poisson','DATA',[X])
      Rayleigh:                   Obj = ERADist('rayleigh','DATA',[X])
      Truncated normal:           Obj = ERADist('truncatednormal','DATA',[[X],[a,b]])
      Uniform:                    Obj = ERADist('uniform','DATA',[X])
      Weibull:                    Obj = ERADist('weibull','DATA',[X])
          
    """
#%%
    def __init__(self, name, opt, val=[0, 1], ID=False):
        """
        Constructor method, for more details have a look at the
        class description.
        """
        
        self.Name = name.lower()
        self.ID = ID
        
        #----------------------------------------------------------------------------
        # definition of the distribution by its parameters
        if opt.upper() == "PAR":
            val = np.array(val, ndmin=1, dtype=float)
            
            
            if name.lower() == "beta":
                """
                beta distribution in lecture notes can be shifted in order to
                account for ranges [a,b] -> this is not implemented yet
                """
                if (val[0] > 0) and (val[1] > 0) and (val[2] < val[3]):
                    self.Par = {'r':val[0],'s':val[1],'a':val[2],'b':val[3]}
                    self.Dist = stats.beta(a=self.Par['r'], b=self.Par['s'],
                                loc=self.Par['a'],scale = self.Par['b']-self.Par['a'])
                else:
                    raise RuntimeError("The Beta distribution is not defined for your parameters.")
            
            
            elif name.lower() == "binomial":
                if (val[1] >= 0) and (val[1] <= 1) and (val[0] % 1 == 0):
                    self.Par = {'n':int(val[0]), 'p':val[1]}
                    self.Dist = stats.binom(n=self.Par['n'], p=self.Par['p'])
                else:
                    raise RuntimeError(
                        "The Binomial distribution is not defined for your parameters.")
                    
            
            elif name.lower() == "chisquare":
                if val[0] > 0 and val[0] < np.inf and val[0] % 1 <= 10 ** (-4):
                    self.Par = {'k':np.around(val[0],0)}
                    self.Dist = stats.chi2(df=self.Par['k'])
                else:
                    raise RuntimeError(
                        "The Chi-Squared distribution is not defined for your parameters.")
                    
            
            elif name.lower() == "exponential":
                if val[0] > 0:
                    self.Par = {'lambda':val[0]}
                    self.Dist = stats.expon(scale=1 / self.Par['lambda'])
                else:
                    raise RuntimeError(
                        "The Exponential distribution is not defined for your parameters.")
                    
            
            elif name.lower() == "frechet":
                if (val[0] > 0) and (val[1] > 0):
                    self.Par = {'a_n':val[0],'k':val[1]}
                    self.Dist = stats.genextreme(
                        c=-1 / self.Par['k'], scale=self.Par['a_n'] / self.Par['k'], loc=self.Par['a_n'])
                else:
                    raise RuntimeError(
                        "The Frechet distribution is not defined for your parameters.")
                    
            
            elif name.lower() == "gamma":
                if val[0] > 0 and val[1] > 0:
                    self.Par = {'lambda':val[0], 'k':val[1]}
                    self.Dist = stats.gamma(a=self.Par['k'], scale=1/self.Par['lambda'])
                else:
                    raise RuntimeError(
                        "The Gamma distribution is not defined for your parameters.")

            
            elif name.lower() == "geometric":
                val = val[0]
                if val > 0 and val <= 1:
                    self.Par = {'p':val}
                    self.Dist = stats.geom(p=self.Par['p'])
                else:
                    raise RuntimeError(
                        "The Geometric distribution is not defined for your parameters.")
                    
            
            elif name.lower() == "gev":
                if val[1] > 0:
                    self.Par = {'beta':val[0], 'alpha':val[1], 'epsilon':val[2]}
                    self.Dist = stats.genextreme(c=-self.Par['beta'], scale=self.Par['alpha'], loc=self.Par['epsilon'])
                else:
                    raise RuntimeError(
                        "The Generalized Extreme Value gistribution is not defined for your parameters.")

            
            elif name.lower() == "gevmin":
                if val[1] > 0:
                    self.Par = {'beta':val[0], 'alpha':val[1], 'epsilon':val[2]}
                    self.Dist = stats.genextreme(c=-self.Par['beta'], scale=self.Par['alpha'], loc=-self.Par['epsilon'])
                else:
                    raise RuntimeError(
                        "The Generalized Extreme Value distribution is not defined for your parameters.")
                    
            
            elif name.lower() == "gumbel":
                if val[0] > 0:
                    self.Par = {'a_n':val[0], 'b_n':val[1]}
                    self.Dist = stats.gumbel_r(scale=self.Par['a_n'], loc=self.Par['b_n'])
                else:
                    raise RuntimeError(
                        "The Gumbel distribution is not defined for your parameters.")
                    
            
            elif name.lower() == "gumbelmin":
                if val[0] > 0:
                    self.Par = {'a_n':val[0], 'b_n':val[1]}
                    self.Dist = stats.gumbel_l(scale=self.Par['a_n'], loc=self.Par['b_n'])
                else:
                    raise RuntimeError(
                        "The Gumbel distribution is not defined for your parameters.")
                    
            
            elif name.lower() == "lognormal":
                if val[1] > 0:
                    self.Par = {'mu_lnx':val[0],'sig_lnx':val[1]}
                    self.Dist = stats.lognorm(s=self.Par['sig_lnx'], scale=np.exp(self.Par['mu_lnx']))
                else:
                    raise RuntimeError(
                        "The Lognormal distribution is not defined for your parameters.")

            
            elif name.lower() == "negativebinomial":
                if (
                    (val[1] > 0)
                    and (val[1] <= 1)
                    and (val[0] > 0)
                    and (val[0] % 1 == 0)
                ):
                    self.Par = {'k':val[0], 'p':val[1]}
                    self.Dist = stats.nbinom(n=self.Par['k'], p=self.Par['p'])
                else:
                    raise RuntimeError(
                        "The Negative Binomial distribution is not defined for your parameters.")
            
            
            elif name.lower() == "normal" or name.lower() == "gaussian":
                if val[1] > 0:
                    self.Par = {'mu':val[0], 'sigma':val[1]}
                    self.Dist = stats.norm(loc=self.Par['mu'], scale=self.Par['sigma'])
                else:
                    raise RuntimeError(
                        "The Normal distribution is not defined for your parameters.")
                    
            
            elif name.lower() == "pareto":
                if val[0] > 0 and val[1] > 0:
                    self.Par = {'x_m':val[0],'alpha':val[1]}
                    self.Dist = stats.genpareto(c=1 / self.Par['alpha'],
                                scale=self.Par['x_m']/self.Par['alpha'], loc=self.Par['x_m'])
                else:
                    raise RuntimeError(
                        "The Pareto distribution is not defined for your parameters.")

            
            elif name.lower() == "poisson":
                n = len(val)
                if n == 1:
                    if val > 0:
                        self.Par = {'lambda':val[0]}
                        self.Dist = stats.poisson(mu=self.Par['lambda'])
                    else:
                        raise RuntimeError(
                            "The Poisson distribution is not defined for your parameters.")
                        
                if n == 2:
                    if val[0] > 0 and val[1] > 0:
                        self.Par = {'v':val[0],'t':val[1]}
                        self.Dist = stats.poisson(mu=self.Par['v'] * self.Par['t'])
                    else:
                        raise RuntimeError(
                            "The Poisson distribution is not defined for your parameters.")
            
            
            elif name.lower() == "rayleigh":
                alpha = val[0]
                if alpha > 0:
                    self.Par = {'alpha':alpha}
                    self.Dist = stats.rayleigh(scale=self.Par['alpha'])
                else:
                    raise RuntimeError(
                        "The Rayleigh distribution is not defined for your parameters.")
                   
                    
            elif (name.lower() == "standardnormal") or (name.lower() == "standardgaussian"):
                self.Par = {'mu':0, 'sigma':1}
                self.Dist = stats.norm(loc=0, scale=1)
                
                
            elif name.lower() == "truncatednormal":
                if val[2] >= val[3]:
                    raise RuntimeError("The upper bound a must be larger than the lower bound b.")
                if val[1] < 0:
                    raise RuntimeError("sigma must be larger than 0.")
                self.Par = {'mu_n':val[0], 'sig_n':val[1], 'a':val[2], 'b':val[3]}
                a_mod = (self.Par['a']-self.Par['mu_n'])/self.Par['sig_n']
                b_mod = (self.Par['b']-self.Par['mu_n'])/self.Par['sig_n']
                self.Dist =  stats.truncnorm(loc=self.Par['mu_n'],scale=self.Par['sig_n'], a=a_mod, b=b_mod)
                
                
            elif name.lower() == "uniform":
                if val[0] < val[1]:
                    self.Par = {'lower':val[0], 'upper': val[1]}
                    self.Dist = stats.uniform(loc=self.Par['lower'], scale=self.Par['upper'] - self.Par['lower'])
                else:
                    raise RuntimeError(
                        "The Uniform distribution is not defined for your parameters.")


            elif name.lower() == "weibull":
                if (val[0] > 0) and (val[1] > 0):
                    self.Par = {'a_n':val[0], 'k':val[1]}
                    self.Dist = stats.weibull_min(c=self.Par['k'], scale=self.Par['a_n'])
                else:
                    raise RuntimeError(
                    "The Weibull distribution is not defined for your parameters.")

            else:
                raise RuntimeError("Distribution type '" + name + "' not available.")
               
        #----------------------------------------------------------------------------
        # if the distribution is defined by its moments
        elif opt.upper() == "MOM":
            val = np.array(val, ndmin=1, dtype=float)
            
            if val.size > 1 and val[1] < 0:
                raise RuntimeError("The standard deviation must be non-negative.")

            if name.lower() == 'beta':
                if val[3] <= val[2]:
                    raise RuntimeError("Please select an other support [a,b].")
                r = ((val[3]-val[0])*(val[0]-val[2])/val[1]**2-1)*(val[0]-val[2])/(val[3]-val[2])
                s = r*(val[3]-val[0])/(val[0]-val[2])
                # Evaluate if distribution can be defined on the parameters
                if r <= 0 and s <= 0:
                    raise RuntimeError("Please select other moments.")
                self.Par = {'r':r,'s':s,'a':val[2],'b':val[3]}
                self.Dist = stats.beta(a=self.Par['r'], b=self.Par['s'],
                            loc=self.Par['a'],scale = self.Par['b']-self.Par['a'])             
                

            elif name.lower() == "binomial":
                # Solve system of two equations for the parameters
                p = 1 - (val[1]) ** 2 / val[0]
                n = val[0] / p
                # Evaluate if distribution can be defined on the parameters
                if n % 1 <= 10 ** (-4):
                    n = int(n)
                else:
                    raise RuntimeError("Please select other moments.")
                if 0 <= p and p <= 1 and 0 < n:
                        self.Par = {'n':n, 'p':p}
                        self.Dist = stats.binom(n=self.Par['n'], p=self.Par['p'])
                else:
                    raise RuntimeError("Please select other moments.")
                    
                    
            elif name.lower() == "chisquare":
                if val[0] > 0 and val[0] < np.inf and val[0] % 1 <= 10 ** (-4):
                    self.Par = {'k':np.around(val[0],0)}
                    self.Dist = stats.chi2(df=self.Par['k'])
                else:
                    raise RuntimeError("Please select other moments.")
                        
                    
            elif name.lower() == "exponential":
                try:
                    lam = 1 / val[0]
                except ZeroDivisionError:
                    raise RuntimeError("The first moment cannot be zero!")
                if 0 <= lam:
                    self.Par = {'lambda':lam}
                    self.Dist = stats.expon(scale=1 / self.Par['lambda'])
                else:
                    raise RuntimeError("Please select other moments.")

            
            elif name.lower() == "frechet":
                par0 = 2.0001

                def equation(par):
                    return (np.sqrt(special.gamma(1 - 2 / par)- special.gamma(1 - 1 / par) ** 2)
                                / special.gamma(1 - 1 / par)- val[1] / val[0])

                sol = optimize.fsolve(equation, x0=par0, full_output=True)
                if sol[2] == 1:
                    k = sol[0][0]
                    a_n = val[0] / special.gamma(1 - 1 / k)
                else:
                    raise RuntimeError(
                        "fsolve could not converge to a solution, therefore"
                        "the parameters of the Frechet distribution could not be determined.")
                if a_n > 0 and k > 0:
                    self.Par = {'a_n':a_n,'k':k}
                    self.Dist = stats.genextreme(
                        c=-1 / self.Par['k'], scale=self.Par['a_n'] / self.Par['k'], loc=self.Par['a_n'])
                else:
                    raise RuntimeError("Please select other moments.")
                    
            
            elif name.lower() == "gamma":
                # Solve system of equations for the parameters
                lam = val[0] / (val[1] ** 2)
                k = lam * val[0]
                # Evaluate if distribution can be defined on the parameters
                if lam > 0 and k > 0:
                    self.Par = {'lambda':lam, 'k':k}
                    self.Dist = stats.gamma(a=self.Par['k'], scale=1/self.Par['lambda'])
                else:
                    raise RuntimeError("Please select other moments.")
                    
                    
            elif name.lower() == "geometric":
                # Solve Equation for the parameter based on the first moment
                p = 1 / val[0]
                if 0 <= p and p <= 1:
                    self.Par = {'p':p}
                    self.Dist = stats.geom(p=self.Par['p'])
                else:
                    raise RuntimeError("Please select other moments.")
                    
            
            elif name.lower() == "gev":
                beta = val[2]
                if beta == 0: # corresponds to Gumbel distribution
                    # Solve two equations for the parameters of the distribution
                    alpha = val[1]*np.sqrt(6)/np.pi   # scale parameter
                    epsilon = val[2] - np.euler_gamma*alpha   # location parameter
                elif beta >= 0.5:
                    raise RuntimeError("MOM can only be used for beta < 0.5 .")
                else:
                    alpha = abs(beta)*val[1]/np.sqrt(special.gamma(1-2*beta)-special.gamma(1-beta)**2)
                    epsilon = val[0]-(alpha/beta*(special.gamma(1-beta)-1))
                self.Par = {'beta':beta, 'alpha':alpha, 'epsilon':epsilon}
                self.Dist = stats.genextreme(c=-self.Par['beta'], scale=self.Par['alpha'], loc=self.Par['epsilon'])
                

            elif name.lower() == "gevmin":
                beta = val[2]
                if beta == 0: # corresponds to Gumbel distribution
                    # Solve two equations for the parameters of the distribution
                    alpha = val[1]*np.sqrt(6)/np.pi   # scale parameter
                    epsilon = val[2] + np.euler_gamma*alpha   # location parameter
                elif beta >= 0.5:
                    raise RuntimeError("MOM can only be used for beta < 0.5 .")
                else:
                    alpha = abs(beta)*val[1]/np.sqrt(special.gamma(1-2*beta)-special.gamma(1-beta)**2)
                    epsilon = val[0]+(alpha/beta*(special.gamma(1-beta)-1))
                self.Par = {'beta':beta, 'alpha':alpha, 'epsilon':epsilon}
                self.Dist = stats.genextreme(c=-self.Par['beta'], scale=self.Par['alpha'], loc=-self.Par['epsilon'])
            
            
            elif name.lower() == "gumbel":
                # solve two equations for the parameters of the distribution
                a_n = val[1] * np.sqrt(6) / np.pi  # scale parameter
                b_n = val[0] - np.euler_gamma * a_n  # location parameter
                if a_n > 0:
                    self.Par = {'a_n':a_n, 'b_n':b_n}
                    self.Dist = stats.gumbel_r(scale=self.Par['a_n'], loc=self.Par['b_n'])
                else:
                    raise RuntimeError("Please select other moments.")

            
            elif name.lower() == "gumbelmin":
                # solve two equations for the parameters of the distribution
                a_n = val[1] * np.sqrt(6) / np.pi  # scale parameter
                b_n = val[0] + np.euler_gamma * a_n  # location parameter
                if a_n > 0:
                    self.Par = {'a_n':a_n, 'b_n':b_n}
                    self.Dist = stats.gumbel_l(scale=self.Par['a_n'], loc=self.Par['b_n'])
                else:
                    raise RuntimeError("Please select other moments.")
                    
            
            elif name.lower() == "lognormal":
                if val[0] <= 0:
                    raise RuntimeError(
                        "Please select other moments, the first moment must be greater than zero.")
                # solve two equations for the parameters of the distribution
                mu_lnx = np.log(val[0] ** 2 / np.sqrt(val[1] ** 2 + val[0] ** 2))
                sig_lnx = np.sqrt(np.log(1 + (val[1] / val[0]) ** 2))
                self.Par = {'mu_lnx':mu_lnx,'sig_lnx':sig_lnx}
                self.Dist = stats.lognorm(s=self.Par['sig_lnx'], scale=np.exp(self.Par['mu_lnx']))
                
                
            elif name.lower() == "negativebinomial":
                # Solve System of two equations for the parameters
                p = val[0] / (val[0] + val[1] ** 2)
                k = val[0] * p
                # Evaluate if distribution can be defined on the parameters
                if k % 1 <= 10 ** (-4):
                    k = round(k, 0)
                    if 0 <= p and p <= 1:
                        self.Par = {'k':k, 'p':p}
                        self.Dist = stats.nbinom(n=self.Par['k'], p=self.Par['p'])
                    else:
                        raise RuntimeError("Please select other moments.")
                else:
                    raise RuntimeError("Please select other moments.")

            
            elif (name.lower() == "normal") or (name.lower() == "gaussian"):
                self.Par = {'mu':val[0], 'sigma':val[1]}
                self.Dist = stats.norm(loc=self.Par['mu'], scale=self.Par['sigma'])
                
                
            elif name.lower() == "pareto":
                alpha = 1 + np.sqrt(1 + (val[0] / val[1]) ** 2)
                x_m = val[0] * (alpha - 1) / alpha
                if x_m > 0 and alpha > 0:
                    self.Par = {'x_m':x_m,'alpha':alpha}
                    self.Dist = stats.genpareto(c=1 / self.Par['alpha'],
                                scale=self.Par['x_m']/self.Par['alpha'], loc=self.Par['x_m'])
                else:
                    raise RuntimeError("Please select other moments.")
                
                
            elif name.lower() == "poisson":
                n = len(val)
                if n == 1:
                    if val > 0:
                        self.Par = {'lambda':val[0]}
                        self.Dist = stats.poisson(mu=self.Par['lambda'])
                    else:
                        raise RuntimeError("Please select other moments.")
                        
                if n == 2:
                    if val[0] > 0 and val[1] > 0:
                        v = val[0]/val[1]
                        if  val[1] <= 0:
                            raise RuntimeError("t must be positive.")   
                        self.Par = {'v':v,'t':val[1]}
                        self.Dist = stats.poisson(mu=self.Par['v'] * self.Par['t'])
                    else:
                        raise RuntimeError("Please select other moments.")
                    
                    
            elif name.lower() == "rayleigh":
                alpha = val[0] / np.sqrt(np.pi / 2)
                if alpha > 0:
                    self.Par = {'alpha':alpha}
                    self.Dist = stats.rayleigh(scale=self.Par['alpha'])
                else:
                    raise RuntimeError("Please select other moments.")
                    
                    
            elif (name.lower() == "standardnormal") or (name.lower() == "standardgaussian"):
                self.Par = {'mu':0, 'sigma':1}
                self.Dist = stats.norm(loc=0, scale=1)
                
                
            elif name.lower() == "truncatednormal":
                if val[2] >= val[3]:
                    raise RuntimeError("The upper bound a must be larger than the lower bound b.")
                if val[0] <= val[2] or val[0] >= val[3]:
                      raise RuntimeError('The mean of the distribution must be within the interval [a,b].')
                def equation(par):
                    f = lambda x: stats.norm.pdf(x,par[0],par[1])/(stats.norm.cdf(val[3],par[0],par[1])-stats.norm.cdf(val[2],par[0],par[1]))
                    expec_eq = sp.integrate.quad(lambda x: x*f(x),val[2],val[3])[0]-val[0]
                    std_eq = np.sqrt(sp.integrate.quad(lambda x: x**2*f(x),val[2],val[3])[0]-(sp.integrate.quad(lambda x: x*f(x),val[2],val[3]))[0]**2)-val[1]
                    eq = [expec_eq, std_eq]
                    return(eq)
                
                x0 = [val[0],val[1]]
                sol = optimize.fsolve(equation, x0=x0, full_output=True)
                if sol[2] == 1:
                    self.Par = {'mu_n':sol[0][0], 'sig_n':sol[0][1], 'a':val[2], 'b':val[3]}
                    a_mod = (self.Par['a']-self.Par['mu_n'])/self.Par['sig_n']
                    b_mod = (self.Par['b']-self.Par['mu_n'])/self.Par['sig_n']
                    self.Dist =  stats.truncnorm(loc=self.Par['mu_n'],scale=self.Par['sig_n'], a=a_mod, b=b_mod)
                else:
                    raise RuntimeError("fsolve did not converge.")
                
                
            elif name.lower() == "uniform":
                # compute parameters
                lower = val[0] - np.sqrt(12) * val[1] / 2
                upper = val[0] + np.sqrt(12) * val[1] / 2
                self.Par = {'lower':lower, 'upper': upper}
                self.Dist = stats.uniform(loc=self.Par['lower'], scale=self.Par['upper'] - self.Par['lower'])
                
                
            elif name.lower() == "weibull":
                def equation(par):
                    return (np.sqrt(special.gamma(1 + 2 / par) - (special.gamma(1 + 1 / par)) ** 2)
                        / special.gamma(1 + 1 / par) - val[1] / val[0])

                sol = optimize.fsolve(equation, x0=0.02, full_output=True)
                if sol[2] == 1:
                    k = sol[0][0]
                    a_n = val[0] / special.gamma(1 + 1 / k)
                else:
                    raise RuntimeError(
                        "fsolve could not converge to a solution, therefore"
                        "the parameters of the Weibull distribution could not be determined.")
                if a_n > 0 and k > 0:
                    self.Par = {'a_n':a_n, 'k':k}
                    self.Dist = stats.weibull_min(c=self.Par['k'], scale=self.Par['a_n'])
                else:
                    raise RuntimeError("Please select other moments.")
                    
                    
            else:
                raise RuntimeError("Distribution type '" + name + "' not available.")

        #----------------------------------------------------------------------------
        # if the distribution is to be fitted to a data vector
        elif opt.upper() == "DATA":
            
            if name.lower() == "beta":
                if val[2] <= val[1]:
                    raise RuntimeError("Please select a different support [a,b].")
                if min(val[0]) >= val[1] and max(val[0]) <= val[2]:
                    pars = stats.beta.fit(val[0], floc=val[1], fscale=val[2]-val[1])
                    self.Par = {'r':pars[0],'s':pars[1],'a':val[1],'b':val[2]}
                    self.Dist = stats.beta(a=self.Par['r'], b=self.Par['s'],
                                loc=self.Par['a'],scale = self.Par['b']-self.Par['a'])  
                else:
                    raise RuntimeError("The given samples must be in the support range [a,b].")
        
            
            elif name.lower() == "binomial":
                # Evaluate if distribution can be defined on the parameters
                if val[1] % 1 <= 10 ** (-4) and val[1] > 0:
                    val[1] = int(val[1])
                else:
                    raise RuntimeError("n must be a positive integer.")
                X = np.array(val[0])
                if all((X) % 1 <= 10 ** (-4)) and all(X >= 0) and all(X <= val[1]):
                    X = np.around(X, 0)
                else:
                    raise RuntimeError("The given samples must be integers in the range [0,n].")
                val[0] = np.mean(val[0])/val[1]
                self.Par = {'n':val[1], 'p':val[0]}
                self.Dist = stats.binom(n=self.Par['n'], p=self.Par['p'])
                
            
            elif name.lower() == "chisquare":
                if min(val) >= 0:
                    pars = stats.chi2.fit(val, floc=0, fscale=1)  
                    self.Par = {'k':np.around(pars[0],0)}
                    self.Dist = stats.chi2(df=self.Par['k'])
                else:
                     raise RuntimeError("The given samples must be non-negative.")           
                
            
            elif name.lower() == "exponential":
                if min(val) >= 0:
                    pars = stats.expon.fit(val, floc=0)
                    self.Par = {'lambda':1 / pars[1]}
                    self.Dist = stats.expon(scale=1 / self.Par['lambda'])
                else:
                     raise RuntimeError("The given samples must be non-negative.")
   

            elif name.lower() == "frechet":
                if min(val) < 0:
                    raise RuntimeError("The given samples must be non-negative.")
                def equation(par):
                    return (-np.sum(np.log(stats.genextreme.pdf(val,c=-1 / par[1], scale=par[0] / par[1], loc=par[0]))))
                    
                par1 = 2.0001
                par0 = par1 / special.gamma(1 - 1 / np.mean(val))
                x0 = np.array([par0,par1])
                bnds = optimize.Bounds(lb=[0,0],ub=[np.inf,np.inf])
                sol = optimize.minimize(equation, x0, bounds=bnds)
                if sol.success == True:                
                    self.Par = {'a_n':sol.x[0],'k':sol.x[1]}
                    self.Dist = stats.genextreme(
                        c=-1 / self.Par['k'], scale=self.Par['a_n'] / self.Par['k'], loc=self.Par['a_n'])
                else:
                    raise RuntimeError("Maximum likelihood estimation did not converge.")
        
                
            elif name.lower() == "gamma":
                pars = stats.gamma.fit(val, floc=0)
                self.Par = {'lambda':1 / pars[2], 'k':pars[0]}
                self.Dist = stats.gamma(a=self.Par['k'], scale=1/self.Par['lambda'])
                
            
            elif name.lower() == "geometric":
                if all(val > 0) and all(val %1 == 0):
                    self.Par = {'p':1/np.mean(val)}
                    self.Dist = stats.geom(p=self.Par['p'])
                else:
                    raise RuntimeError("The given samples must be integers larger than 0.")
                
                
            elif name.lower() == "gev":
                pars = gevfit_alt(np.squeeze(val))
                self.Par = {'beta':pars[0], 'alpha':pars[1], 'epsilon':pars[2]}
                self.Dist = stats.genextreme(c=-self.Par['beta'], scale=self.Par['alpha'], loc=self.Par['epsilon'])

            
            elif name.lower() == "gevmin":
                pars = gevfit_alt(np.squeeze(-val))
                self.Par = {'beta':pars[0], 'alpha':pars[1], 'epsilon':-pars[2]}
                self.Dist = stats.genextreme(c=-self.Par['beta'], scale=self.Par['alpha'], loc=-self.Par['epsilon'])
                
                
            elif name.lower() == "gumbel":
                pars = stats.gumbel_r.fit(val)
                self.Par = {'a_n':pars[1], 'b_n':pars[0]}
                self.Dist = stats.gumbel_r(scale=self.Par['a_n'], loc=self.Par['b_n'])

            
            elif name.lower() == "gumbelmin":
                pars = stats.gumbel_l.fit(val)
                self.Par = {'a_n':pars[1], 'b_n':pars[0]}
                self.Dist = stats.gumbel_l(scale=self.Par['a_n'], loc=self.Par['b_n'])
            
            
            elif name.lower() == "lognormal":
                pars = stats.lognorm.fit(val, floc=0)
                self.Par = {'mu_lnx':np.log(pars[2]),'sig_lnx':pars[0]}
                self.Dist = stats.lognorm(s=self.Par['sig_lnx'], scale=np.exp(self.Par['mu_lnx']))
                
                
            elif name.lower() == "negativebinomial":
                # first estimation of k,p with method of moments
                p = np.mean(val)/(np.mean(val)+np.var(val))
                k = np.mean(val)*p
                if k==0:
                    raise RuntimeError("No suitable parameters can be estimated from the given data.")
                k = round(k, 0) # rounding of k, since k must be a positive integer according to ERADist definition
                p = k/np.mean(val); # estimation of p for rounded k (mle)
                self.Par = {'k':k, 'p':p}
                self.Dist = stats.nbinom(n=self.Par['k'], p=self.Par['p'])

            
            elif name.lower() == "normal" or name.lower() == "gaussian":
                pars = stats.norm.fit(val)
                self.Par = {'mu':pars[0], 'sigma':pars[1]}
                self.Dist = stats.norm(loc=self.Par['mu'], scale=self.Par['sigma'])
                
                
            elif name.lower() == "pareto":
                x_m = min(val)
                if x_m > 0:
                    def equation(par):
                            return (-np.sum(np.log(stats.genpareto.pdf(val,c = 1 / par, scale = x_m / par, loc = x_m))))

                    x0 = x_m
                    sol = optimize.minimize(equation, x0)
                    if sol.success == True:
                        self.Par = {'x_m':x_m,'alpha':float(sol.x)}
                        self.Dist = stats.genpareto(c=1 / self.Par['alpha'],
                                scale=self.Par['x_m']/self.Par['alpha'], loc=self.Par['x_m'])
                    else:
                        raise RuntimeError("Maximum likelihood estimation did not converge.")
                else:
                    raise RuntimeError("The given data must be positive.")
                
                
            elif name.lower() == "poisson":
                n = len(val)
                if n == 2:
                    X = val[0]
                    t = val[1]
                    if t <= 0:
                        raise RuntimeError("t must be positive.")                                            
                    if all(X >=0) and all(X %1 == 0):
                        v = np.mean(X)/t
                        self.Par = {'v':v,'t':t}
                        self.Dist = stats.poisson(mu=self.Par['v'] * self.Par['t'])
                    else:
                        raise RuntimeError("The given samples must be non-negative integers.")                       
                else:
                    if all(val >= 0) and all(val %1 == 0):
                        self.Par = {'lambda':np.mean(val)}
                        self.Dist = stats.poisson(mu=self.Par['lambda'])
                    else:
                        raise RuntimeError("The given samples must be non-negative integers.")
            
            
            elif name.lower() == "rayleigh":
                pars = stats.rayleigh.fit(val, floc=0)
                self.Par = {'alpha':pars[1]}
                self.Dist = stats.rayleigh(scale=self.Par['alpha'])
            
            
            elif name.lower() == "truncatednormal":
                X = val[0]
                if val[1] >= val[2]:
                    raise RuntimeError("The upper bound a must be larger than the lower bound b.")
                if not (all(X >= val[1]) and all(X <= val[2])):
                    raise RuntimeError("The given samples must be in the range [a,b].")
                def equation(par):
                    return (-np.sum(np.log(stats.norm.pdf(X,loc=par[0], scale=par[1])/
                                           (stats.norm.cdf(val[2],par[0],par[1])-stats.norm.cdf(val[1],par[0],par[1])))))
                    
                x0 = np.array([np.mean(X),np.std(X)])
                bnds = optimize.Bounds(lb=[-np.inf,0],ub=[np.inf,np.inf])
                sol = optimize.minimize(equation, x0, bounds=bnds)
                if sol.success == True:                
                    self.Par = {'mu_n':float(sol.x[0]), 'sig_n':float(sol.x[1]), 'a':val[1], 'b':val[2]}
                    a_mod = (self.Par['a']-self.Par['mu_n'])/self.Par['sig_n']
                    b_mod = (self.Par['b']-self.Par['mu_n'])/self.Par['sig_n']
                    self.Dist =  stats.truncnorm(loc=self.Par['mu_n'],scale=self.Par['sig_n'], a=a_mod, b=b_mod)
                else:
                    raise RuntimeError("Maximum likelihood estimation did not converge.")
                
                
            elif name.lower() == "uniform":
                self.Par = {'lower':min(val), 'upper': max(val)}
                self.Dist = stats.uniform(loc=self.Par['lower'], scale=self.Par['upper'] - self.Par['lower'])
            
            elif name.lower() == "weibull":
                pars = stats.weibull_min.fit(val, floc=0)
                self.Par = {'a_n':pars[2], 'k':pars[0]}
                self.Dist = stats.weibull_min(c=self.Par['k'], scale=self.Par['a_n'])

            
            else:
                raise RuntimeError("Distribution type '" + name + "' not available.")

        else:
            raise RuntimeError("Unknown option :" + opt)           
            
#%%            
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        
        if self.Name == "gevmin":
            return -self.Dist.mean()
        
        elif self.Name == "negativebinomial":
            return self.Dist.mean() + self.Par['k']

        else:
            return self.Dist.mean()

#%%           
    def std(self):
        """
        Returns the standard deviation of the distribution.
        """
        
        return self.Dist.std()

#%%       
    def pdf(self, x):
        """
        Returns the PDF value.
        """
        
        if self.Name == "binomial":
            return self.Dist.pmf(x)

        elif self.Name == "geometric":
            return self.Dist.pmf(x)
        
        elif self.Name == "gevmin":
            return self.Dist.pdf(-x)

        elif self.Name == "negativebinomial":
            return self.Dist.pmf(x - self.Par['k'])

        elif self.Name == "poisson":
            return self.Dist.pmf(x)

        else:
            return self.Dist.pdf(x)
        
#%%            
    def cdf(self, x):
        """
        Returns the CDF value.
        """
          
        if self.Name == "gevmin":
            return 1-self.Dist.cdf(-x)  # <-- this is not a proper cdf !
        
        elif self.Name == "negativebinomial":
            return self.Dist.cdf(x - self.Par['k'])

        else:
            return self.Dist.cdf(x)

#%%            
    def random(self, size=None):
        """
        Generates random samples according to the distribution of the
        object.
        """

        if self.Name == "gevmin":
            return self.Dist.rvs(size=size) * (-1)
        
        elif self.Name == "negativebinomial":
            samples = self.Dist.rvs(size=size) + self.Par['k']
            return samples

        else:
            samples = self.Dist.rvs(size=size)
            return samples

#%%           
    def icdf(self, y):
        """
        Returns the value of the inverse CDF.
        """
        
        if self.Name == "gevmin":
            return -self.Dist.ppf(1-y)
        
        elif self.Name == "negativebinomial":
            return self.Dist.ppf(y) + self.Par['k']

        else:
            return self.Dist.ppf(y)
        
#%% Nested functions: for GEV-parameter fitting
    
def gevfit_alt(y):
    '''Author: Iason Papaioannou
    The function gevfit_alt evaluates the parameters of the generalized
    extreme value distribution with the method of Probability Weighted
    Moments (PWM) and Maximum Likelihood Estimation (MLE).'''

    # compute PWM estimates
    x01 = gevpwm(y)
    
    if x01[0] > 0:
        # Compute mle estimates using PWM estimates as starting points
        x02 =  stats.genextreme.fit(y,scale=x01[1], loc=x01[2])
        x02 = np.array([-x02[0],x02[2],x02[1]])
        # if alpha reasonable
        if x02[1] >= 1.e-6:
            # set parameters
            par = x02
            if par[0] < -1:
                par = x01
                warnings.warn('The MLE estimate of the shape parameter of the GEV is not in the range where the MLE estimator is valid. PWM estimation is used.')
                if par[0] > 0.4:   
                    warnings.warn('The shape parameter of the GEV is not in the range where PWM asymptotic results are valid.')       
        else:       
            # set parameters obtained by PWM
            par = x01       
            if par[0] > 0.4:
                warnings.warn('The shape parameter of the GEV is not in the range where PWM asymptotic results are valid.')
    else:  
        # set parameters obtained by PWM
        par = x01
        if par[0] < -0.4:
            warnings.warn('The shape parameter of the GEV is not in the range where PWM asymptotic results are valid.')
    
    return par
        

#------------------------------------------------------------------------------
        
def gevpwm(y):
    '''Author: Iason Papaioannou
    The function gevpwm evaluates the parameters of the generalized
    extreme value distribution applying the method of Probability Weighted
    Moments.'''

    # compute PWM estimates
    y2 = np.sort(y)
    beta0 = np.mean(y)
    
    p1 = np.arange(len(y))/(len(y)-1)
    p2 = p1*(np.arange(len(y))-1)/(len(y)-2)
    beta1 = p1@y2
    beta2 = p2@y2
    
    beta1 = beta1/len(y)
    beta2 = beta2/len(y)
    
    c = (2*beta1-beta0)/(3*beta2-beta0)-np.log(2)/np.log(3)
    par0 = -7.8590*c -2.9554*c**2
    equation = lambda x: (3*beta2-beta0)/(2*beta1-beta0)-(1-3**x)/(1-2**x)
    sol = optimize.fsolve(equation, x0=par0, full_output=True)
    sol = optimize.fsolve(equation, x0=0.02, full_output=True)
    if sol[2] == 1:
        par = np.zeros(3)
        par[0] = sol[0][0]
        par[1] = -(2*beta1-beta0)*par[0]/special.gamma(1-par[0])/(1-2**par[0])
        par[2] = beta0 - par[1]/par[0]*(special.gamma(1-par[0])-1)
    else:
        warnings.warn("fsolve could not converge to a solution for the PWM estimate.")
    

    return par