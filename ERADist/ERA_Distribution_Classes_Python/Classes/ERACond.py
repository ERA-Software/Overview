# import of modules
import numpy as np
from scipy import optimize, stats, special, integrate
import types
import inspect

'''
---------------------------------------------------------------------------
Generation of conditional distribution objects for the use within the
ERARosen class.
---------------------------------------------------------------------------
Developed by:
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Luca Sardi
Nicola Bronzetti
Alexander von Ramm
Matthias Willer
Peter Kaplan

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Version 2022-01
* Change of the definition of the parameter/moment functions.
Allows the use of user-defined variables and functions within the given
function handles. Also allows the definition of functions in matrix-vector 
form.

First Release, 2021-03 
--------------------------------------------------------------------------
This software generates conditional distribution objects according to the
parameters and definitions used in the distribution table of the ERA Group 
of TUM.
They can be defined either by their parameters or the first and second
moment.
The class is meant to be an auxiliary class for the ERARosen class.
---------------------------------------------------------------------------
References:
1. Documentation of the ERA Distribution Classes
---------------------------------------------------------------------------
'''
#%%
class ERACond(object):
    """
    Generation of conditional distribution objects for the use within the
    ERARosen class.
    
    Construction of the conditional distribution object with
    
            Obj = ERACond(name,opt,param) 
    or      Obj = ERACond(name,opt,param,id)
    
    The available distributions, represented by the input variable 'name', 
    are the same as in the ERADist class (see below). They can be described 
    either by parameters (opt='PAR') or by the first and second moment
    (opt='MOM'). 
    
    The parameters or moments must be given as a lambda function. Examples
    for lambda functions given by the input 'param' of a two parametric
    distribution depending on two other random variables could be:       
                    
      param = lambda x,y: [x+y,0.2*x^2],     param = lambda a,b: [3*a-2*b,4]
    
    The input 'id' can be used to better identify the different variables
    (nodes) when plotting the graph describing the dependency between the
    different variables in the ERARosen class (method plotGraph). The input
    'id' is however not mandatory.
    
    
    The following distribution types are available:
    
    opt = "PAR", if you want to specify the distibution by its parameters:
      Beta:                       Obj = ERADist('beta','PAR',lambda ... :[r,s,a,b])
      Binomial:                   Obj = ERADist('binomial','PAR',lambda ... :[n,p])
      Chi-squared:                Obj = ERADist('chisquare','PAR',lambda ... :[k])
      Exponential:                Obj = ERADist('exponential','PAR',lambda ... :[lambda])
      Frechet:                    Obj = ERADist('frechet','PAR',lambda ... :[a_n,k])
      Gamma:                      Obj = ERADist('gamma','PAR',lambda ... :[lambda,k])
      Geometric:                  Obj = ERADist('geometric','PAR',lambda ... :[p])
      GEV (to model maxima):      Obj = ERADist('GEV','PAR',lambda ... :[beta,alpha,epsilon])
      GEV (to model minima):      Obj = ERADist('GEVMin','PAR',lambda ... :[beta,alpha,epsilon])
      Gumbel (to model maxima):   Obj = ERADist('gumbel','PAR',lambda ... :[a_n,b_n])
      Gumbel (to model minima):   Obj = ERADist('gumbelMin','PAR',lambda ... :[a_n,b_n])
      Log-normal:                 Obj = ERADist('lognormal','PAR',lambda ... :[mu_lnx,sig_lnx])
      Negative binomial:          Obj = ERADist('negativebinomial','PAR',lambda ... :[k,p])
      Normal:                     Obj = ERADist('normal','PAR',lambda ... :[mean,std])
      Pareto:                     Obj = ERADist('pareto','PAR',lambda ... :[x_m,alpha])
      Poisson:                    Obj = ERADist('poisson','PAR',lambda ... :[v,t])
                              or  Obj = ERADist('poisson','PAR',lambda ... :[lambda])
      Rayleigh:                   Obj = ERADist('rayleigh','PAR',lambda ... :[alpha])
      Truncated normal:           Obj = ERADist('truncatednormal','PAR',lambda ... :[mu_N,sig_N,a,b])
      Uniform:                    Obj = ERADist('uniform','PAR',lambda ... :[lower,upper])
      Weibull:                    Obj = ERADist('weibull','PAR',lambda ... :[a_n,k]) 
    
    
    opt = "MOM", if you want to specify the distibution by its moments:
      Beta:                       Obj = ERADist('beta','MOM',lambda ... :[mean,std,a,b])
      Binomial:                   Obj = ERADist('binomial','MOM',lambda ... :[mean,std])
      Chi-squared:                Obj = ERADist('chisquare','MOM',lambda ... :[mean])
      Exponential:                Obj = ERADist('exponential','MOM',lambda ... :[mean])
      Frechet:                    Obj = ERADist('frechet','MOM',lambda ... :[mean,std])
      Gamma:                      Obj = ERADist('gamma','MOM',lambda ... :[mean,std])
      Geometric:                  Obj = ERADist('geometric','MOM',lambda ... :[mean])
      GEV (to model maxima):      Obj = ERADist('GEV','MOM',lambda ... :[mean,std,beta])
      GEV (to model minima):      Obj = ERADist('GEVMin','MOM',lambda ... :[mean,std,beta])
      Gumbel (to model maxima):   Obj = ERADist('gumbel','MOM',lambda ... :[mean,std])
      Gumbel (to model minima):   Obj = ERADist('gumbelMin','MOM',lambda ... :[mean,std])
      Log-normal:                 Obj = ERADist('lognormal','MOM',lambda ... :[mean,std])
      Negative binomial:          Obj = ERADist('negativebinomial','MOM',lambda ... :[mean,std])
      Normal:                     Obj = ERADist('normal','MOM',lambda ... :[mean,std])
      Pareto:                     Obj = ERADist('pareto','MOM',lambda ... :[mean,std])
      Poisson:                    Obj = ERADist('poisson','MOM',lambda ... :[mean,t]
                              or  Obj = ERADist('poisson','MOM',lambda ... :[mean])
      Rayleigh:                   Obj = ERADist('rayleigh','MOM',lambda ... :[mean])
      Truncated normal:           Obj = ERADist('truncatednormal','MOM',lambda ... :[mean,std,a,b])
      Uniform:                    Obj = ERADist('uniform','MOM',lambda ... :[mean,std])
      Weibull:                    Obj = ERADist('weibull','MOM',lambda ... :[mean,std])
      
      """
    
    def __init__(self, name, opt, param, ID=False):
        """
        Constructor method, for more details have a look at the
        class description.
        """
        
        self.Name = name.lower()
        
        if opt.upper() == "PAR" or opt.upper() == "MOM":
            self.Opt = opt.upper()
        else:
            raise RuntimeError("Conditional distributions can only be defined "
                               "by moments (opt = 'MOM') or by parameters (opt = 'PAR').")
            
        self.ID = ID
        
        # check if param is a lambda function
        if type(param) == types.LambdaType:
            self.Param = param
        else:
            raise RuntimeError("The input param must be a lambda function.")

        self.modParam = param          
            
#%%           
    def condParam(self,cond):
        """
        Evaluates the parameters of the distribution for the
        different given conditions.
        In case that the distribution is described by its moments, 
        the evaluated moments are used to obtain the distribution 
        parameters. 
        This method is used by the ERACond methods condCDF, condPDF,
        condiCDF and condRandom.
        """
        
        cond = np.array(cond, ndmin=2, dtype=float).T
        par = self.modParam(cond)
        n_cond = np.shape(cond)[0]
        
        #----------------------------------------------------------------------------
        # for the case of Opt == PAR
        if self.Opt == "PAR":
            if self.Name == "beta":
                Par = [par[0], par[1], par[2], par[3]-par[2]]                
            elif self.Name == "binomial":
                Par = [par[0].astype(int), par[1]]
            elif self.Name == "chisquare":
                Par = np.around(par,0)
            elif self.Name == "exponential":
                Par = 1/par
            elif self.Name == "frechet":
                Par = [-1/par[1], par[0]/par[1], par[0]]
            elif self.Name == "gamma":
                Par = [par[1], 1/par[0]]
            elif self.Name == "geometric":
                Par = par
            elif self.Name == "gev":
                Par = [-par[0], par[1], par[2]]
            elif self.Name == "gevmin":
                Par = [-par[0], par[1], -par[2]]
            elif self.Name == "gumbel":
                Par = par
            elif self.Name == "gumbelmin":
                Par = par
            elif self.Name == "lognormal":
                Par = [par[1],np.exp(par[0])]
            elif self.Name == "negativebinomial":
                Par = par
            elif self.Name == "normal":
                Par = par
            elif self.Name == "pareto":
                Par = [1/par[1], par[0]/par[1], par[0]]
            elif self.Name == "poisson":
                if isinstance(par, list):
                    Par = par[0]*par[1]
                else:
                    Par = par                
            elif self.Name == "rayleigh":
                Par = par
            elif self.Name == "truncatednormal":
                a = (par[2]-par[0])/par[1]
                b = (par[3]-par[0])/par[1]
                Par = [par[0], par[1], a, b]                
            elif self.Name == "uniform":
                Par = [par[0], par[1]-par[0]]                
            elif self.Name == "weibull":
                Par = par
            
        #----------------------------------------------------------------------------       
        # for the case of Opt == MOM
        else:                    
            if self.Name == "beta":
                r = ((par[3]-par[0])*(par[0]-par[2])/par[1]**2-1)*(par[0]-par[2])/(par[3]-par[2])
                s = r*(par[3]-par[0])/(par[0]-par[2])
                Par = [r, s, par[2], par[3]-par[2]]
            elif self.Name == "binomial":
                p = 1 - (par[1]) ** 2 / par[0]
                n = par[0] / p
                Par = [n.astype(int), p]
            elif self.Name == "chisquare":
                Par = np.around(par,0)                
            elif self.Name == "exponential":
                Par = par                            
            elif self.Name == "frechet":
                c = np.zeros(n_cond)
                scale = np.zeros(n_cond)
                loc = np.zeros(n_cond)               
                for i in range(n_cond):
                    param0 = 2.0001
                    def equation(param):
                        return (np.sqrt(special.gamma(1 - 2 / param)- special.gamma(1 - 1 / param) ** 2)
                                / special.gamma(1 - 1 / param)- par[1][i] / par[0][i])
                    sol = optimize.fsolve(equation, x0=param0, full_output=True)
                    if sol[2] == 1:
                        k = sol[0][0]
                        a_n = par[0][i] / special.gamma(1 - 1 / k)
                        c[i] = -1/k
                        scale[i] = a_n/k
                        loc[i] = a_n
                    else:
                        c[i] = np.nan
                        scale[i] = np.nan
                        loc[i] = np.nan
                Par = [c, scale, loc]               
            elif self.Name == "gamma":
                Par = [(par[0]/par[1])**2, par[1]**2/par[0]]
            elif self.Name == "geometric":
                Par = 1/par
            elif self.Name == "gev":
                beta = par[2]
                alpha = abs(beta)*par[1]/np.sqrt(special.gamma(1-2*beta)-special.gamma(1-beta)**2)
                epsilon = par[0]-(alpha/beta*(special.gamma(1-beta)-1))
                Par = [-beta, alpha, epsilon]
            elif self.Name == "gevmin":
                beta = par[2]
                alpha = abs(beta)*par[1]/np.sqrt(special.gamma(1-2*beta)-special.gamma(1-beta)**2)
                epsilon = par[0]+(alpha/beta*(special.gamma(1-beta)-1))
                Par = [-beta, alpha, -epsilon]
            elif self.Name == "gumbel":
                a_n = par[1] * np.sqrt(6)/np.pi
                b_n = par[0] - np.euler_gamma * a_n
                Par = [a_n, b_n]
            elif self.Name == "gumbelmin":
                a_n = par[1] * np.sqrt(6) / np.pi
                b_n = par[0] + np.euler_gamma * a_n
                Par = [a_n, b_n]
            elif self.Name == "lognormal":
                mu_lnx = np.log(par[0] ** 2 / np.sqrt(par[1] ** 2 + par[0] ** 2))
                sig_lnx = np.sqrt(np.log(1 + (par[1] / par[0]) ** 2))
                Par = [sig_lnx, np.exp(mu_lnx)]
            elif self.Name == "negativebinomial":
                p = par[0] / (par[0] + par[1] ** 2)
                k = par[0] * p
                Par = [k, p]
            elif self.Name == "normal":
                Par = par
            elif self.Name == "pareto":
                alpha = 1 + np.sqrt(1 + (par[0] / par[1]) ** 2)
                x_m = par[0] * (alpha - 1) / alpha
                Par = [1/alpha, x_m/alpha, x_m]
            elif self.Name == "poisson":
                if isinstance(par, list):
                    Par = par[0]
                else:
                    Par = par
            elif self.Name == "rayleigh":
                Par = par / np.sqrt(np.pi / 2)
            elif self.Name == "truncatednormal":
                mu = np.zeros(n_cond)
                sig = np.zeros(n_cond)
                a = par[2]
                b = par[3]
                for i in range(n_cond):
                    mean = par[0][i]; std = par[1][i];
                    if a[i] >= b[i] or mean <= a[i] or mean >= b[i]:
                        a[i] = np.nan; b[i] = np.nan; mu[i] = np.nan; sig[i] = np.nan;
                        continue
                    def equation(param):
                        f = lambda x: stats.norm.pdf(x,param[0],param[1])/(stats.norm.cdf(b[i],param[0],param[1])-stats.norm.cdf(a[i],param[0],param[1]))
                        expec_eq = integrate.quadrature(lambda x: x*f(x),a[i],b[i])[0]-mean
                        std_eq = np.sqrt(integrate.quadrature(lambda x: x**2*f(x),a[i],b[i])[0]-(integrate.quadrature(lambda x: x*f(x),a[i],b[i]))[0]**2)-std
                        eq = [expec_eq, std_eq]
                        return(eq)                           
                    x0 = [mean, std]
                    sol = optimize.fsolve(equation, x0=x0, full_output=True)
                    if sol[2] == 1:
                        mu[i] = sol[0][0]; sig[i] = sol[0][1];
                    else:
                        a[i] = np.nan; b[i] = np.nan; mu[i] = np.nan; sig[i] = np.nan;
                Par = [mu, sig, (a-mu)/sig, (b-mu)/sig]                
            elif self.Name == "uniform":
                lower = par[0] - np.sqrt(12) * par[1] / 2
                upper = par[0] + np.sqrt(12) * par[1] / 2
                Par = [lower, upper-lower]
            elif self.Name == "weibull":
                a_n = np.zeros(n_cond)
                k = np.zeros(n_cond)
                for i in range(n_cond):
                    def equation(param):
                        return (np.sqrt(special.gamma(1 + 2 / param) - (special.gamma(1 + 1 / param)) ** 2)
                                / special.gamma(1 + 1 / param) - par[1][i] / par[0][i])
                    
                    sol = optimize.fsolve(equation, x0=0.02, full_output=True)
                    if sol[2] == 1:
                        k[i] = sol[0][0]
                        a_n[i] = par[0][i] / special.gamma(1 + 1 / k[i])
                    else:
                        k[i] = np.nan; a_n[i] = np.nan;
                Par = [a_n, k]
        
        for i in range(0,len(Par)):
            Par[i] = np.squeeze(Par[i])
            
        return Par        
        
#%%
    def condCDF(self,x,cond):
        """
        Evaluates the CDF of the conditional distribution at x for
        the given conditions.
        This method is used by the ERARosen method X2U.
        """
        
        par = self.condParam(cond) #computation of the conditional parameters
        x = np.array(x, ndmin=1, dtype=float)
        
        if self.Name == "beta":
            CDF = stats.beta.cdf(x, a=par[0], b=par[1], loc=par[2], scale=par[3])
        elif self.Name == "binomial":
            CDF = stats.binom.cdf(x, n=par[0], p=par[1])
        elif self.Name == "chisquare":
            CDF = stats.chi2.cdf(x,df=par)
        elif self.Name == "exponential":
            CDF = stats.expon.cdf(x, scale=par)
        elif self.Name == "frechet":
            CDF = stats.genextreme.cdf(x, c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "gamma":
            CDF = stats.gamma.cdf(x, a=par[0], scale=par[1])
        elif self.Name == "geometric":
            CDF = stats.geom.cdf(x, p=par)
        elif self.Name == "gev":
            CDF = stats.genextreme.cdf(x, c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "gevmin":
            CDF = 1-stats.genextreme.cdf(-x, c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "gumbel":
            CDF = stats.gumbel_r.cdf(x, scale=par[0], loc=par[1])
        elif self.Name == "gumbelmin":
            CDF = stats.gumbel_l.cdf(x, scale=par[0], loc=par[1])
        elif self.Name == "lognormal":
            CDF = stats.lognorm.cdf(x, s=par[0], scale=par[1])
        elif self.Name == "negativebinomial":
            CDF = stats.nbinom.cdf(x-par[0], n=par[0], p=par[1])
        elif self.Name == "normal":
            CDF = stats.norm.cdf(x, loc=par[0], scale=par[1])
        elif self.Name == "pareto":
            CDF = stats.genpareto.cdf(x, c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "poisson":
            CDF = stats.poisson.cdf(x, mu=par)
        elif self.Name == "rayleigh":
            CDF = stats.rayleigh.cdf(x, scale=par)
        elif self.Name == "truncatednormal":
            CDF =  stats.truncnorm.cdf(x, loc=par[0], scale=par[1], a=par[2], b=par[3])
        elif self.Name == "uniform":
            CDF = stats.uniform.cdf(x, loc=par[0], scale=par[1])
        elif self.Name == "weibull":
            CDF = stats.weibull_min.cdf(x, c=par[1], scale=par[0])
     
        return CDF
    
#%%
    def condiCDF(self,y,cond):
        """
        Evaluates the inverse CDF of the conditional distribution at
        y for the given conditions.
        This method is used by the ERARosen method U2X.
        """
        
        par = self.condParam(cond) #computation of the conditional parameters
        y = np.array(y, ndmin=1, dtype=float)
        
        if self.Name == "beta":
            iCDF = stats.beta.ppf(y, a=par[0], b=par[1], loc=par[2], scale=par[3])
        elif self.Name == "binomial":
            iCDF = stats.binom.ppf(y, n=par[0], p=par[1])
        elif self.Name == "chisquare":
            iCDF = stats.chi2.ppf(y,df=par)
        elif self.Name == "exponential":
            iCDF = stats.expon.ppf(y, scale=par)
        elif self.Name == "frechet":
            iCDF = stats.genextreme.ppf(y, c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "gamma":
            iCDF = stats.gamma.ppf(y, a=par[0], scale=par[1])
        elif self.Name == "geometric":
            iCDF = stats.geom.ppf(y, p=par)
        elif self.Name == "gev":
            iCDF = stats.genextreme.ppf(y, c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "gevmin":
            iCDF = -stats.genextreme.ppf(1-y, c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "gumbel":
            iCDF = stats.gumbel_r.ppf(y, scale=par[0], loc=par[1])
        elif self.Name == "gumbelmin":
            iCDF = stats.gumbel_l.ppf(y, scale=par[0], loc=par[1])
        elif self.Name == "lognormal":
            iCDF = stats.lognorm.ppf(y, s=par[0], scale=par[1])
        elif self.Name == "negativebinomial":
            iCDF = stats.nbinom.ppf(y, n=par[0], p=par[1])+par[0]
        elif self.Name == "normal":
            iCDF = stats.norm.ppf(y, loc=par[0], scale=par[1])
        elif self.Name == "pareto":
            iCDF = stats.genpareto.ppf(y, c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "poisson":
            iCDF = stats.poisson.ppf(y, mu=par)
        elif self.Name == "rayleigh":
            iCDF = stats.rayleigh.ppf(y, scale=par)
        elif self.Name == "truncatednormal":
            iCDF =  stats.truncnorm.ppf(y, loc=par[0], scale=par[1], a=par[2], b=par[3])
        elif self.Name == "uniform":
            iCDF = stats.uniform.ppf(y, loc=par[0], scale=par[1])
        elif self.Name == "weibull":
            iCDF = stats.weibull_min.ppf(y, c=par[1], scale=par[0])

        return iCDF
    
#%%
    def condPDF(self,x,cond):
        """
        Evaluates the PDF of the conditional distribution at x for
        the given conditions.
        This method is used by the ERARosen method pdf.
        """
        
        par = self.condParam(cond) #computation of the conditional parameters
        x = np.array(x, ndmin=1, dtype=float)
        
        if self.Name == "beta":
            PDF = stats.beta.pdf(x, a=par[0], b=par[1], loc=par[2], scale=par[3])
        elif self.Name == "binomial":
            PDF = stats.binom.pmf(x, n=par[0], p=par[1])
        elif self.Name == "chisquare":
            PDF = stats.chi2.pdf(x,df=par)
        elif self.Name == "exponential":
            PDF = stats.expon.pdf(x, scale=par)
        elif self.Name == "frechet":
            PDF = stats.genextreme.pdf(x, c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "gamma":
            PDF = stats.gamma.pdf(x, a=par[0], scale=par[1])
        elif self.Name == "geometric":
            PDF = stats.geom.pmf(x, p=par)
        elif self.Name == "gev":
            PDF = stats.genextreme.pdf(x, c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "gevmin":
            PDF = stats.genextreme.pdf(-x, c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "gumbel":
            PDF = stats.gumbel_r.pdf(x, scale=par[0], loc=par[1])
        elif self.Name == "gumbelmin":
            PDF = stats.gumbel_l.pdf(x, scale=par[0], loc=par[1])
        elif self.Name == "lognormal":
            PDF = stats.lognorm.pdf(x, s=par[0], scale=par[1])
        elif self.Name == "negativebinomial":
            PDF = stats.nbinom.pmf(x-par[0], n=par[0], p=par[1])
        elif self.Name == "normal":
            PDF = stats.norm.pdf(x, loc=par[0], scale=par[1])
        elif self.Name == "pareto":
            PDF = stats.genpareto.pdf(x, c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "poisson":
            PDF = stats.poisson.pmf(x, mu=par)
        elif self.Name == "rayleigh":
            PDF = stats.rayleigh.pdf(x, scale=par)
        elif self.Name == "truncatednormal":
            PDF =  stats.truncnorm.pdf(x, loc=par[0], scale=par[1], a=par[2], b=par[3])
        elif self.Name == "uniform":
            PDF = stats.uniform.pdf(x, loc=par[0], scale=par[1])
        elif self.Name == "weibull":
            PDF = stats.weibull_min.pdf(x, c=par[1], scale=par[0])
            
        return PDF
    
#%%
    def condRandom(self,cond):
        """
        Creates one random sample for each given condition.
        This method is used by the ERARosen method random.
        """
        
        par = self.condParam(cond) #computation of the conditional parameters

        if self.Name == "beta":
            Random = stats.beta.rvs(a=par[0], b=par[1], loc=par[2], scale=par[3])
        elif self.Name == "binomial":
            Random = stats.binom.rvs(n=par[0], p=par[1])
        elif self.Name == "chisquare":
            Random = stats.chi2.rvs(df=par)
        elif self.Name == "exponential":
            Random = stats.expon.rvs(scale=par)
        elif self.Name == "frechet":
            Random = stats.genextreme.rvs(c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "gamma":
            Random = stats.gamma.rvs(a=par[0], scale=par[1])
        elif self.Name == "geometric":
            Random = stats.geom.rvs(p=par)
        elif self.Name == "gev":
            Random = stats.genextreme.rvs(c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "gevmin":
            Random = -stats.genextreme.rvs(c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "gumbel":
            Random = stats.gumbel_r.rvs(scale=par[0], loc=par[1])
        elif self.Name == "gumbelmin":
            Random = stats.gumbel_l.rvs(scale=par[0], loc=par[1])
        elif self.Name == "lognormal":
            Random = stats.lognorm.rvs(s=par[0], scale=par[1])
        elif self.Name == "negativebinomial":
            Random = stats.nbinom.rvs(n=par[0], p=par[1])+par[0]
        elif self.Name == "normal":
            Random = stats.norm.rvs(loc=par[0], scale=par[1])
        elif self.Name == "pareto":
            Random = stats.genpareto.rvs(c=par[0], scale=par[1], loc=par[2])
        elif self.Name == "poisson":
            Random = stats.poisson.rvs(mu=par)
        elif self.Name == "rayleigh":
            Random = stats.rayleigh.rvs(scale=par)
        elif self.Name == "truncatednormal":
            Random =  stats.truncnorm.rvs(loc=par[0], scale=par[1], a=par[2], b=par[3])
        elif self.Name == "uniform":
            Random = stats.uniform.rvs(loc=par[0], scale=par[1])
        elif self.Name == "weibull":
            Random = stats.weibull_min.rvs(c=par[1], scale=par[0])
        
        return Random
     