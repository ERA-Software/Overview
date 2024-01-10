
classdef ERADist
    
% Generation of marginal distribution objects.
% Construction of the distribution object with 
%   
%         Obj = ERADist(name,opt,val)
% or      Obj = ERADist(name,opt,val,ID)
%
% The second option is only useful when using the ERADist object within
% the scope of an ERARosen object.
%
%
% The following distribution types are available:
%
% opt = "PAR", specification of the distribution by its parameters:
%   Beta:                       Obj = ERADist('beta','PAR',[r,s,a,b])
%   Binomial:                   Obj = ERADist('binomial','PAR',[n,p])
%   Chi-squared:                Obj = ERADist('chisquare','PAR',[k])
%   Exponential:                Obj = ERADist('exponential','PAR',[lambda])
%   Frechet:                    Obj = ERADist('frechet','PAR',[a_n,k])
%   Gamma:                      Obj = ERADist('gamma','PAR',[lambda,k])
%   Geometric:                  Obj = ERADist('geometric','PAR',[p])
%   GEV (to model maxima):      Obj = ERADist('GEV','PAR',[beta,alpha,epsilon])
%   GEV (to model minima):      Obj = ERADist('GEVMin','PAR',[beta,alpha,epsilon])
%   Gumbel (to model maxima):   Obj = ERADist('gumbel','PAR',[a_n,b_n])
%   Gumbel (to model minima):   Obj = ERADist('gumbelMin','PAR',[a_n,b_n])
%   Log-normal:                 Obj = ERADist('lognormal','PAR',[mu_lnx,sig_lnx])
%   Negative binomial:          Obj = ERADist('negativebinomial','PAR',[k,p])
%   Normal:                     Obj = ERADist('normal','PAR',[mean,std])
%   Pareto:                     Obj = ERADist('pareto','PAR',[x_m,alpha])
%   Poisson:                    Obj = ERADist('poisson','PAR',[v,t])
%                           or  Obj = ERADist('poisson','PAR',[lambda])
%   Rayleigh:                   Obj = ERADist('rayleigh','PAR',[alpha])
%   Standard normal:            Obj = ERADist('standardnormal','PAR',[])
%   Truncated normal:           Obj = ERADist('truncatednormal','PAR',[mu_n,sig_n,a,b])
%   Uniform:                    Obj = ERADist('uniform','PAR',[lower,upper])
%   Weibull:                    Obj = ERADist('weibull','PAR',[a_n,k]) 
%
%
% opt = "MOM", specification of the distribution by its moments:
%   Beta:                       Obj = ERADist('beta','MOM',[mean,std,a,b])
%   Binomial:                   Obj = ERADist('binomial','MOM',[mean,std])
%   Chi-squared:                Obj = ERADist('chisquare','MOM',[mean])
%   Exponential:                Obj = ERADist('exponential','MOM',[mean])
%   Frechet:                    Obj = ERADist('frechet','MOM',[mean,std])
%   Gamma:                      Obj = ERADist('gamma','MOM',[mean,std])
%   Geometric:                  Obj = ERADist('geometric','MOM',[mean])
%   GEV (to model maxima):      Obj = ERADist('GEV','MOM',[mean,std,beta])
%   GEV (to model minima):      Obj = ERADist('GEVMin','MOM',[mean,std,beta])
%   Gumbel (to model maxima):   Obj = ERADist('gumbel','MOM',[mean,std])
%   Gumbel (to model minima):   Obj = ERADist('gumbelMin','MOM',[mean,std])
%   Log-normal:                 Obj = ERADist('lognormal','MOM',[mean,std])
%   Negative binomial:          Obj = ERADist('negativebinomial','MOM',[mean,std])
%   Normal:                     Obj = ERADist('normal','MOM',[mean,std])
%   Pareto:                     Obj = ERADist('pareto','MOM',[mean,std])
%   Poisson:                    Obj = ERADist('poisson','MOM',[mean,t])
%                           or  Obj = ERADist('poisson','MOM',[mean])
%   Rayleigh:                   Obj = ERADist('rayleigh','MOM',[mean])
%   Standard normal:            Obj = ERADist('standardnormal','MOM',[])
%   Truncated normal:           Obj = ERADist('truncatednormal','MOM',[mean,std,a,b])
%   Uniform:                    Obj = ERADist('uniform','MOM',[mean,std])
%   Weibull:                    Obj = ERADist('weibull','MOM',[mean,std])
%
%
% opt = "DATA", specification of the distribution by data given as a vector:
%   Beta:                       Obj = ERADist('beta','DATA',{[X],[a,b]})
%   Binomial:                   Obj = ERADist('binomial','DATA',{[X],n})
%   Chi-squared:                Obj = ERADist('chisquare','DATA',[X])
%   Exponential:                Obj = ERADist('exponential','DATA',[X])
%   Frechet:                    Obj = ERADist('frechet','DATA',[X])
%   Gamma:                      Obj = ERADist('gamma','DATA',[X])
%   Geometric:                  Obj = ERADist('geometric','DATA',[X])
%   GEV (to model maxima):      Obj = ERADist('GEV','DATA',[X])
%   GEV (to model minima):      Obj = ERADist('GEVMin','DATA',[X])
%   Gumbel (to model maxima):   Obj = ERADist('gumbel','DATA',[X])
%   Gumbel (to model minima):   Obj = ERADist('gumbelMin','DATA',[X])
%   Log-normal:                 Obj = ERADist('lognormal','DATA',[X])
%   Negative binomial:          Obj = ERADist('negativebinomial','DATA',[X])
%   Normal:                     Obj = ERADist('normal','DATA',[X])
%   Pareto:                     Obj = ERADist('pareto','DATA',[X])
%   Poisson:                    Obj = ERADist('poisson','DATA',{[X],t})
%                           or  Obj = ERADist('poisson','DATA',[X])
%   Rayleigh:                   Obj = ERADist('rayleigh','DATA',[X])
%   Truncated normal:           Obj = ERADist('truncatednormal','DATA',{[X],[a,b]})
%   Uniform:                    Obj = ERADist('uniform','DATA',[X])
%   Weibull:                    Obj = ERADist('weibull','DATA',[X])

%{
---------------------------------------------------------------------------
Developed by:
Antonios Kamariotis (antonis.kamariotis@tum.de)
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
---------------------------------------------------------------------------
Version 2021-03:
*   Correction of the bounds of the beta distribution in the case DATA

Version 2020-10:
*   Until now every ERADist class contained a distribution object created
    with the MATLAB built-in function makedist as a property. Everytime a
    method of the ERADist class was called, the ERADist class referred to
    the MATLAB distribution object to generate the requested result.
    To increase the performance of the ERADist class from now on the
    creation of the MATLAB distribution object is omitted. The requested
    results are instead obtained by the use of distribution specific MATLAB
    built-in functions or self-implemented solution procedures.
 
*   Implementation of all the missing distribution types for the option
    'DATA'. Now all distribution types are available for all three options
    'MOM','PAR' and 'DATA'.

*   Introduction of the truncated normal distribution.
    
%--------------------------------------------------------------------------
This software generates marginal distribution objects according to the 
parameters and definitions used in the distribution table of the ERA Group 
of TUM. They can be defined either by their parameters, the first and 
second moment or by data, given as a vector.
---------------------------------------------------------------------------
References:
1. Documentation of the ERA Distribution Classes
---------------------------------------------------------------------------
%}    
    
    %% MATLAB class: definition of the 'properties' block
    
    properties
        Name % type of distribution
        Par  % parameters of the distribution
        ID   % name by which the distribution can be identified
    end
        
    
    %% MATLAB class: definition of the 'methods' block  

    methods
        
        function Obj = ERADist(name,opt,val,id)
            Obj.Name = lower(name);

            % check if ID of the random variable is given
            if nargin == 3
                Obj.ID = [];
            elseif nargin == 4
                if ~ischar(id)
                    error('id must be given as a character array.')
                end
                Obj.ID = id;
            elseif nargin == 2 && (strcmpi(name,'standardnormal') || strcmpi(name,'standardgaussian'))
                val=[0,1];
            elseif nargin < 3
                error('Not enough input arguments.')
            else
                error('To many input arguments.')
            end
            
            switch upper(opt)
                
                %------------------------------------------------------------------------------------------------------------------
                % If PAR is chosen, the validity of the given parameters is
                % checked.
                
                case 'PAR'
                    Obj.Par = val;
                    switch lower(Obj.Name)
                        
                        case 'beta'
                            if (Obj.Par(1) > 0) && (Obj.Par(2) > 0) && (Obj.Par(3) < Obj.Par(4))
                            else
                                error('The Beta distribution is not defined for your parameters');
                            end
                        
                        case 'binomial'
                            if (0 <= Obj.Par(2)) && (Obj.Par(2) <= 1) && (Obj.Par(1) > 0) && (mod(Obj.Par(1),1) == 0)
                            else
                                error('The Binomial distribution is not defined for your parameters');
                            end
                            
                        case 'chisquare'
                            % special case of gamma distribution
                            if (Obj.Par>0) && (mod(Obj.Par,1)==0)   
                            else
                                error('The Chisquared distribution is not defined for your parameters');
                            end

                        case 'exponential'
                            if Obj.Par > 0
                            else
                                error('The Exponential distribution is not defined for your parameter');
                            end

                        case 'frechet'
                            if (Obj.Par(1) > 0) && (Obj.Par(2) > 0) 
                            else
                                error('The Frechet distribution is not defined for your parameters');
                            end

                        case 'gamma'
                            if (Obj.Par(1) > 0) && (Obj.Par(2) > 0)
                            else
                                error('The Gamma distribution is not defined for your parameters');
                            end

                        case 'geometric'
                            % special case of negative binomial distribution
                            if (0 < Obj.Par) && (Obj.Par <= 1)
                            else
                                error('The Geometric distribution is not defined for your parameter');
                            end

                        case 'gev'
                            if Obj.Par(2) > 0  
                            else
                                error('The Generalized Extreme Value distribution is not defined for your parameters');
                            end
                            
                        case 'gevmin'
                            if Obj.Par(2) > 0    
                            else
                                error('The Generalized Extreme Value distribution is not defined for your parameters');
                            end
                            
                        case 'gumbel'  % mirror image of this distribution can be used to model maxima
                            if Obj.Par(1) > 0
                                % sigma is the scale parameter
                                % mu is the location parameter 
                            else
                                error('The Gumbel distribution is not defined for your parameters');
                            end
                            
                        case 'gumbelmin'  % this distribution can be used to model minima
                            if Obj.Par(1) > 0
                                % sigma is the scale parameter
                                % mu is the location parameter 
                            else
                                error('The Gumbel (min) distribution is not defined for your parameters');
                            end
                            
                        case 'lognormal'
                            if Obj.Par(2) > 0    
                            else
                                error('The Lognormal distribution is not defined for your parameters');
                            end
                            
                        case 'negativebinomial'
                            if (0 < Obj.Par(2)) && (Obj.Par(2) <= 1) && (Obj.Par(1) > 0) && (mod(Obj.Par(1),1) == 0)
                            else
                                error('The Negative Binomial distribution is not defined for your parameters');
                            end

                        case {'normal','gaussian'}
                            if Obj.Par(2) > 0  
                            else
                                error('The Normal distribution is not defined for your parameters');
                            end

                        case 'pareto'
                            if (Obj.Par(1) > 0) && (Obj.Par(2) > 0) 
                            else
                                error('The Pareto distribution is not defined for your parameters');
                            end

                        case 'poisson'
                            n = length(Obj.Par);
                            if n == 2
                                if (Obj.Par(1) > 0) && (Obj.Par(2) > 0)
                                else
                                    error('The Poisson distribution is not defined for your parameters');
                                end
                            elseif n == 1
                                if (Obj.Par > 0)
                                    Obj.Par(2)=1;
                                else
                                    error('The Poisson distribution is not defined for your parameter');
                                end
                            else
                                error('The Poisson distribution is not defined for your parameter');
                            end
                            
                        case 'rayleigh'
                            if Obj.Par > 0
                            else
                                error('The Rayleigh distribution is not defined for your parameters');
                            end
                            
                        case {'standardnormal','standardgaussian'}
                            if ~isempty(val)
                                if ~all(val==[0,1])
                                    error('Wrong parameters are given as input.')
                                end
                            end
                            Obj.Par = [0,1];
                            
                        case 'truncatednormal'
                            if Obj.Par(3)>=Obj.Par(4)
                                error('The upper bound b must be larger than the lower bound a.');
                            end
                            if Obj.Par(2)<=0
                                error('The parameter sig_N must be non-negative.')
                            end
                            
                        case 'uniform'
                             if val(2) <= val(1)
                                error('The upper bound b must be larger than the lower bound a.');
                             end

                        case 'weibull'
                            if (Obj.Par(1) > 0) && (Obj.Par(2) > 0)   
                            else
                                error('The Weibull distribution is not defined for your parameters');
                            end
   
                        otherwise
                            error('Distribution type not available');
                            
                    end
                    
                %------------------------------------------------------------------------------------------------------------------
                % IF MOM is chosen, the parameters of the distribution are
                % obtained with the according transformation.
                
                case 'MOM'
                    if length(val) > 1 && val(2) < 0
                        error('The standard deviation must be non-negative.');
                    else
                        switch lower(Obj.Name)
                            
                             case 'beta'
                                if val(4) <= val(3)
                                    error('The upper bound b must be larger than the lower bound a.');
                                end
                                % Solve System of two equations for the parameters of the distribution
                                Obj.Par(1) = ((val(4)-val(1))*(val(1)-val(3))/val(2)^2-1)*...
                                    (val(1)-val(3))/(val(4)-val(3));
                                Obj.Par(2) = Obj.Par(1)*(val(4)-val(1))/(val(1)-val(3));
                                Obj.Par(3) = val(3);
                                Obj.Par(4) = val(4);
                                % Evaluate if distribution can be defined on the parameters
                                if (Obj.Par(1)>0) && (Obj.Par(2)>0)
                                else
                                    error('Please select other moments');
                                end
                                
                            case 'binomial'
                                % Solve System of two equations for the parameters of the distribution
                                Obj.Par(2) = 1-(val(2)^2/val(1));
                                Obj.Par(1) = val(1)/Obj.Par(2);
                                % Evaluate if distribution can be defined on the parameters
                                if mod(Obj.Par(1),1) <= 1e-4
                                    Obj.Par(1) = round(Obj.Par(1),0);
                                else
                                    error('Please select other moments');
                                end
                                if (0 <= Obj.Par(2)) && (Obj.Par(2) <= 1) % OK
                                else
                                    error('Please select other moments');
                                end
                                
                            case 'chisquare'
                                % Solve Equation for the parameter of the distribution based on the first moment
                                Obj.Par = val(1);
                                % Evaluate if distribution can be defined on the paramater and if the moments are well defined
                                if mod(Obj.Par,1) <= 1e-4
                                    Obj.Par = round(Obj.Par,0);
                                end
                                if (0 < Obj.Par) && (Obj.Par <= Inf)                                    
                                else
                                    error('Please select other moments');
                                end
                                
                            case 'exponential'
                                Obj.Par = 1/val(1);
                                % Evaluate if distribution can be defined on the paramater and if the moments are well defined
                                if (0 <= Obj.Par) && (Obj.Par <= Inf)
                                else
                                    error('Please select other moments');
                                end
                                
                            case 'frechet'
                                % Solve equation for the parameters of the distribution
                                options = optimset('Display','off');
                                par0    = [2.001,1.0e3];
                                fun     = @(par) sqrt(gamma(1-2/par)-(gamma(1-1/par)).^2)./...
                                    gamma(1-1/par)-val(2)/val(1);
                                [xs,~,exitflag] = fzero(fun,par0,options);
                                if exitflag > 0
                                    Obj.Par(2) = xs;
                                    Obj.Par(1) = val(1)/gamma(1-1/Obj.Par(2));
                                else
                                    error('fzero could not converge to a solution for determining the parameters of the frechet distribution');
                                end
                                % Evaluate if distribution can be defined on the parameters
                                if (Obj.Par(1) > 0) && (Obj.Par(2) > 0)
                                else
                                    error('Please select other moments');
                                end
                                
                            case 'gamma'
                                Obj.Par(1)= val(1)/(val(2)^2);
                                Obj.Par(2)= val(1)^2/(val(2)^2);
                                % Evaluate if distribution can be defined on the parameters
                                if (0 < Obj.Par(1)) && (0 < Obj.Par(2))
                                else
                                    error('Please select other moments');
                                end
                                
                            case 'geometric'
                                % Solve Equation for the parameter of the distribution based on the first moment
                                Obj.Par = 1/val(1);
                                % Evaluate if distribution can be defined on the paramater and if the moments are well defined
                                if (0 <= Obj.Par) && (Obj.Par <= 1)
                                else
                                    error('Please select other moments');
                                end

                            case 'gev'
                                %set parameter Beta
                                Obj.Par(1) = val(3);
                                if Obj.Par(1) == 0 % corresponds to Gumbel distribution
                                    ne = 0.57721566490153;   % euler constant
                                    % Solve two equations for the parameters of the distribution
                                    Obj.Par(2) = val(2)*sqrt(6)/pi;       % scale parameter
                                    Obj.Par(3) = val(1) - ne*Obj.Par(2);  % location parameter
                                elseif Obj.Par(1) >= 0.5
                                    error('''MOM''-description can only be used for beta < 0.5 .')
                                else
                                    %calculating parameter sigma = alpha, Beta=csi must be
                                    %taken with absolute value in the
                                    %multiplication
                                    Obj.Par(2) = abs(Obj.Par(1))*val(2)/sqrt(gamma(1-2*Obj.Par(1))-gamma(1-Obj.Par(1))^2);
                                    %calculating parameter mu = epsilon
                                    Obj.Par(3)= val(1)-(Obj.Par(2)/Obj.Par(1)*(gamma(1-Obj.Par(1))-1));
                                end

                            case 'gevmin'
                                %set parameter Beta
                                Obj.Par(1) = val(3);
                                if Obj.Par(1) == 0 % corresponds to Gumbel distribution
                                    ne = 0.57721566490153;   % euler constant
                                    % Solve two equations for the parameters of the distribution
                                    Obj.Par(2) = val(2)*sqrt(6)/pi;       % scale parameter
                                    Obj.Par(3) = val(1) + ne*Obj.Par(2);  % location parameter
                                elseif Obj.Par(1) >= 0.5
                                    error('''MOM''-description can only be used for beta < 0.5 .')
                                else
                                    %calculating parameter sigma, Beta must be
                                    %taken with absolute value in the
                                    %multiplication
                                    Obj.Par(2) = abs(Obj.Par(1))*val(2)/sqrt(gamma(1-2*Obj.Par(1))-gamma(1-Obj.Par(1))^2);
                                    %calculating parameter mu
                                    Obj.Par(3)= val(1)+(Obj.Par(2)/Obj.Par(1)*(gamma(1-Obj.Par(1))-1));
                                end
                                
                            case 'gumbel'   % gumbel can be used to model maxima
                                ne = 0.57721566490153;   % euler constant
                                % Solve two equations for the parameters of the distribution
                                Obj.Par(1) = val(2)*sqrt(6)/pi;       % scale parameter
                                Obj.Par(2) = val(1) - ne*Obj.Par(1);  % location parameter
                                
                            case 'gumbelmin'   % mirror image of gumbel can be used to model minima
                                ne = 0.57721566490153;   % euler constant
                                % Solve two equations for the parameters of the distribution
                                Obj.Par(1) = val(2)*sqrt(6)/pi;       % scale parameter
                                Obj.Par(2) = val(1) + ne*Obj.Par(1);  % location parameter

                            case 'lognormal'
                                % Solve two equations for the parameters of the distribution
                                Obj.Par(1) = log(val(1)) - log(sqrt(1+(val(2)/val(1))^2));  % mean normal
                                Obj.Par(2) = sqrt(log(1+(val(2)/val(1))^2));   % sigma normal
                                
                            case 'negativebinomial'
                                % Solve System of two equations for the parameters of the distribution
                                Obj.Par(2) = val(1)/(val(1)+val(2)^2);
                                Obj.Par(1) = Obj.Par(2)*val(1);
                                % Evaluate if distribution can be defined on the parameters
                                if mod(Obj.Par(1),1) <= 1e-4
                                    Obj.Par(1) = round(Obj.Par(1),0);
                                elseif (0 <= Obj.Par(2)) && (Obj.Par(2) <= 1)
                                else
                                    error('Please select other moments');
                                end

                            case {'normal','gaussian'}
                                if val(2) > 0
                                    Obj.Par  = val;
                                else
                                    error('The Normal distribution is not defined for your parameters');
                                end

                            case 'pareto'
                                % Solve System of two equations for the parameters of the distribution
                                Obj.Par(2) = 1 + sqrt(1+(val(1)/val(2))^2);
                                Obj.Par(1) = val(1)*(Obj.Par(2)-1)/Obj.Par(2);
                                % Evaluate if distribution can be defined on the parameters
                                if (Obj.Par(1) > 0) && (Obj.Par(2 )> 0)
                                else
                                    error('Please select other moments');
                                end
                                
                            case 'poisson'
                                n=length(val);
                                if n==1
                                    Obj.Par(1) = val(1);
                                    Obj.Par(2) = 1;
                                elseif n==2
                                    Obj.Par(1) = val(1)/val(2);
                                    Obj.Par(2) = val(2);
                                end
                                % Evaluate if moments match
                                if 0 < Obj.Par(1) && 0 < Obj.Par(2)
                                else
                                    error('Please select other moments');
                                end
                                
                            case 'rayleigh'
                                % Solve Equation for the parameter of the distribution based on the first moment
                                Obj.Par = val(1)/sqrt(pi/2);
                                % Evaluate if distribution can be defined on the paramater and if the moments are well defined
                                if (0 < Obj.Par) && (Obj.Par <= Inf)
                                else
                                    error('Please select other moments');
                                end

                            case {'standardnormal','standardgaussian'}
                                % special case of normal distribution
                                Obj.Par  = [0,1];
                                
                            case 'truncatednormal'
                                if val(3) >= val(4)
                                    error('The upper bound b must be larger than the lower bound a.');
                                end
                                if val(1) <= val(3) || val(1) >= val(4)
                                    error('The mean of the distribution must be within the interval (a,b].')
                                end    
                                
                                f = @(x,par)(1/sqrt(2*par(2)^2*pi)*exp(-(x-par(1)).^2/(2*par(2)^2)))/(normcdf(val(4),par(1),par(2))-normcdf(val(3),par(1),par(2))); % pdf
                                expec_eq = @(par)integral((@(x)x.*f(x,par)),val(3),val(4))-val(1); % difference between actual mean and targeted mean
                                std_eq = @(par)sqrt(integral((@(x)x.^2.*f(x,par)),val(3),val(4))-(integral((@(x)x.*f(x,par)),val(3),val(4)))^2)-val(2); % difference between actual std and targeted std
                                eq = @(val)[expec_eq(val);std_eq(val)];
                                opts = optimoptions('fsolve','FunctionTolerance',1e-12,'display','off'); % options for solution procedure
                                [sol,~,flag] = fsolve(eq,[val(1),val(2)],opts);
                                if flag < 1
                                    error('No suitable distribution parameters can be found for the given moments and bounds.');
                                end
                                Obj.Par = [round(sol(1),4),round(sol(2),4),val(3),val(4)];

                            case 'uniform'
                                % compute parameters
                                Obj.Par(1) = val(1) - sqrt(12)*val(2)/2;
                                Obj.Par(2) = val(1) + sqrt(12)*val(2)/2;
  
                            case 'weibull'
                                % Solve equation for the parameters of the distribution
                                options = optimset('Display','off');
                                par0    = [0.02,1.0e3];
                                fun     = @(par) sqrt(gamma(1+2/par)-(gamma(1+1/par)).^2)./gamma(1+1/par)-val(2)/val(1);
                                [xs,~,exitflag] = fzero(fun,par0,options);
                                if exitflag > 0
                                    Obj.Par(2) = xs;
                                    Obj.Par(1) = val(1)/gamma(1+1/Obj.Par(2));
                                else
                                    error('fzero could not converge to a solution for determining the parameters of the weibull distribution')
                                end
                                % Evaluate if distribution can be defined on the parameters
                                if (Obj.Par(1) > 0) && (Obj.Par(2) > 0)    
                                else
                                    error('Please select other moments');
                                end
   
                            otherwise
                                error('Distribution type not available');
                        end
                    end
                    
                %------------------------------------------------------------------------------------------------------------------
                % IF DATA is chosen, the parameters of the distribution are
                % estimated from the given input vector [X] with maximum
                % likelihood estimation.
                
                case 'DATA'
                    
                    if ~isvector(val)
                        error('Input val must be given in vector shape.')
                    end
                    if isrow(val)
                        val=val';
                    end
                        
                    switch lower(name)
                        
                        case 'beta'
                            Obj.Par(3:4) = val{2}; % 'lower' and 'upper' must be given
                            val = val{1};
                            if ~isvector(val)
                                error('Input val must be given in vector shape.')
                            end
                            if isrow(val)
                                val=val';
                            end
                            if min(val)>=Obj.Par(3) && max(val)<=Obj.Par(4)
                                val = (val-Obj.Par(3))/(Obj.Par(4)-Obj.Par(3));
                                Obj.Par(1:2) = betafit(val);
                            else
                                error('val must be in the given range [lower,upper]');
                            end
                            
                        case 'binomial'
                            Obj.Par(1) = val{2}; % n must be given
                            if Obj.Par(1)<1 &&  mod(Obj.Par(1),1)~=0
                                error('n must be a positive integer');
                            end
                            val = val{1};
                            if ~isvector(val)
                                error('Input val must be given in vector shape.')
                            end
                            if isrow(val)
                                val=val';
                            end
                            if sum(mod(val,1)) == 0 && min(val)>=0 && max(val)<=Obj.Par(1)
                                Obj.Par(2) = mean(val)/Obj.Par(1);
                            else
                                error('val must consist of integers in the range [0,n]');
                            end
                            
                        case 'chisquare'
                            if min(val)>=0
                                Obj.Par = mle(val,'pdf',@(x,v)chi2pdf(x,v),'start',mean(val));
                            else
                                error('val must be non-negative');
                            end
                            
                        case 'exponential'
                            if min(val)>=0
                                Obj.Par = 1/mean(val);
                            else
                                error('val must be non-negative');
                            end
                            
                        case 'frechet'
                            if min(val)>=0
                                Obj.Par = mle(val,'pdf',@(val,a_n,k)gevpdf(val,1/k,a_n/k,a_n),'start',[1,1]);
                            else
                                error('val must be non-negative');
                            end
                            
                        case 'gamma'
                            if min(val)>=0
                                param = gamfit(val);
                                Obj.Par(1) = 1/param(2);
                                Obj.Par(2) = param(1);
                            else
                                error('val must be non-negative');
                            end
                            
                        case 'geometric'
                            if sum(mod(val,1)) == 0 && min(val)>0
                                Obj.Par=1/mean(val);
                            else
                                error('val must consist of positive integers');
                            end
                            
                        case 'gev'
                            Obj.Par = gevfit_alt(val); % function can be found at the bottom of this file
                            
                        case 'gevmin'
                            Obj.Par = gevfit_alt(-val); % function can be found at the bottom of this file
                            Obj.Par(3) = -Obj.Par(3);
                            
                        case 'gumbel'
                            ne = 0.57721566490153;
                            an_s = sqrt(std(val)^2*6/pi^2);
                            bn_s = -an_s*ne+mean(val);
                            Obj.Par = mle(val,'pdf',@(val,sig,mu)gevpdf(val,0,sig,mu),'start',[an_s,bn_s]);
                            
                        case 'gumbelmin'
                            ne = 0.57721566490153;
                            an_s = sqrt(std(val)^2*6/pi^2);
                            bn_s = an_s*ne-mean(-val);
                            Obj.Par = mle(val,'pdf',@(val,sig,mu)gevpdf(-val,0,sig,-mu),'start',[an_s,bn_s]);
                            
                        case 'lognormal'
                            if min(val)>0
                                Obj.Par(1) = mean(log(val));
                                Obj.Par(2) = sqrt(mean((log(val)-Obj.Par(1)).^2));
                            else
                                error('val must be positive');
                            end
                            
                        case 'negativebinomial'
                            % first estimation of k,p with method of
                            % moments
                            p = mean(val)/(mean(val)+var(val));
                            k = mean(val)*p;
                            Obj.Par(1) = round(k); % rounding of k, since k must be a positive integer according to ERADist definition
                            Obj.Par(2) = Obj.Par(1)/mean(val); % estimation of p for rounded k (mle)
                            if k==0
                                error('No suitable parameters can be estimated from the given data.')
                            end
                            
                        case {'normal','gaussian'}
                            Obj.Par(1)= mean(val);
                            Obj.Par(2)= std(val);
                            
                        case 'pareto'
                            Obj.Par(1) = min(val);
                            if Obj.Par(1)>0
                                Obj.Par(2) = mle(val,'pdf',@(val,alpha)gppdf(val,1/alpha,Obj.Par(1)/alpha,Obj.Par(1)),'start',Obj.Par(1));
                            else
                                error('val must be positive');
                            end
                            
                        case 'poisson'
                            if iscell(val)
                                Obj.Par(2) = val{2}; % t may be given
                                val = val{1};
                                if ~isvector(val)
                                    error('Input val must be given in vector shape.')
                                end
                                if isrow(val)
                                    val=val';
                                end
                                if Obj.Par(2)<=0
                                    error('t must be positive');
                                elseif sum(mod(val,1)) == 0 && min(val)>=0
                                    Obj.Par(1) = mean(val)/Obj.Par(2);
                                else
                                    error('val must consist of non-negative integers');
                                end
                            else
                                if sum(mod(val,1)) == 0 && min(val)>=0
                                    Obj.Par(1) = mean(val);
                                    Obj.Par(2) = 1;
                                else
                                    error('val must consist of non-negative integers');
                                end
                            end
                            
                        case 'rayleigh'
                            if min(val)>=0
                                Obj.Par = sqrt(mean(val.^2/2));
                            else
                                error('val must be non-negative');
                            end
                            
                        case 'truncatednormal'
                            bounds = val{2};
                            data = val{1};
                            if bounds(1) >= bounds(2)
                                error('The upper bound b must be larger than the lower bound a.');
                            end
                            if ~all(data>=bounds(1)) ||  ~all(data<=bounds(2))
                                error('All samples in X must be within the interval (a,b].')
                            end
                            par = mle(data,'pdf',@(data,mu_N,sig_N)(data >= bounds(1) & data <= bounds(2)).*normpdf(data,mu_N,sig_N)/(normcdf(bounds(2),mu_N,sig_N)-normcdf(bounds(1),mu_N,sig_N)),'start',[mean(data),std(data)]);
                            Obj.Par=[par(1),par(2),bounds(1),bounds(2)];
                            
                        case 'uniform'
                            Obj.Par(1) = min(val);
                            Obj.Par(2) = max(val);
                            
                        case 'weibull'
                            if min(val)>=0
                                Obj.Par = wblfit(val);
                            else
                                error('val must be non-negative');
                            end
                            
                        otherwise
                            disp('Distribution type not available');
                    end
                    
                otherwise
                    error('opt must be PAR, MOM or DATA');
            end
        end
        
        %------------------------------------------------------------------------------------------------------------------
        function MEAN = mean(Obj)
            % Returns the mean of the distribution.
            %
            % MEAN = mean(Obj)
            %
            
            switch lower(Obj.Name)
                case 'beta'
                    MEAN = (Obj.Par(1)*Obj.Par(4)+Obj.Par(2)*Obj.Par(3))/(Obj.Par(1)+Obj.Par(2));
                case 'binomial'
                    MEAN = Obj.Par(1)*Obj.Par(2);
                case 'chisquare'
                    MEAN = Obj.Par;
                case 'exponential'
                    MEAN = 1/Obj.Par;
                case 'frechet'
                    MEAN = Obj.Par(1)*(gamma(1-1/Obj.Par(2)));
                case 'gamma'
                    MEAN = Obj.Par(2)/Obj.Par(1);
                case 'geometric'
                    MEAN = 1/Obj.Par;
                case 'gev'
                    if Obj.Par(1) == 0 % corresponds to Gumbel distribution
                        ne = 0.57721566490153;
                        MEAN = Obj.Par(3)+Obj.Par(2)*ne;
                    else
                        MEAN = Obj.Par(3)+Obj.Par(2)*(gamma(1-Obj.Par(1))-1)/Obj.Par(1);
                    end
                case 'gevmin'
                    if Obj.Par(1) == 0 % corresponds to Gumbel distribution
                        ne = 0.57721566490153;
                        MEAN = Obj.Par(3)-Obj.Par(2)*ne;
                    else
                        MEAN = Obj.Par(3)-Obj.Par(2)*(gamma(1-Obj.Par(1))-1)/Obj.Par(1);
                    end
                case 'gumbel'
                    ne = 0.57721566490153;
                    MEAN = Obj.Par(2)+Obj.Par(1)*ne;
                case 'gumbelmin'
                    ne = 0.57721566490153;
                    MEAN = Obj.Par(2)-Obj.Par(1)*ne;
                case 'lognormal'
                    MEAN = exp(Obj.Par(1)+(Obj.Par(2)^2/2));
                case 'negativebinomial'
                    MEAN = Obj.Par(1)/Obj.Par(2);
                case {'normal','gaussian','standardnormal','standardgaussian'}
                    MEAN = Obj.Par(1);
                case 'pareto'
                    MEAN = Obj.Par(1)*Obj.Par(2)/(Obj.Par(2)-1);
                case 'poisson'
                    MEAN = Obj.Par(1)*Obj.Par(2);
                case 'rayleigh'
                    MEAN = Obj.Par*sqrt(pi/2);
                case 'truncatednormal'
                    f = @(x)(1/sqrt(2*Obj.Par(2)^2*pi)*exp(-(x-Obj.Par(1)).^2/(2*Obj.Par(2)^2)))/(normcdf(Obj.Par(4),Obj.Par(1),Obj.Par(2))-normcdf(Obj.Par(3),Obj.Par(1),Obj.Par(2))); % pdf
                    MEAN = integral((@(x)x.*f(x)),Obj.Par(3),Obj.Par(4));
                case 'uniform'
                    MEAN = (Obj.Par(2)+Obj.Par(1))/2;
                case 'weibull'
                    MEAN = Obj.Par(1)*(gamma(1+1/Obj.Par(2)));
                otherwise
                    disp('Error - distribution not available');
            end
        end
        
        %------------------------------------------------------------------------------------------------------------------
        function Standarddeviation = std(Obj)
            % Returns the standard deviation of the distribution.
            %
            % Standarddeviation = std(Obj)
            %
            
            switch lower(Obj.Name)
                case 'beta'
                    Standarddeviation = (Obj.Par(4)-Obj.Par(3))*sqrt(Obj.Par(1)*Obj.Par(2)/(Obj.Par(1)+Obj.Par(2))^2/(Obj.Par(1)+Obj.Par(2)+1));
                case 'binomial'
                    Standarddeviation = sqrt(Obj.Par(1)*Obj.Par(2)*(1-Obj.Par(2)));
                case 'chisquare'
                    Standarddeviation = sqrt(2*Obj.Par);
                case 'exponential'
                    Standarddeviation = 1/Obj.Par;
                case 'frechet'
                    Standarddeviation = Obj.Par(1)*(gamma(1-2/Obj.Par(2))-gamma(1-1/Obj.Par(2))^2)^0.5;
                case 'gamma'
                    Standarddeviation = sqrt(Obj.Par(2)/Obj.Par(1)^2);
                case 'geometric'
                    Standarddeviation = sqrt((1-Obj.Par)/Obj.Par^2);
                case 'gev'
                    if Obj.Par(1) == 0
                        Standarddeviation = sqrt(pi^2*Obj.Par(2)^2/6);
                    elseif Obj.Par(1) >= 0.5
                        Standarddeviation = Inf;
                    else   
                        Standarddeviation = sqrt(Obj.Par(2)^2/Obj.Par(1)^2*(gamma(1-2*Obj.Par(1))-gamma(1-Obj.Par(1))^2));
                    end
                case 'gevmin'
                    if Obj.Par(1) == 0
                        Standarddeviation = sqrt(pi^2*Obj.Par(2)^2/6);
                    elseif Obj.Par(1) >= 0.5
                        Standarddeviation = Inf;
                    else   
                        Standarddeviation = sqrt(Obj.Par(2)^2/Obj.Par(1)^2*(gamma(1-2*Obj.Par(1))-gamma(1-Obj.Par(1))^2));
                    end
                case 'gumbel'
                    Standarddeviation = sqrt(pi^2*Obj.Par(1)^2/6);
                case 'gumbelmin'
                    Standarddeviation = sqrt(pi^2*Obj.Par(1)^2/6);
                case 'lognormal'
                    Standarddeviation = sqrt((exp(Obj.Par(2)^2)-1)*exp(2*Obj.Par(1)+Obj.Par(2)^2));
                case 'negativebinomial'
                    Standarddeviation = sqrt(Obj.Par(1)/Obj.Par(2)^2*(1-Obj.Par(2)));
                case {'normal','gaussian','standardnormal','standardgaussian'}
                    Standarddeviation = Obj.Par(2);
                case 'pareto'
                    Standarddeviation = sqrt((Obj.Par(1)^2*Obj.Par(2))/(Obj.Par(2)-1)^2/(Obj.Par(2)-2));
                case 'poisson'
                    Standarddeviation = sqrt(Obj.Par(1)*Obj.Par(2));
                case 'rayleigh'
                    Standarddeviation = sqrt((4-pi)/2*Obj.Par^2);
                case 'truncatednormal'
                    f = @(x)(1/sqrt(2*Obj.Par(2)^2*pi)*exp(-(x-Obj.Par(1)).^2/(2*Obj.Par(2)^2)))/(normcdf(Obj.Par(4),Obj.Par(1),Obj.Par(2))-normcdf(Obj.Par(3),Obj.Par(1),Obj.Par(2))); % pdf
                    Standarddeviation = sqrt(integral((@(x)x.^2.*f(x)),Obj.Par(3),Obj.Par(4))-(integral((@(x)x.*f(x)),Obj.Par(3),Obj.Par(4)))^2);
                case 'uniform'
                    Standarddeviation = sqrt(((Obj.Par(2)-Obj.Par(1))^2)/12);
                case 'weibull'
                    Standarddeviation = Obj.Par(1)*(gamma(1+2/Obj.Par(2))-gamma(1+1/Obj.Par(2))^2)^0.5;
                otherwise
                    disp('Error - distribution not available');
            end
        end
 
        %------------------------------------------------------------------------------------------------------------------
        function CDF = cdf(Obj,x)
            % Returns the CDF value.
            %
            % CDF = cdf(Obj,x)
            %
            
            switch lower(Obj.Name)
                case 'beta'
                    CDF = betacdf((x-Obj.Par(3))/(Obj.Par(4)-Obj.Par(3)),Obj.Par(1),Obj.Par(2));
                case 'binomial'
                    CDF = binocdf(x,Obj.Par(1),Obj.Par(2));
                case 'chisquare'
                    CDF = chi2cdf(x,Obj.Par);
                case 'exponential'
                    CDF = expcdf(x,1/Obj.Par(1));
                case 'frechet'
                    CDF = gevcdf(x,1/Obj.Par(2),Obj.Par(1)/Obj.Par(2),Obj.Par(1));
                case 'gamma'
                    CDF = gamcdf(x,Obj.Par(2),1/Obj.Par(1));
                case 'geometric'
                    CDF = geocdf(x-1,Obj.Par);
                case 'gev'
                    CDF = gevcdf(x,Obj.Par(1),Obj.Par(2),Obj.Par(3));
                case 'gevmin'
                    CDF = 1-gevcdf(-x,Obj.Par(1),Obj.Par(2),-Obj.Par(3));
                case 'gumbel'
                    CDF = gevcdf(x,0,Obj.Par(1),Obj.Par(2));
                case 'gumbelmin'
                    CDF = 1-gevcdf(-x,0,Obj.Par(1),-Obj.Par(2));
                case 'lognormal'
                    CDF = logncdf(x,Obj.Par(1),Obj.Par(2));
                case 'negativebinomial'
                    CDF = nbincdf(x-Obj.Par(1),Obj.Par(1),Obj.Par(2));
                case {'normal','gaussian','standardnormal','standardgaussian'}
                    CDF = normcdf(x,Obj.Par(1),Obj.Par(2));
                case 'pareto'
                    CDF = gpcdf(x,1/Obj.Par(2),Obj.Par(1)/Obj.Par(2),Obj.Par(1));
                case 'poisson'
                    CDF = poisscdf(x, Obj.Par(1)*Obj.Par(2));
                case 'rayleigh'
                    CDF = raylcdf(x,Obj.Par);
                case 'truncatednormal'
                    CDF = (Obj.Par(3) <= x & x <= Obj.Par(4)).*(normcdf(x,Obj.Par(1),Obj.Par(2))-normcdf(Obj.Par(3),Obj.Par(1),Obj.Par(2)))/(normcdf(Obj.Par(4),Obj.Par(1),Obj.Par(2))-normcdf(Obj.Par(3),Obj.Par(1),Obj.Par(2)));
                case 'uniform'
                    CDF = unifcdf(x,Obj.Par(1),Obj.Par(2));
                case 'weibull'
                    CDF = wblcdf(x,Obj.Par(1),Obj.Par(2));
                otherwise
                    disp('Distribution type not available');
            end
        end
        
        %------------------------------------------------------------------------------------------------------------------
        function InverseCDF = icdf(Obj,y)
            % Returns the value of the inverse CDF.
            %
            % InverseCDF = icdf(Obj,y)
            %
            
            switch lower(Obj.Name)
                case 'beta'
                    InverseCDF = betainv(y,Obj.Par(1),Obj.Par(2))*(Obj.Par(4)-Obj.Par(3))+Obj.Par(3);
                case 'binomial'
                    InverseCDF = binoinv(y,Obj.Par(1),Obj.Par(2));
                case 'chisquare'
                    InverseCDF = chi2inv(y,Obj.Par);
                case 'exponential'
                    InverseCDF = expinv(y,1/Obj.Par);
                case 'frechet'
                    InverseCDF = gevinv(y,1/Obj.Par(2),Obj.Par(1)/Obj.Par(2),Obj.Par(1));
                case 'gamma'
                    InverseCDF = gaminv(y,Obj.Par(2),1/Obj.Par(1));
                case 'geometric'
                    InverseCDF = geoinv(y,Obj.Par)+1;
                case 'gev'
                    InverseCDF = gevinv(y,Obj.Par(1),Obj.Par(2),Obj.Par(3));
                case 'gevmin'
                    InverseCDF = -gevinv(1-y,Obj.Par(1),Obj.Par(2),-Obj.Par(3));
                case 'gumbel'
                    InverseCDF = gevinv(y,0,Obj.Par(1),Obj.Par(2));
                case 'gumbelmin'
                    InverseCDF = -gevinv(1-y,0,Obj.Par(1),-Obj.Par(2));
                case 'lognormal'
                    InverseCDF = logninv(y,Obj.Par(1),Obj.Par(2));
                case 'negativebinomial'
                    InverseCDF = nbininv(y,Obj.Par(1),Obj.Par(2))+Obj.Par(1);
                case {'normal','gaussian','standardnormal','standardgaussian'}
                    InverseCDF = norminv(y,Obj.Par(1),Obj.Par(2));
                case 'pareto'
                    InverseCDF = gpinv(y,1/Obj.Par(2),Obj.Par(1)/Obj.Par(2),Obj.Par(1));
                case 'poisson'
                    InverseCDF = poissinv(y, Obj.Par(1)*Obj.Par(2));
                case 'rayleigh'
                    InverseCDF = raylinv(y,Obj.Par);
                case 'truncatednormal'
                    InverseCDF = round(norminv(y*(normcdf(Obj.Par(4),Obj.Par(1),Obj.Par(2))-normcdf(Obj.Par(3),Obj.Par(1),Obj.Par(2)))+normcdf(Obj.Par(3),Obj.Par(1),Obj.Par(2)),Obj.Par(1),Obj.Par(2)),10);
                    InverseCDF(InverseCDF < Obj.Par(3) | Obj.Par(4) < InverseCDF) = NaN;
                case 'uniform'
                    InverseCDF = unifinv(y,Obj.Par(1),Obj.Par(2));
                case 'weibull'
                    InverseCDF = wblinv(y,Obj.Par(1),Obj.Par(2));
                otherwise
                    disp('Distribution type not available');
            end
        end
        
        %------------------------------------------------------------------------------------------------------------------
        function PDF = pdf(Obj,x)
            % Returns the PDF value.
            %
            % PDF = pdf(Obj,x)
            %
            
            switch lower(Obj.Name)
                case 'beta'
                    PDF = betapdf((x-Obj.Par(3))/(Obj.Par(4)-Obj.Par(3)),Obj.Par(1),Obj.Par(2))/(Obj.Par(4)-Obj.Par(3));
                case 'binomial'
                    PDF = binopdf(x,Obj.Par(1),Obj.Par(2));
                case 'chisquare'
                    PDF = chi2pdf(x,Obj.Par);
                case 'exponential'
                    PDF = exppdf(x,1/Obj.Par);
                case 'frechet'
                    PDF = gevpdf(x,1/Obj.Par(2),Obj.Par(1)/Obj.Par(2),Obj.Par(1));
                case 'gamma'
                    PDF = gampdf(x,Obj.Par(2),1/Obj.Par(1));
                case 'geometric'
                    PDF = geopdf(x-1,Obj.Par);
                case 'gev'
                    PDF = gevpdf(x,Obj.Par(1),Obj.Par(2),Obj.Par(3));
                case 'gevmin'
                    PDF = gevpdf(-x,Obj.Par(1),Obj.Par(2),-Obj.Par(3));
                case 'gumbel'
                    PDF = gevpdf(x,0,Obj.Par(1),Obj.Par(2));
                case 'gumbelmin'
                    PDF = gevpdf(-x,0,Obj.Par(1),-Obj.Par(2));
                case 'lognormal'
                    PDF = lognpdf(x,Obj.Par(1),Obj.Par(2));
                case 'negativebinomial'
                    PDF = nbinpdf(x-Obj.Par(1),Obj.Par(1),Obj.Par(2));
                case {'normal','gaussian','standardnormal','standardgaussian'}
                    PDF = normpdf(x,Obj.Par(1),Obj.Par(2));
                case 'pareto'
                    PDF = gppdf(x,1/Obj.Par(2),Obj.Par(1)/Obj.Par(2),Obj.Par(1));
                case 'poisson'
                    PDF = poisspdf(x, Obj.Par(1)*Obj.Par(2));
                case 'rayleigh'
                    PDF = raylpdf(x,Obj.Par);
                case 'truncatednormal'
                    PDF = (Obj.Par(3) <= x & x <= Obj.Par(4)).*normpdf(x,Obj.Par(1),Obj.Par(2))/(normcdf(Obj.Par(4),Obj.Par(1),Obj.Par(2))-normcdf(Obj.Par(3),Obj.Par(1),Obj.Par(2)));
                case 'uniform'
                    PDF = unifpdf(x,Obj.Par(1),Obj.Par(2));
                case 'weibull'
                    PDF = wblpdf(x,Obj.Par(1),Obj.Par(2));
                otherwise
                    disp('Distribution type not available');
            end
        end

        %------------------------------------------------------------------------------------------------------------------
        function Random = random(Obj,m,n)
            % Generates random samples according to the distribution of the
            % object.
            %
            % Random = random(Obj,m,n)
            %
            
            if nargin == 2
                
                switch lower(Obj.Name)
                    case 'beta'
                        Random = betarnd(Obj.Par(1),Obj.Par(2),m)*(Obj.Par(4)-Obj.Par(3))+Obj.Par(3);
                    case 'binomial'
                        Random = binornd(Obj.Par(1),Obj.Par(2),m);
                    case 'chisquare'
                        Random = chi2rnd(Obj.Par,m);
                    case 'exponential'
                        Random = random('exponential',1/Obj.Par,m);
                    case 'frechet'
                        Random = gevrnd(1/Obj.Par(2),Obj.Par(1)/Obj.Par(2),Obj.Par(1),m);
                    case 'gamma'
                        Random = gamrnd(Obj.Par(2),1/Obj.Par(1),m);
                    case 'geometric'
                        Random = geornd(Obj.Par,m)+1;
                    case 'gev'
                        Random = gevrnd(Obj.Par(1),Obj.Par(2),Obj.Par(3),m);
                    case 'gevmin'
                        Random = -gevrnd(Obj.Par(1),Obj.Par(2),-Obj.Par(3),m);
                    case 'gumbel'
                        Random = gevrnd(0,Obj.Par(1),Obj.Par(2),m);
                    case 'gumbelmin'
                        Random = -gevrnd(0,Obj.Par(1),-Obj.Par(2),m);
                    case 'lognormal'
                        Random = lognrnd(Obj.Par(1),Obj.Par(2),m);
                    case 'negativebinomial'
                        Random = nbinrnd(Obj.Par(1),Obj.Par(2),m)+Obj.Par(1);
                    case {'normal','gaussian','standardnormal','standardgaussian'}
                        Random = normrnd(Obj.Par(1),Obj.Par(2),m);
                    case 'pareto'
                        Random = gprnd(1/Obj.Par(2),Obj.Par(1)/Obj.Par(2),Obj.Par(1),m);
                    case 'poisson'
                        Random = poissrnd(Obj.Par(1)*Obj.Par(2),m);
                    case 'rayleigh'
                        Random = raylrnd(Obj.Par,m);
                    case 'truncatednormal'
                        u = rand(m);
                        Random = Obj.icdf(u);
                    case 'uniform'
                        Random = random('uniform',Obj.Par(1),Obj.Par(2),m);
                    case 'weibull'
                        Random = wblrnd(Obj.Par(1),Obj.Par(2),m);
                    otherwise
                        disp('Error - distribution not available');
                end
            end
            
            if nargin == 3
                
                switch lower(Obj.Name)
                    case 'beta'
                        Random = betarnd(Obj.Par(1),Obj.Par(2),m,n)*(Obj.Par(4)-Obj.Par(3))+Obj.Par(3);
                    case 'binomial'
                        Random = binornd(Obj.Par(1),Obj.Par(2),m,n);
                    case 'chisquare'
                        Random = chi2rnd(Obj.Par,m,n);
                    case 'exponential'
                        Random = random('exponential',1/Obj.Par,m,n);
                    case 'frechet'
                        Random = gevrnd(1/Obj.Par(2),Obj.Par(1)/Obj.Par(2),Obj.Par(1),m,n);
                    case 'gamma'
                        Random = gamrnd(Obj.Par(2),1/Obj.Par(1),m,n);
                    case 'geometric'
                        Random = geornd(Obj.Par,m,n)+1;
                    case 'gev'
                        Random = gevrnd(Obj.Par(1),Obj.Par(2),Obj.Par(3),m,n);
                    case 'gevmin'
                        Random = -gevrnd(Obj.Par(1),Obj.Par(2),-Obj.Par(3),m,n);
                    case 'gumbel'
                        Random = gevrnd(0,Obj.Par(1),Obj.Par(2),m,n);
                    case 'gumbelmin'
                        Random = -gevrnd(0,Obj.Par(1),-Obj.Par(2),m,n);
                    case 'lognormal'
                        Random = lognrnd(Obj.Par(1),Obj.Par(2),m,n);
                    case 'negativebinomial'
                        Random = nbinrnd(Obj.Par(1),Obj.Par(2),m,n)+Obj.Par(1);
                    case {'normal','gaussian','standardnormal','standardgaussian'}
                        Random = normrnd(Obj.Par(1),Obj.Par(2),m,n);
                    case 'pareto'
                        Random = gprnd(1/Obj.Par(2),Obj.Par(1)/Obj.Par(2),Obj.Par(1),m,n);
                    case 'poisson'
                        Random = poissrnd(Obj.Par(1)*Obj.Par(2),m,n);
                    case 'rayleigh'
                        Random = raylrnd(Obj.Par,m,n);
                    case 'truncatednormal'
                        u = rand(m,n);
                        Random = Obj.icdf(u);
                    case 'uniform'
                        Random = random('uniform',Obj.Par(1),Obj.Par(2),m,n);
                    case 'weibull'
                        Random = wblrnd(Obj.Par(1),Obj.Par(2),m,n);
                    otherwise
                        disp('Error - distribution not available');
                end
            end
        end
        
    end
end

%% Nested functions: for GEV-parameter fitting

function par = gevfit_alt (y)
% Author: Iason Papaioannou
% The function gevfit_alt evaluates the parameters of the generalized
% extreme value distribution with the method of Probability Weighted
% Moments (PWM) and Maximum Likelihood Estimation (MLE).

% compute PWM estimates
x01 = gevpwm (y);

if x01(1) > 0
    % Compute mle estimates
    x02 = gevfit (y);
    % if alpha reasonable
    if x02(2) >= 1.e-6
        % set parameters
        par = x02;
        if par(1) < -1
            par = x01;
            disp(' ');
            disp('  Warning: The MLE estimate of the shape parameter of the GEV is not in the range where the MLE estimator is valid.');
            disp('  PWM estimation is used.');           
            if par(1) > 0.4   
                disp(' ');
                disp('  Warning: The shape parameter of the GEV is not in the range where PWM asymptotic results are valid.');
            end          
        end        
    else        
        % set parameters obtained by PWM
        par = x01;       
        if par(1) > 0.4
            disp(' ');
            disp('  Warning: The shape parameter of the GEV is not in the range where PWM asymptotic results are valid.');
        end        
    end    
else   
    % set parameters obtained by PWM
    par = x01;
    if par(1) < -0.4
        disp(' ');
        disp('  Warning: The shape parameter of the GEV is not in the range where PWM asymptotic results are valid.');
    end   
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function par = gevpwm (y)
% Author: Iason Papaioannou
% The function gevpwm evaluates the parameters of the generalized
% extreme value distribution applying the method of Probability Weighted
% Moments.

% compute PWM estimates
y2 = sort(y);
beta0 = mean(y);

p1 = ((1:length(y))-1)/(length(y)-1);
p2 = p1.*((1:length(y))-2)/(length(y)-2);
beta1 = p1*y2;
beta2 = p2*y2;

beta1 = beta1/length(y);
beta2 = beta2/length(y);

c = (2*beta1-beta0)/(3*beta2-beta0)-log(2)/log(3);
par0 = -7.8590*c -2.9554*c^2;
par(1) = fzero(@(x) (3*beta2-beta0)/(2*beta1-beta0)-(1-3^x)/(1-2^x),par0);
par(2) = -(2*beta1-beta0)*par(1)/gamma(1-par(1))/(1-2^par(1));
par(3) = beta0 - par(2)/par(1)*(gamma(1-par(1))-1); 
end

