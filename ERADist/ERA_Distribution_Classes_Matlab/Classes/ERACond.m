classdef ERACond
    
% Generation of conditional distribution objects for the use within the
% ERARosen class.
%
% Construction of the conditional distribution object with
%
%         Obj = ERACond(name,opt,param) 
% or      Obj = ERACond(name,opt,param,id)
%
% The available distributions, represented by the input variable 'name', 
% are the same as in the ERADist class (see below). They can be described 
% either by parameters (opt='PAR') or by the first and second moment
% (opt='MOM'). 
%
% The parameters/moments must be given as a cell array of function handles 
% (and numeric scalars) in case of multiple parameters. For distributions with 
% one parameter a simple function handle also works. Examples for function
% handles given by the input 'param' of a two parametric distribution 
% depending on two other random variables (parents) could be:       
%                 
%   param = {@(x)x(:,1)+x(:,2),@(x)0.2*x(:,1).^2},     
%   param = {@(y)3*y(:,1)-2*y(:,2),4}
%
% The input 'id' can be used to better identify the different variables
% (nodes) when plotting the graph describing the dependency between the
% different variables in the ERARosen class (method plotGraph). The input
% 'id' is however not mandatory.
% 
%
% The following distribution types are available:
%
% opt = "PAR", if you want to specify the distibution by its parameters:
%   Beta:                       Obj = ERADist('beta','PAR',{@()r,@()s,@()a,@()b})
%   Binomial:                   Obj = ERADist('binomial','PAR',{@()n,@()p})
%   Chi-squared:                Obj = ERADist('chisquare','PAR',@()k)
%   Exponential:                Obj = ERADist('exponential','PAR',@()lambda)
%   Frechet:                    Obj = ERADist('frechet','PAR',{@()a_n,@()k})
%   Gamma:                      Obj = ERADist('gamma','PAR',{@()lambda,@()k})
%   Geometric:                  Obj = ERADist('geometric','PAR',@()p)
%   GEV (to model maxima):      Obj = ERADist('GEV','PAR',{@()beta,@()alpha,@()epsilon})
%   GEV (to model minima):      Obj = ERADist('GEVMin','PAR',{@()beta,@()alpha,@()epsilon})
%   Gumbel (to model maxima):   Obj = ERADist('gumbel','PAR',{@()a_n,@()b_n})
%   Gumbel (to model minima):   Obj = ERADist('gumbelMin','PAR',{@()a_n,@()b_n})
%   Log-normal:                 Obj = ERADist('lognormal','PAR',{@()mu_lnx,@()sig_lnx})
%   Negative binomial:          Obj = ERADist('negativebinomial','PAR',{@()k,@()p})
%   Normal:                     Obj = ERADist('normal','PAR',{@()mean,@()std})
%   Pareto:                     Obj = ERADist('pareto','PAR',{@()x_m,@()alpha})
%   Poisson:                    Obj = ERADist('poisson','PAR',{@()v,@()t})
%                           or  Obj = ERADist('poisson','PAR',@()lambda)
%   Rayleigh:                   Obj = ERADist('rayleigh','PAR',@()alpha)
%   Truncated normal:           Obj = ERADist('truncatednormal','PAR',{@()mu_N,@()sig_N,@()a,@()b})
%   Uniform:                    Obj = ERADist('uniform','PAR',{@()lower,@()upper})
%   Weibull:                    Obj = ERADist('weibull','PAR',{@()a_n,@()k}) 
%
%
% opt = "MOM", if you want to specify the distibution by its moments:
%   Beta:                       Obj = ERADist('beta','MOM',{@()mean,@()std,@()a,@()b})
%   Binomial:                   Obj = ERADist('binomial','MOM',{@()mean,@()std})
%   Chi-squared:                Obj = ERADist('chisquare','MOM',@()mean)
%   Exponential:                Obj = ERADist('exponential','MOM',@()mean)
%   Frechet:                    Obj = ERADist('frechet','MOM',{@()mean,@()std})
%   Gamma:                      Obj = ERADist('gamma','MOM',{@()mean,@()std})
%   Geometric:                  Obj = ERADist('geometric','MOM',@()mean)
%   GEV (to model maxima):      Obj = ERADist('GEV','MOM',{@()mean,@()std,@()beta})
%   GEV (to model minima):      Obj = ERADist('GEVMin','MOM',{@()mean,@()std,@()beta})
%   Gumbel (to model maxima):   Obj = ERADist('gumbel','MOM',{@()mean,@()std})
%   Gumbel (to model minima):   Obj = ERADist('gumbelMin','MOM',{@()mean,@()std})
%   Log-normal:                 Obj = ERADist('lognormal','MOM',{@()mean,@()std})
%   Negative binomial:          Obj = ERADist('negativebinomial','MOM',{@()mean,@()std})
%   Normal:                     Obj = ERADist('normal','MOM',{@()mean,@()std})
%   Pareto:                     Obj = ERADist('pareto','MOM',{@()mean,@()std})
%   Poisson:                    Obj = ERADist('poisson','MOM',{@()mean,@()t})
%                           or  Obj = ERADist('poisson','MOM',@()mean)
%   Rayleigh:                   Obj = ERADist('rayleigh','MOM',@()mean)
%   Truncated normal:           Obj = ERADist('truncatednormal','MOM',{@()mean,@()std,@()a,@()b})
%   Uniform:                    Obj = ERADist('uniform','MOM',{@()mean,@()std})
%   Weibull:                    Obj = ERADist('weibull','MOM',{@()mean,@()std})

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
Version 2022-01
* Change of the definition of the parameter/moment functions by using
cell arrays of function handles depending on a single input variable.
Allows the use of user-defined variables and functions within the given
function handles. Also allows the definition of functions in matrix-vector 
form. 
Version 2021-03
* Minor changes
First Release, 2020-10  
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
%}    
    
    %% MATLAB class: definition of the 'properties' block
    properties
        Name        % type of distribution
        Opt         % type of parameter defining the distribution ('MOM' or 'PAR')
        Param       % parameters of the distribution as function handle
        ID          % name by which the distribution can be identified
    end
    
    %% MATLAB class: definition of the 'methods' block
    
    methods
        function Obj = ERACond(name,opt,param,id)
            
            Obj.Name = name;
            
            if iscell(param)
                outlen = length(param);
                for i=1:outlen
                    if ~(isa(param{i},'function_handle') || (isscalar(param{i}) && isnumeric(param{i})))
                        error('Input param must be given as a cell consisting of function handles and/or numeric scalars.')
                    end
                end
                Obj.Param = param;
            elseif isa(param,'function_handle')
                Obj.Param = {param};
                outlen = 1;
            else
                error('Input param must be given as a cell consisting of function handles and/or numeric scalars.')
            end
                        
            switch lower(Obj.Name)
                case {'beta','truncatednormal'}
                    if outlen ~= 4
                        error('Incorrect definition of param for the given distribution.')
                    end
                case {'binomial','frechet','gaussian','gamma','gumbel','gumbelmin','lognormal',...
                        'negativebinomial','normal','pareto','uniform','weibull'}
                    if outlen ~= 2
                        error('Incorrect definition of param for the given distribution.')
                    end
                case {'chisquare','exponential','geometric','rayleigh'}
                    if outlen ~= 1
                        error('Incorrect definition of param for the given distribution.')
                    end
                case {'gev','gevmin'}
                    if outlen ~= 3
                        error('Incorrect definition of param for the given distribution.')
                    end
                case 'poisson' % modify function handle so that it has two output parameters, [lambda,t] or [mean,t], in case it hasn't already
                    if outlen == 1
                        Obj.Param=[Obj.Param,1];
                    elseif outlen == 2
                    else 
                        error('Incorrect definition of param for the given distribution.')
                    end       
                otherwise
                    error('Distribution type not available.')
            end
            
            % check if given opt is valid
            switch upper(opt)
                case {'MOM','PAR'}
                    Obj.Opt = upper(opt);                   
                otherwise
                    error('Conditional distributions can only be defined by moments (opt = MOM) or by parameters (opt = PAR).')
            end           

            % check if ID of the random variable is given
            if nargin == 3
                Obj.ID = [];
            elseif nargin == 4
                if ~ischar(id)
                    error('id must be given as a character array.')
                end
                Obj.ID = id;
            elseif nargin < 3
                error('Not enough input arguments.')
            else
                error('To many input arguments.')
            end
            
        end
        
        %------------------------------------------------------------------------------------------------------------------
        function Param = condParam(Obj,cond)
            % Evaluates the parameters of the distribution for the
            % different given conditions.
            % In case that the distribution is described by its moments, 
            % the evaluated moments are used to obtain the distribution 
            % parameters. 
            % This method is used by the ERACond methods condCDF, condPDF,
            % condiCDF and condRandom.
            
            % compute the parameters ('MOM' or 'PAR') for the given
            % conditions
            [n_cond,~] = size(cond); % number of different conditions and parents
            outlen = length(Obj.Param); % number of parameters for the distribution
            param = zeros(n_cond,outlen);
            
            for i = 1:outlen
                if isa(Obj.Param{i},'function_handle')
                    param(:,i)=Obj.Param{i}(cond);
                else
                    param(:,i)=Obj.Param{i};
                end
            end
            
            Param = zeros(n_cond,outlen); % pre-allocation of the output array
            
            % if 'MOM' transform param into actual parametric description
            % (see how 'MOM' inputs are handled in ERADist (same way))
            switch upper(Obj.Opt)
                case 'MOM'
                    
                    switch lower(Obj.Name)
                        
                        case 'beta'
                            Param(:,1) = ((param(:,4)-param(:,1)).*(param(:,1)-param(:,3))./param(:,2).^2-1).*...
                                (param(:,1)-param(:,3))./(param(:,4)-param(:,3));
                            Param(:,2) = Param(:,1).*(param(:,4)-param(:,1))./(param(:,1)-param(:,3));
                            Param(:,3) = param(:,3);
                            Param(:,4) = param(:,4);
                            
                        case 'binomial'
                            Param(:,2) = 1-(param(:,2).^2./param(:,1));
                            Param(:,1) = param(:,1)./Param(:,2);
                            if ~all(mod(Param(:,1),1) <= 1e-4)
                                error('Please select other moments.');
                            else
                                Param(:,1) = round(Param(:,1),0);
                            end
                            
                        case 'chisquare'
                            Param = param;
                            
                        case 'exponential'
                            Param = 1./param;
                            
                        case 'frechet'
                            options = optimset('Display','off');
                            par0    = [2.001,1.0e3];
                            for i = 1:n_cond
                                fun = @(par) sqrt(gamma(1-2/par)-(gamma(1-1/par)).^2)./...
                                    gamma(1-1/par)-param(i,2)/param(i,1);
                                [xs,~,exitflag] = fzero(fun,par0,options);
                                if exitflag > 0
                                    Param(i,2) = xs;
                                    Param(i,1) = param(i,1)/gamma(1-1/Param(i,2));
                                else
                                    Param(i,:)= NaN;
                                end
                                if (Param(i,1) > 0) && (Param(i,2) > 0)
                                else
                                    Param(i,:)= NaN;
                                end
                            end
                            
                        case 'gamma'
                            Param(:,1)= param(:,1)./(param(:,2).^2);
                            Param(:,2)= param(:,1).^2./(param(:,2).^2);
                            
                        case 'geometric'
                            Param = 1./param;
                            
                        case 'gev'
                            Param(:,1) = param(:,3);
                            Param(:,2) = abs(Param(:,1)).*param(:,2)./sqrt(gamma(1-2.*Param(:,1))-gamma(1-Param(:,1)).^2);
                            Param(:,3)= param(:,1)-(Param(:,2)./Param(:,1).*(gamma(1-Param(:,1))-1));
                            
                        case 'gevmin'
                            Param(:,1) = param(:,3);
                            Param(:,2) = abs(Param(:,1)).*param(:,2)./sqrt(gamma(1-2.*Param(:,1))-gamma(1-Param(:,1)).^2);
                            Param(:,3)= param(:,1)+(Param(:,2)./Param(:,1).*(gamma(1-Param(:,1))-1));
                            
                        case 'gumbel'
                            ne = 0.57721566490153;
                            Param(:,1) = param(:,2).*sqrt(6)./pi;
                            Param(:,2) = param(:,1) - ne.*Param(:,1);
                            
                        case 'gumbelmin'
                            ne = 0.57721566490153;
                            Param(:,1) = param(:,2).*sqrt(6)./pi;
                            Param(:,2) = param(:,1) + ne.*Param(:,1);
                            
                        case 'lognormal'
                            Param(:,1) = log(param(:,1)) - log(sqrt(1+(param(:,2)./param(:,1)).^2));
                            Param(:,2) = sqrt(log(1+(param(:,2)./param(:,1)).^2));
                            
                        case 'negativebinomial'
                            Param(:,2) = param(:,1)./(param(:,1)+param(:,2).^2);
                            Param(:,1) = Param(:,2).*param(:,1);
                            
                        case {'normal','gaussian'}
                            Param = param;
                            
                        case 'pareto'
                            Param(:,2) = 1 + sqrt(1+(param(:,1)./param(:,2)).^2);
                            Param(:,1) = param(:,1).*(Param(:,2)-1)./Param(:,2);
                            
                        case 'poisson'
                            Param(:,1) = param(:,1)./param(:,2);
                            Param(:,2) = param(:,2);
                            
                        case 'rayleigh'
                            Param = param./sqrt(pi./2);
                            
                        case 'truncatednormal'
                            for i = 1:n_cond
                                if param(i,3) >= param(i,4) || param(i,1) <= param(i,3) || param(i,1) >= param(i,4)
                                    Param(i,:) = NaN;
                                else
                                    val = param(i,:);
                                    f = @(x,par)(1/sqrt(2*par(2)^2*pi)*exp(-(x-par(1)).^2/(2*par(2)^2)))/(normcdf(val(4),par(1),par(2))-normcdf(val(3),par(1),par(2))); % pdf
                                    expec_eq = @(par)integral((@(x)x.*f(x,par)),val(3),val(4))-val(1); % difference between actual mean and targeted mean
                                    std_eq = @(par)sqrt(integral((@(x)x.^2.*f(x,par)),val(3),val(4))-(integral((@(x)x.*f(x,par)),val(3),val(4)))^2)-val(2); % difference between actual std and targeted std
                                    eq = @(val)[expec_eq(val);std_eq(val)];
                                    opts = optimoptions('fsolve','FunctionTolerance',1e-12,'display','off'); % options for solution procedure
                                    [sol,~,flag,~] = fsolve(eq,[val(1),val(2)],opts);
                                    if flag < 1
                                        Param(i,:) = NaN;
                                    else
                                        %Param(i,:) = [round(sol(1),4),round(sol(2),4),val(3),val(4)];
                                        Param(i,:) = [sol(1),sol(2),val(3),val(4)];
                                    end
                                end  
                            end
                            
                        case 'uniform'
                            Param(:,1) = param(:,1) - sqrt(12).*param(:,2)./2;
                            Param(:,2) = param(:,1) + sqrt(12).*param(:,2)./2;
                            
                        case 'weibull'
                            options = optimset('Display','off');
                            par0    = [0.02,1.0e3];
                            for i = 1:n_cond
                                fun = @(par) sqrt(gamma(1+2/par)-(gamma(1+1/par)).^2)./gamma(1+1/par)-param(i,2)/param(i,1);
                                [xs,~,exitflag] = fzero(fun,par0,options);
                                if exitflag > 0
                                    Param(i,2) = xs;
                                    Param(i,1) = param(i,1)/gamma(1+1/Param(i,2));
                                else
                                    Param(i,:)= NaN;
                                end
                                if (Param(i,1) > 0) && (Param(i,2) > 0)
                                else
                                    Param(i,:)= NaN;
                                end
                            end
                        otherwise
                            disp('Error - distribution not available');
                    end
                case 'PAR'
                    Param = param;
                    if strcmpi(Obj.Name,'binomial')
                       if ~all(mod(Param(:,1),1) <= 1e-4)
                                error('Please select other moments.');
                            else
                                Param(:,1) = round(Param(:,1),0);
                       end
                    elseif strcmpi(Obj.Name,'truncatednormal')
                        Param(Param(:,3) >= Param(:,4),:) = NaN;
                    end
                otherwise
                    error('Input opt must be ''MOM'' or ''PAR''.')
            end
            
        end
        
        %------------------------------------------------------------------------------------------------------------------
        function CDF = condCDF(Obj,x,cond)
            % Evaluates the CDF of the conditional distribution at x for
            % the given conditions.
            % This method is used by the ERARosen method X2U.
   
            param = Obj.condParam(cond); % computation of the conditional parameters          
            
            % computation of CDF values (same as in ERADist)
            switch lower(Obj.Name)
                case 'beta'
                    CDF = betacdf((x-param(:,3))./(param(:,4)-param(:,3)),param(:,1),param(:,2));
                case 'binomial'
                    CDF = binocdf(x,param(:,1),param(:,2));
                case 'chisquare'
                    CDF = chi2cdf(x,param);
                case 'exponential'
                    CDF = expcdf(x,1./param(:,1));
                case 'frechet'
                    CDF = gevcdf(x,1./param(:,2),param(:,1)./param(:,2),param(:,1));
                case 'gamma'
                    CDF = gamcdf(x,param(:,2),1./param(:,1));
                case 'geometric'
                    CDF = geocdf(x-1,param);
                case 'gev'
                    CDF = gevcdf(x,param(:,1),param(:,2),param(:,3));
                case 'gevmin'
                    CDF = 1-gevcdf(-x,param(:,1),param(:,2),-param(:,3));
                case 'gumbel'
                    CDF = gevcdf(x,0,param(:,1),param(:,2));
                case 'gumbelmin'
                    CDF = 1-gevcdf(-x,0,param(:,1),-param(:,2));
                case 'lognormal'
                    CDF = logncdf(x,param(:,1),param(:,2));
                case 'negativebinomial'
                    CDF = nbincdf(x-param(:,1),param(:,1),param(:,2));
                case {'normal','gaussian'}
                    CDF = normcdf(x,param(:,1),param(:,2));
                case 'pareto'
                    CDF = gpcdf(x,1./param(:,2),param(:,1)./param(:,2),param(:,1));
                case 'poisson'
                    CDF = poisscdf(x,param(:,1).*param(:,2));
                case 'rayleigh'
                    CDF = raylcdf(x,param);
                case 'truncatednormal'
                    CDF = (param(:,3) <= x & x <= param(:,4)).*(normcdf(x,param(:,1),param(:,2))-normcdf(param(:,3),param(:,1),param(:,2)))./(normcdf(param(:,4),param(:,1),param(:,2))-normcdf(param(:,3),param(:,1),param(:,2)));
                case 'uniform'
                    CDF = unifcdf(x,param(:,1),param(:,2));
                case 'weibull'
                    CDF = wblcdf(x,param(:,1),param(:,2));
                otherwise
                    disp('Distribution type not available');
            end 
            
        end
        
        %------------------------------------------------------------------------------------------------------------------
        function InverseCDF = condiCDF(Obj,y,cond)
            % Evaluates the inverse CDF of the conditional distribution at
            % y for the given conditions.
            % This method is used by the ERARosen method U2X.
            
            param = Obj.condParam(cond);  % computation of the conditional parameters          
            
            % computation of CDF values (same as in ERADist)
            switch lower(Obj.Name)
                case 'beta'
                    InverseCDF = betainv(y,param(:,1),param(:,2)).*(param(:,4)-param(:,3))+param(:,3);
                case 'binomial'
                    InverseCDF = binoinv(y,param(:,1),param(:,2));
                case 'chisquare'
                    InverseCDF = chi2inv(y,param);
                case 'exponential'
                    InverseCDF = expinv(y,1./param);
                case 'frechet'
                    InverseCDF = gevinv(y,1./param(:,2),param(:,1)./param(:,2),param(:,1));
                case 'gamma'
                    InverseCDF = gaminv(y,param(:,2),1./param(:,1));
                case 'geometric'
                    InverseCDF = geoinv(y,param)+1;
                case 'gev'
                    InverseCDF = gevinv(y,param(:,1),param(:,2),param(:,3));
                case 'gevmin'
                    InverseCDF = -gevinv(1-y,param(:,1),param(:,2),-param(:,3));
                case 'gumbel'
                    InverseCDF = gevinv(y,0,param(:,1),param(:,2));
                case 'gumbelmin'
                    InverseCDF = -gevinv(1-y,0,param(:,1),-param(:,2));
                case 'lognormal'
                    InverseCDF = logninv(y,param(:,1),param(:,2));
                case 'negativebinomial'
                    InverseCDF = nbininv(y,param(:,1),param(:,2))+param(:,1);
                case {'normal','gaussian'}
                    InverseCDF = norminv(y,param(:,1),param(:,2));
                case 'pareto'
                    InverseCDF = gpinv(y,1./param(:,2),param(:,1)./param(:,2),param(:,1));
                case 'poisson'
                    InverseCDF = poissinv(y, param(:,1).*param(:,2));
                case 'rayleigh'
                    InverseCDF = raylinv(y,param);
                case 'truncatednormal'
                    InverseCDF = round(norminv(y.*(normcdf(param(:,4),param(:,1),param(:,2))-normcdf(param(:,3),param(:,1),param(:,2)))+normcdf(param(:,3),param(:,1),param(:,2)),param(:,1),param(:,2)),10);
                    InverseCDF(InverseCDF < param(:,3) | param(:,4) < InverseCDF) = NaN;
                case 'uniform'
                    InverseCDF = unifinv(y,param(:,1),param(:,2));
                case 'weibull'
                    InverseCDF = wblinv(y,param(:,1),param(:,2));
                otherwise
                    disp('Distribution type not available');
            end

        end
        
        %------------------------------------------------------------------------------------------------------------------
        function PDF = condPDF(Obj,x,cond)
            % Evaluates the PDF of the conditional distribution at x for
            % the given conditions.
            % This method is used by the ERARosen method pdf.
            
            param = Obj.condParam(cond);  % computation of the conditional parameters          
            
            % computation of CDF values (same as in ERADist)
            switch lower(Obj.Name)
                case 'beta'
                    PDF = betapdf((x-param(:,3))./(param(:,4)-param(:,3)),param(:,1),param(:,2))./(param(:,4)-param(:,3));
                case 'binomial'
                    PDF = binopdf(x,param(:,1),param(:,2));
                case 'chisquare'
                    PDF = chi2pdf(x,param);
                case 'exponential'
                    PDF = exppdf(x,1./param);
                case 'frechet'
                    PDF = gevpdf(x,1./param(:,2),param(:,1)./param(:,2),param(:,1));
                case 'gamma'
                    PDF = gampdf(x,param(:,2),1./param(:,1));
                case 'geometric'
                    PDF = geopdf(x-1,param);
                case 'gev'
                    PDF = gevpdf(x,param(:,1),param(:,2),param(:,3));
                case 'gevmin'
                    PDF = gevpdf(-x,param(:,1),param(:,2),-param(:,3));
                case 'gumbel'
                    PDF = gevpdf(x,0,param(:,1),param(:,2));
                case 'gumbelmin'
                    PDF = gevpdf(-x,0,param(:,1),-param(:,2));
                case 'lognormal'
                    PDF = lognpdf(x,param(:,1),param(:,2));
                case 'negativebinomial'
                    PDF = nbinpdf(x-param(:,1),param(:,1),param(:,2));
                case {'normal','gaussian'}
                    PDF = normpdf(x,param(:,1),param(:,2));
                case 'pareto'
                    PDF = gppdf(x,1./param(:,2),param(:,1)./param(:,2),param(:,1));
                case 'poisson'
                    PDF = poisspdf(x, param(:,1).*param(:,2));
                case 'rayleigh'
                    PDF = raylpdf(x,param);
                case 'truncatednormal'
                    PDF = (param(:,3) <= x & x <= param(:,4)).*normpdf(x,param(:,1),param(:,2))./(normcdf(param(:,4),param(:,1),param(:,2))-normcdf(param(:,3),param(:,1),param(:,2)));
                case 'uniform'
                    PDF = unifpdf(x,param(:,1),param(:,2));
                case 'weibull'
                    PDF = wblpdf(x,param(:,1),param(:,2));
                otherwise
                    disp('Distribution type not available');
            end
            
        end
        
        %------------------------------------------------------------------------------------------------------------------
        function Random = condRandom(Obj,cond)
            % Creates one random sample for each given condition.
            % This method is used by the ERARosen method random.
            
            param = Obj.condParam(cond); % computation of the conditional parameters          
            
            % creation of the random samples for the given conditions
            switch lower(Obj.Name)
                case 'beta'
                    Random = betarnd(param(:,1),param(:,2)).*(param(:,4)-param(:,3))+param(:,3);
                case 'binomial'
                    Random = binornd(param(:,1),param(:,2));
                case 'chisquare'
                    Random = chi2rnd(param);
                case 'exponential'
                    Random = exprnd(1./param);
                case 'frechet'
                    Random = gevrnd(1./param(:,2),param(:,1)./param(:,2),param(:,1));
                case 'gamma'
                    Random = gamrnd(param(:,2),1./param(:,1));
                case 'geometric'
                    Random = geornd(param)+1;
                case 'gev'
                    Random = gevrnd(param(:,1),param(:,2),param(:,3));
                case 'gevmin'
                    Random = -gevrnd(param(:,1),param(:,2),-param(:,3));
                case 'gumbel'
                    Random = gevrnd(0,param(:,1),param(:,2));
                case 'gumbelmin'
                    Random = -gevrnd(0,param(:,1),-param(:,2));
                case 'lognormal'
                    Random = lognrnd(param(:,1),param(:,2));
                case 'negativebinomial'
                    Random = nbinrnd(param(:,1),param(:,2))+param(:,1);
                case {'normal','gaussian'}
                    Random = normrnd(param(:,1),param(:,2));
                case 'pareto'
                    Random = gprnd(1./param(:,2),param(:,1)./param(:,2),param(:,1));
                case 'poisson'
                    Random = poissrnd(param(:,1).*param(:,2));
                case 'rayleigh'
                    Random = raylrnd(param);
                case 'truncatednormal'
                    n = size(param,1);
                    u = rand(n,1);
                    Random = norminv(u.*(normcdf(param(:,4),param(:,1),param(:,2))-normcdf(param(:,3),param(:,1),param(:,2)))+normcdf(param(:,3),param(:,1),param(:,2)),param(:,1),param(:,2));
                case 'uniform'
                    %n = size(param,1);
                    Random=random('uniform',param(:,1),param(:,2));
                    %random = rand(n,1).*(param(:,2)-param(:,1))+param(:,1);
                case 'weibull'
                    Random = wblrnd(param(:,1),param(:,2));
                otherwise
                    disp('Error - distribution not available');
            end 
            
        end
            
        end
end
