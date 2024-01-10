function [Pr, lv, N_tot, samplesU, samplesX, S_F1] = ...
    iCE_SG(N, g_fun, distr, max_it, CV_target, sensitivity_analysis, samples_return)
%% Cross entropy-based importance sampling with Single Gaussian distribution
%{
---------------------------------------------------------------------------
Improved cross entropy-based importance sampling with Single Gaussian
---------------------------------------------------------------------------
Created by:
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Fong-Lin Wu
Matthias Willer
Peter Kaplan
Daniel Koutas

Engineering Risk Analysis Group
Technische Universitaet Muenchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2022-04
* Inclusion of sensitivity analysis
---------------------------------------------------------------------------
Comments:
* Adopt draft scripts from Sebastian and reconstruct the code to comply
  with the style of the published codes
* W is the original importance weight (likelihood ratio)
* W_t is the transitional importance weight of samples
* W_approx is the weight (ratio) between real and approximated indicator functions
---------------------------------------------------------------------------
Input:
* N                    : number of samples per level
* g_fun                : limit state function
* max_it               : maximum number of iterations
* distr                : Nataf distribution object or marginal distribution object of the input variables
* CV_target            : taeget correlation of variation of weights
* sensitivity_analysis : implementation of sensitivity analysis: 1 - perform, 0 - not perform
* samples_return       : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr        : probability of failure
* lv        : total number of levels
* N_tot     : total number of samples
* samplesU  : object with the samples in the standard normal space
* samplesX  : object with the samples in the original space
* S_F1      : vector of first order Sobol' indices
---------------------------------------------------------------------------
Based on:
1. Papaioannou, I., Geyer, S., & Straub, D. (2019).
   Improved cross entropy-based importance sampling with a flexible mixture model.
   Reliability Engineering & System Safety, 191, 106564
2. Geyer, S., Papaioannou, I., & Straub, D. (2019).
   Cross entropy-based importance sampling using Gaussian densities revisited. 
   Structural Safety, 76, 15–27
---------------------------------------------------------------------------
%}

%% transform to the standard Gaussian space
if any(strcmp('Marginals',fieldnames(distr))) == 1   % use Nataf transform (dependence)
    dim = length(distr.Marginals);    % number of random variables (dimension)
    u2x = @(u) distr.U2X(u);          % from u to x
    
else   % use distribution information for the transformation (independence)
    % Here we are assuming that all the parameters have the same distribution !!!
    % Adjust accordingly otherwise
    dim = length(distr);                    % number of random variables (dimension)
    u2x = @(u) distr(1).icdf(normcdf(u));   % from u to x
end

%% LSF in standard space
G_LSF = @(u) g_fun(u2x(u));

%% Initialization of variables and storage
N_tot  = 0;        % total number of samples

% Definition of initial parameters of the random variables (uncorrelated standard normal)
mu_init   = zeros(1,dim);        % Initial Gaussian mean
si_init   = eye(dim);            % Initial Gaussian standard deviation
sigma_t   = zeros(max_it,1);     % squared difference between COV and target COV
samplesU  = cell(1,1);           % space for the samples in the standard space

%% iCE procedure
% Initializing parameters
mu_hat       = mu_init;
si_hat       = si_init;

for j = 1:max_it
    % generate samples
    X = mvnrnd(mu_hat', si_hat, N);
        
    % count generated samples
    N_tot = N_tot+N;
    
    % evaluation of the limit state function
    geval = G_LSF(X);
    
    % initialize sigma_0
    if j==1,    sigma_t(1) = 10*mean(geval);    end
    
    % calculating h for the likelihood ratio
    h = mvnpdf(X, mu_hat, si_hat);
    
    % likelihood ratio
    W = mvnpdf(X,zeros(1,dim),eye(dim))./h;
    
    % indicator function
    I = (geval <= 0);
    
    % Samples return - all / all by default
    if ~ismember(samples_return, [0 1])
        samplesU{j} = X;
    end

    % check convergence
    % transitional weight W_t=I*W when sigma_t approches 0 (smooth approximation:)
    W_approx = I ./ approx_normCDF(-geval/sigma_t(j)); % weight of indicator approximations
    %Cov_x   = std(I.*W) / mean(I.*W);                 % poorer numerical stability
    Cov_x    = std(W_approx) / mean(W_approx);
    if Cov_x <= CV_target
        % Samples return - last
        if (samples_return == 1) || (samples_return == 0 && sensitivity_analysis == 1)
            samplesU{1} = X;
        end
        break;
    end
    
    % compute sigma and weights for distribution fitting
    % minimize COV of W_t (W_t=normalCDF*W)
    fmin      = @(x) abs(std(approx_normCDF(-geval/x) .* W) / mean(approx_normCDF(-geval/x).*W) - CV_target);
    sigma_new = fminbnd(fmin,0,sigma_t(j));
    sigma_t(j+1) = sigma_new;
    
    % update W_t
    W_t = approx_normCDF(-geval/sigma_t(j+1)).*W;
    
    % parameter update: closed-form update
    mu_hat = (W_t'*X) ./ sum(W_t);
    Xo     = bsxfun(@times, X - mu_hat, sqrt(W_t));
    si_hat = Xo'*Xo/sum(W_t) + 1e-6*eye(dim);
    
end

% Samples return - all by default message
if ~ismember(samples_return, [0 1 2])
    fprintf('\n-Invalid input for samples return, all samples are returned by default \n');
end

% store the needed steps
lv=j;

%% Calculation of the Probability of failure
Pr      = 1/N*sum(W(I));

%% transform the samples to the physical/original space
samplesX = cell(length(samplesU),1);
if (samples_return ~= 0) || (samples_return == 0 && sensitivity_analysis == 1)
	for m = 1:length(samplesU)
		samplesX{m} = u2x(samplesU{m});
	end
end

%% sensitivity analysis
if sensitivity_analysis == 1
    % resample 1e4 failure samples with final weights W
    weight_id = randsample(find(I),1e4,'true',W(I));
    f_s = samplesX{end}(weight_id,:);
    
    if size(f_s,1) == 0
        fprintf("\n-Sensitivity analysis could not be performed, because no failure samples are available \n")
        S_F1 = [];
    else
        [S_F1, exitflag, errormsg] = Sim_Sobol_indices(f_s, Pr, distr);
        if exitflag == 1
            fprintf("\n-First order indices: \n");
            disp(S_F1);
        else
            fprintf('\n-Sensitivity analysis could not be performed, because: \n')
            fprintf(errormsg);
        end
    end
	if samples_return == 0
        samplesU = cell(1,1);  % empty return samples U
        samplesX = cell(1,1);  % and X
    end
else 
    S_F1 = [];
end

% Convergence is not achieved message
if j == max_it
    fprintf('-Exit with no convergence at max iterations \n\n');
end

return;

function p = approx_normCDF(x)

% Returns an approximation for the standard normal CDF based on a
% polynomial fit of degree 9

erfun=zeros(size(x));

idpos=x>0;
idneg=x<0;

t=(1+0.5*abs(x/sqrt(2))).^-1;

tau=t.*exp(-(x/sqrt(2)).^2-1.26551223+1.0000236*t+0.37409196*t.^2+0.09678418*t.^3-0.18628806*t.^4+0.27886807*t.^5-1.13520398*t.^6+1.48851587*t.^7-0.82215223*t.^8+0.17087277*t.^9);
erfun(idpos)=1-tau(idpos);
erfun(idneg)=tau(idneg)-1;

p=0.5*(1+erfun);

%%END