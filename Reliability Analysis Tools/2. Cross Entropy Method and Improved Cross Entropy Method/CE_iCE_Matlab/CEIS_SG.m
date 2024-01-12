function [Pr, lv, N_tot, gamma_hat, samplesU, samplesX, k_fin, S_F1] = CEIS_SG(N, p, g_fun, distr, sensitivity_analysis, samples_return)
%% Cross entropy-based importance sampling with Single Gaussian distribution
%{
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
Matthias Willer
Fong-Lin Wu
Daniel Koutas

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2022-04
* Inclusion of sensitivity analysis
---------------------------------------------------------------------------
Comments:
* Remove redundant dimension adjustment of limit state function. It should
  be restricted in the main script
* The CE method in combination with a Gaussian Mixture model can only be
  applied for low-dimensional problems, since its accuracy decreases
  dramatically in high dimensions.
* General convergence issues can be observed with linear LSFs.
---------------------------------------------------------------------------
Input:
* N                    : number of samples per level
* p                    : quantile value to select samples for parameter update
* g_fun                : limit state function
* distr                : Nataf distribution object or
                         marginal distribution object of the input variables
* sensitivity_analysis : implementation of sensitivity analysis: 1 - perform, 0 - not perform
* samples_return       : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr        : probability of failure
* lv        : total number of levels
* N_tot     : total number of samples
* gamma_hat : intermediate levels
* samplesU  : object with the samples in the standard normal space
* samplesX  : object with the samples in the original space
* k_fin     : final number of Gaussians in the mixture (for SG k_fin = 1)
* S_F1      : vector of first order Sobol' indices
---------------------------------------------------------------------------
Based on:
1."Cross entropy-based importance sampling using Gaussian densities revisited"
   Geyer et al.
   To appear in Structural Safety
2."A new flexible mixture model for cross entropy based importance sampling".
   Papaioannou et al. (2018)
   In preparation.
---------------------------------------------------------------------------
%}
if (N*p ~= fix(N*p)) || (1/p ~= fix(1/p))
   error('N*p and 1/p must be positive integers. Adjust N and p accordingly');
end

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
max_it = 50;     % maximum number of iterations
N_tot  = 0;      % total number of samples

% Definition of initial parameters of the random variables (uncorrelated standard normal)
mu_init   = zeros(1,dim);   
Si_init   = eye(dim);       
gamma_hat = zeros(max_it+1,1);   % space for intermediate failure thresholds
samplesU  = cell(1,1);           % space for the samples in the standard space

%% CE procedure
% initializing parameters
gamma_hat(1) = 1;
mu_hat       = mu_init;
Si_hat       = Si_init;

% Iterations
for j = 1:max_it
   % generate samples
   X = mvnrnd(mu_hat, Si_hat, N);
      
   % count generated samples
   N_tot = N_tot+N;
   
   % evaluation of the limit state function
   geval = G_LSF(X);
   
   % calculating h for the likelihood ratio
   h = mvnpdf(X, mu_hat, Si_hat);
   
   % Samples return - all / all by default
    if ~ismember(samples_return, [0 1])
        samplesU{j} = X;
    end

    % Check convergence
    if gamma_hat(j) == 0
        % Samples return - last
        if (samples_return == 1) || (samples_return == 0 && sensitivity_analysis == 1)
            samplesU{1} = X;
        end
        break;
    end
   
   % obtaining estimator gamma
   gamma_hat(j+1) = max(0, prctile(geval, p*100));
   fprintf('\nIntermediate threshold: %g\n',gamma_hat(j+1));
   
   % indicator function
   I = (geval <= gamma_hat(j+1));
   
   % likelihood ratio
   W = mvnpdf(X,zeros(1,dim),eye(dim))./h;
   
   % parameter update: closed-form update
   mu_hat = (W(I)'*X(I,:))./sum(W(I));
   Xo     = bsxfun(@times, X(I,:) - mu_hat, sqrt(W(I)));
   Si_hat = Xo'*Xo/sum(W(I))+1e-6*eye(dim);
end

% Samples return - all by default message
if ~ismember(samples_return, [0 1 2])
    fprintf('\n-Invalid input for samples return, all samples are returned by default \n');
end

% required steps
lv = j;
gamma_hat(lv+1:end) = [];

%% Fix and remove on 03.2020
% adjust the dimension
% [mm,nn] = size(geval);
% if mm > nn
%    geval = geval';
% end

%% Calculation of the Probability of failure
W_final = mvnpdf(X, zeros(1,dim), eye(dim))./h;
I_final = geval<=0;
%Pr      = 1/N*sum(I_final*W_final);
Pr      = 1/N*sum(I_final.*W_final);
k_fin   = 1;

%% transform the samples to the physical/original space
samplesX = cell(length(samplesU),1);
if (samples_return ~= 0) || (samples_return == 0 && sensitivity_analysis == 1)
	for i = 1:length(samplesU)
	   samplesX{i} = u2x(samplesU{i});
	end
end

%% sensitivity analysis
if sensitivity_analysis == 1
    % resample 1e4 failure samples with final weights W
    weight_id = randsample(find(I_final),1e4,'true',W_final(I_final));
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
%%END