function [Pr, lv, N_tot, gamma_hat, samplesU, samplesX, k_fin, W_final, f_s_iid] = ...
    CEIS_GM(N, p , g_fun, distr, k_init, samples_return)
%% Cross entropy-based importance sampling with Gaussian Mixture
%{
---------------------------------------------------------------------------
Created by:
Sebastian Geyer
Matthias Willer
Fong-Lin Wu
Daniel Koutas
Ivan Olarte-Rodriguez

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2023-08
* Generation of i.i.d. samples for Sensitivity Analysis
* Modified calls to LSF function to handle non-vectorized defintions
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
* k_init               : initial number of Gaussians in the mixture model
* samples_return       : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr        : probability of failure
* lv        : total number of levels
* N_tot     : total number of samples
* gamma_hat : intermediate levels
* samplesU  : object with the samples in the standard normal space
* samplesX  : object with the samples in the original space
* k_fin     : final number of Gaussians in the mixture
* W_final   : final weights
* f_s       : i.i.d failure samples
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
    u2x = @(u) reshape(distr.U2X(u),[],dim);          % from u to x
    
else   % use distribution information for the transformation (independence)
    % Here we are assuming that all the parameters have the same distribution !!!
    % Adjust accordingly otherwise
    dim = length(distr);                    % number of random variables (dimension)
    u2x = @(u) reshape(distr(1).icdf(normcdf(u)),[],dim);   % from u to x
end

%% LSF in standard space
G_LSF = @(u) g_fun(u2x(u));

%% Initialization of variables and storage
% j      = 0;        % initial level
max_it = 50;       % maximum number of iterations
N_tot  = 0;        % total number of samples
k      = k_init;   % number of Gaussians in mixture

% Definition of parameters of the random variables (uncorrelated standard normal)
mu_init   = zeros(1,dim);
Si_init   = eye(dim);
Pi_init   = 1;
gamma_hat = zeros(max_it+1,1);   % space for intermediate failure thresholds
samplesU  = cell(1,1);           % space for the samples in the standard space

%% CE procedure
% initializing parameters
gamma_hat(1) = 1;
mu_hat       = mu_init;
Si_hat       = Si_init;
Pi_hat       = Pi_init;

% Iteration
for j = 1:max_it
 
    % Generate samples
    X = GM_sample(mu_hat, Si_hat, Pi_hat, N);
    
    % Count generated samples
    N_tot = N_tot+N;
    
    % evaluation of the limit state function
    geval = zeros(size(X,1),1);
    for ii = 1:numel(geval)
       geval(ii) = G_LSF(X(ii,:));
    end
    
    % Calculating h for the likelihood ratio
    h = h_calc(X,mu_hat,Si_hat,Pi_hat);
      
    % Samples return - all / all by default
    if ~ismember(samples_return, [0 1])
        samplesU{j} = X;
    end

    % Check convergence
    if gamma_hat(j) == 0
        % Samples return - last
        if (samples_return == 1) 
            samplesU{1} = X;
        end
        break;
    end
      
    % obtaining estimator gamma
    gamma_hat(j+1) = max(0, prctile(geval, p*100));
    fprintf('\nIntermediate threshold: %g\n',gamma_hat(j+1));
    
    % Indicator function
    I = (geval <= gamma_hat(j+1));
    
    % Likelihood ratio
    W = mvnpdf(X,zeros(1,dim),eye(dim))./h;
    
    % Parameter update
    [mu, si, pi] = EMGM(X(I,:)',W(I),k_init);
    
    % Assigning the variables with updated parameters
    mu_hat = mu';
    Si_hat = si;
    Pi_hat = pi';
    k      = length(pi);

end

% Samples return - all by default message
if ~ismember(samples_return, [0 1 2])
    fprintf('\n-Invalid input for samples return, all samples are returned by default \n');
end

% needed steps
lv     = j;
k_fin = k;
gamma_hat(lv+1:end) = [];

%% Fix and remove on 03.2020
% adjust the dimension
% [mm,nn] = size(geval);
% if mm > nn
%     geval = geval';
% end

%% Calculation of the Probability of failure
W_final = mvnpdf(X, zeros(1,dim), eye(dim))./h;
I_final = (geval <=0 );
Pr      = 1/N*sum(I_final.*W_final);

%% Transform the samples to the physical/original space
samplesX = cell(length(samplesU),1);
f_s_iid = [];
if (samples_return ~= 0) 
	for m = 1:length(samplesU)
        if ~isempty(samplesU{m})
            samplesX{m} = u2x(samplesU{m});
        end
    end

    %% Output for Sensitivity Analysis

    % resample 1e4 failure samples with final weights W
    weight_id = randsample(find(I_final),1e4,'true',W_final(I_final));
    if ~isempty(samplesX{end})
        f_s_iid = samplesX{end}(weight_id,:);
    end
end


%% Error Messages
% Convergence is not achieved message
if j == max_it
    fprintf('-Exit with no convergence at max iterations \n\n');
end

return;


%==========================================================================
%===========================NESTED FUNCTIONS===============================
%==========================================================================
function X = GM_sample(mu,Si,Pi,N)
% Algorithm to draw samples from a Gaussian-Mixture (GM) distribution
%{
---------------------------------------------------------------------------
Input:
* mu : [npi x d]-array of means of Gaussians in the Mixture
* Si : [d x d x npi]-array of cov-matrices of Gaussians in the Mixture
* Pi : [npi]-array of weights of Gaussians in the Mixture (sum(Pi) = 1)
* N  : number of samples to draw from the GM distribution
---------------------------------------------------------------------------
Output:
* X  : samples from the GM distribution
---------------------------------------------------------------------------
%}

if size(mu,1) == 1
    X = mvnrnd(mu,Si,N);
else
    % Determine number of samples from each distribution
    
    ind = randsample(size(mu,1),N,true,Pi);
    z = histcounts(ind,[(1:size(mu,1)) size(mu,1)+1]);
    
    %     z = round(Pi*N);
    %     if sum(z) ~= N
    %         dif     = sum(z)-N;
    %         [~,ind] = max(z);
    %         z(ind)  = z(ind)-dif;
    %     end
    % Generate samples
    X   = zeros(N,size(mu,2));
    ind = 1;
    for i = 1:size(Pi,1)
        np                = z(i);
        X(ind:ind+np-1,:) = mvnrnd(mu(i,:),Si(:,:,i),np);
        ind               = ind+np;
    end
end
return;


%===========================================================================
function h = h_calc(X, mu, Si, Pi)
% Basic algorithm to calculate h for the likelihood ratio
%{
---------------------------------------------------------------------------
Input:
* X  : input samples
* mu : [npi x d]-array of means of Gaussians in the Mixture
* Si : [d x d x npi]-array of cov-matrices of Gaussians in the Mixture
* Pi : [npi]-array of weights of Gaussians in the Mixture (sum(Pi) = 1)
---------------------------------------------------------------------------
Output:
* h  : parameters h (IS density)
---------------------------------------------------------------------------
%}

N = size(X,1);
if size(Pi,1) == 1
    h = mvnpdf(X,mu,Si);
else
    h_pre = zeros(N,size(Pi,1));
    for q = 1:size(Pi,1)
        h_pre(:,q) = Pi(q)*mvnpdf(X,mu(q,:),Si(:,:,q));
    end
    h = sum(h_pre,2);
end
return;
%%END