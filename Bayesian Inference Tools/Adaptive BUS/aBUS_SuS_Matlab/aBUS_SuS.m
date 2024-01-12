function [h, samplesU, samplesX, logcE, c, sigma] = aBUS_SuS(N, p0, log_likelihood, distr)
%% adaptive BUS
%{
---------------------------------------------------------------------------
Created by:
Fong-Lin Wu
Luca Sardi
Felipe Uribe
Iason Papaioannou (iason.papaioannou@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Current version 2021-03
* use of log-evidence logcE
---------------------------------------------------------------------------
Version 2020-10
* Adaptation to new ERANataf class
Version 2019-06
* Update LSF with log_likelihood and clean up code
Version 2018-05
* minor modifications
Version 2018-01
* Bug fix in the computation of the LSF
* Notation of the paper Ref.1
* Bug fix in the computation of the LSF
* aCS_BUS.m function modified for Bayesian updating
Version 2017-07
* Bug fixes
Version 2017-04
* T_nataf is now an input
---------------------------------------------------------------------------
Input:
* N              : number of samples per level
* p0             : conditional probability of each subset
* log_likelihood : log-Likelihood function of the problem at hand
* T_nataf        : Nataf distribution object (probabilistic transformation)
---------------------------------------------------------------------------
Output:
* h        : intermediate levels of the subset simulation method
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* logcE    : model log-evidence
* c        : scaling constant that holds 1/c >= Lmax
* sigma    : final spread of the proposal
---------------------------------------------------------------------------
Based on:
1."Bayesian inference with subset simulation: strategies and improvements"
   Betz et al.
   Computer Methods in Applied Mechanics and Engineering. Accepted (2017)
2."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2015) 1-13.
---------------------------------------------------------------------------
%}
if (N*p0 ~= fix(N*p0)) || (1/p0 ~= fix(1/p0))
    error('N*p0 and 1/p0 must be positive integers. Adjust N and p0 accordingly');
end

%% transform to the standard Gaussian space
if any(strcmp('Marginals',fieldnames(distr))) == 1   % use Nataf transform (dependence)
    n   = length(distr.Marginals)+1;    % number of random variables (dimension)
    u2x = @(u) distr.U2X(u);            % from u to x
    
else   % use distribution information for the transformation (independence)
    % Here we are assuming that all the parameters have the same distribution !!!
    % Adjust accordingly otherwise or use an ERANataf object
    n   = length(distr)+1;                    % number of random variables (dimension)
    u2x = @(u) distr(1).icdf(normcdf(u));     % from u to x
end

%% limit state funtion for the observation event (Ref.1 Eq.12)
% log likelihood in standard space
loglike_fun = @(u) log_likelihood(u2x(u(1:end-1)));

% LSF
% h_LSF = log(pi) + log_l - log_Likelihood;
h_LSF = @(u, log_l, log_Likelihood) log(normcdf(u)) + log_l - log_Likelihood;

%% Initialization of variables
i      = 1;                    % number of conditional level
lambda = 0.6;                  % initial scaling parameter \in (0,1)
max_it = 20;                   % maximum number of iterations

% Memory allocation
samplesU  = cell(max_it,1);    % space for the samples in the standard space
log_leval = zeros(1,N);        % space for the log-likelihood evaluations
h         = zeros(max_it,1);   % space for the intermediate leveles
prob      = zeros(max_it,1);   % space for the failure probability at each level
nF        = zeros(max_it,1);   % space for the number of failure point per level

%% aBUS-SuS procedure
% initial log-likelihood function evaluations
u_j = randn(n,N);   % N samples from the prior distribution
fprintf('Evaluating log-likelihood function...\t');
for j = 1:N
    log_leval(j) = loglike_fun(u_j(:,j));   % evaluate likelihood
end
logl_hat = max(log_leval);   % =-log(c) (Ref.1 Alg.5 Part.3)
fprintf('Done!\n');
fprintf('Initial maximum log-likelihood: %g\n', logl_hat);

% SuS stage
h(i) = Inf;
while h(i) > 0  && (i < max_it)
    % increase counter
    i = i+1;
    
    % compute the limit state function (Ref.1 Eq.12)
    geval = h_LSF(u_j(end,:), logl_hat, log_leval);   % evaluate LSF (Ref.1 Eq.12)
    
    % sort values in ascending order
    [~,idx] = sort(geval);
    
    % order the samples according to idx
    u_j_sort      = u_j(:,idx)';
    samplesU{i-1} = u_j_sort;     % store the ordered samples
    
    % intermediate level
    h(i) = prctile(geval,p0*100);
    
    % number of failure points
    nF(i) = sum(geval <= max(h(i),0));
    
    % assign conditional probability to the level
    if h(i) < 0
        h(i)      = 0;
        prob(i-1) = nF(i)/N;
    else
        prob(i-1) = p0;
    end
    fprintf('\n-Threshold level %g = %g\n', i-1, h(i));
    
    % randomize the ordering of the samples (to avoid possible bias)
    seeds    = u_j_sort(1:nF(i),:);
    idx_rnd  = randperm(nF(i));
    rnd_seed = seeds(idx_rnd,:);      % non-ordered seeds
    
    % sampling process using adaptive conditional sampling (Ref.1 Alg.5 Part.4c)
    [u_j, log_leval, lambda, sigma, acc] = aCS_aBUS(N, lambda, h(i), rnd_seed, loglike_fun, logl_hat, h_LSF);
    fprintf('\t*aCS lambda = %g \t*aCS sigma = %g \t *aCS accrate = %g\n', lambda, sigma(1), acc);
    
    % update the value of the scaling constant (Ref.1 Alg.5 Part.4d)
    l_new    = max(logl_hat, max(log_leval));
    h(i)     = h(i) - logl_hat + l_new;
    logl_hat = l_new;
    fprintf(' Modified threshold level %g = %g\n', i-1, h(i));
    
    % decrease the dependence of the samples (Ref.1 Alg.5 Part.4e)
    p          = unifrnd( zeros(1,N), min(ones(1,N), exp(log_leval -logl_hat + h(i))) );
    u_j(end,:) = norminv(p);   % to the standard space
end

% number of intermediate levels
m = i;

% store final posterior samples
samplesU{m} = u_j';

% delete unnecesary data
samplesU(m+1:end) = [];
h(m+1:end)        = [];
prob(m:end)       = [];

%% acceptance probability and model evidence (Ref.1 Alg.5 Part.6and7)
log_p_acc = sum(log(prob));
c     = 1/exp(logl_hat);
logcE  = log_p_acc+logl_hat; 

%% transform the samples to the physical/original space
samplesX = cell(m,1);
for j = 1:m
    p           = normcdf(samplesU{j}(:,end));
    samplesX{j} = [u2x(samplesU{j}(:,1:end-1)), p];
end

return;
