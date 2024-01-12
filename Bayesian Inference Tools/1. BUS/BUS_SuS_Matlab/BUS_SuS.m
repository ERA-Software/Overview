function [h,samplesU,samplesX,logcE,sigma] = BUS_SuS(N, p0, c, log_likelihood, distr)
%% Subset simulation function adapted for BUS
%{
---------------------------------------------------------------------------
Created by:
Fong-Lin
Felipe Uribe
Luca Sardi
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
* Update LSF with log_likelihood, while loop and clean up code
Version 2018-05
* Organizing the code
Version 2017-10
* Fixing a bug in the computation of the posterior samples
Version 2017-04
* distr is now an input
---------------------------------------------------------------------------
Input:
* N               : number of samples per level
* p0              : conditional probability of each subset
* c               : scaling constant of the BUS method
* log_likelihood  : log likelihood function of the problem at hand
* distr           : Nataf distribution object (probabilistic transformation)
---------------------------------------------------------------------------
Output:
* h        : intermediate levels of the subset simulation method
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* logcE    : model log-evidence
* sigma    : final spread of the proposal for each parameter
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SubSim"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
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
    n   = length(distr.Marginals)+1;    % number of random variables + p of BUS
    u2x = @(u) distr.U2X(u);            % from u to x
    
else   % use distribution information for the transformation (independence)
    % Here we are assuming that all the parameters have the same distribution !!!
    % Adjust accordingly otherwise or use ERANataf object
    n   = length(distr)+1;                  % number of random variables + p of BUS
    u2x = @(u) distr(1).icdf(normcdf(u));   % from u to x
end

%% limit state function in the standard space
%h_LSF = @(u) normcdf(u(end)) - c*likelihood(u2x(u(1:end-1)));
%h_LSF = @(u) u(end) - norminv(c*likelihood(u2x(u(1:end-1))));
ell   = log(1/c);
h_LSF = @(u) log(normcdf(u(end))) + ell - log_likelihood(u2x(u(1:end-1)));

%% Initialization of variables
i      = 1;                   % number of conditional level
lambda = 0.6;                 % initial scaling parameter \in (0,1)
max_it = 20;                  % maximum number of iterations

% Memory allocation
samplesU.total = cell(max_it,1);   % space for the samples in the standard space
samplesU.seeds = cell(max_it,1);   % space for the samples in the standard space
gsort = zeros(max_it,N);            % space for the sorted LSF evaluations
h     = zeros(max_it,1);            % space for the intermediate levels
prob  = zeros(max_it,1);            % space for the failure probability at each level

%% BUS-SuS procedure
% initial MCS step
fprintf('Initial BUS-MCS step...\t')
u_j   = randn(n,N)';        % samples in the standard space N,n
geval = zeros(1,N);        % space for the LSF evaluations
for j = 1:N
    geval(j) = h_LSF(u_j(j,:));    % limit state function in standard (Ref. 2 Eq. 21)
end

% SuS stage
fprintf('\nBUS-SuS steps...\n')
h(i) = inf;
while (h(i) > 0) && (i < max_it)
    % next level
    i = i+1;
    
    % sort values in ascending order
    [gsort(i-1,:), idx] = sort(geval);
    
    % order the samples according to idx
    u_j_sort            = u_j(idx,:);
    samplesU.total{i-1} = u_j_sort;   % store the ordered samples
    
    % intermediate level
    h(i) = prctile(geval, p0*100);
    
    % number of failure points in the next level
    nF = sum(geval <= max(h(i),0));
    
    % assign conditional probability to the level
    if h(i) <= 0
        h(i)      = 0;
        prob(i-1) = nF/N;
    else
        prob(i-1) = p0;
    end
    fprintf('\n-Threshold intermediate level %g = %g \n', i-1, h(i));
    
    % select seeds and randomize the ordering (to avoid bias)
    seeds     = u_j_sort(1:nF,:);
    idx_rnd   = randperm(nF);
    rnd_seeds = seeds(idx_rnd,:);      % non-ordered seeds
    samplesU.seeds{i-1} = seeds;       % store ordered level seeds
    
    % sampling process using adaptive conditional sampling
    [u_j, geval, lambda, sigma, acc] = aCS(N, lambda, h(i), rnd_seeds, h_LSF);
    fprintf('\t*aCS lambda = %g \t*aCS sigma = %g \t *aCS accrate = %g\n', lambda, sigma(1), acc);
end
m = i;
samplesU.total{i} = u_j;  % store final posterior samples (non-ordered)

% delete unnecesary data
samplesU.total(m+1:end)  = [];  
samplesU.seeds(m+1:end)  = [];  
prob(m:end)              = [];
h(m+1:end)               = [];

%% acceptance probability and evidence
log_p_acc = sum(log(prob));
logcE    = log_p_acc-log(c);

%% transform the samples to the physical (original) space
samplesX = cell(m,1);
for j = 1:m
    pp          = normcdf(samplesU.total{j}(:,end));
    samplesX{j} = [u2x(samplesU.total{j}(:,1:end-1)), pp];
end

return;