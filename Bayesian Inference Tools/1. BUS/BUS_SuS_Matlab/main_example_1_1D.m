%% example 1: 1D posterior
%{
---------------------------------------------------------------------------
By:                Date:          Topic:
Fong-Lin Wu        July.2019      
---------------------------------------------------------------------------
Current version 2021-03
* use of log-evidence logcE
---------------------------------------------------------------------------
References:
1."Bayesian inference with subset simulation: strategies and improvements"
   Wolfgang Betz et al.
   Computer Methods in Applied Mechanics and Engineering 331 (2018) 72-93.
---------------------------------------------------------------------------
%}
clear; clc; close all; 

% add path to ERADist and ERANataf classes
% Source: https://www.bgu.tum.de/era/software/eradist/
% addpath('../../../ERA_Dist/ERADistNataf_MATLAB/')

%% initial data
d = 1;      % number of dimensions (number of uncertain parameters)

%% prior PDF for theta
prior_pdf = ERADist('normal','PAR',[0,1]);

%% likelihood PDF (mixture of gaussians in 2D)
s = RandStream.create('mt19937ar','seed',12001);   % fixed the seed to get the same data points
RandStream.setGlobalStream(s);
%
mu             = 5;
sigma          = 0.2;
likelihood     = @(theta) normpdf(theta,mu,sigma);
log_likelihood = @(theta) log(likelihood(theta)+realmin);   % realmin to avoid Inf values in log(0)

%% find scale constant c
method = 1;
switch method
   % use MLE to find c
   case 1  
      f_start = log(mu);
      fun     = @(lnF) -log(likelihood(exp(lnF))+realmin);
      options = optimset('MaxIter',1e7,'MaxFunEvals',1e7);
      MLE_ln  = fminsearch(fun,f_start,options);
      MLE     = exp(MLE_ln);   % likelihood(MLE) = 1
      c       = 1/likelihood(MLE);
      
   % some likelihood evaluations to find c
   case 2
      K  = 5e3;                  % number of samples      
      u  = randn(d,K);           % samples in standard space
      x  = u;       % samples in physical space
      
      % likelihood function evaluation
      L_eval = zeros(K,1);
      for i = 1:K
         L_eval(i) = likelihood(x(i));
      end
      c = 1/max(L_eval);    % Ref. 1 Eq. 34
      
   % use approximation to find c
   case 3
      fprintf('This method requires a large number of measurements\n');
      m = length(mu);
      p = 0.05;
      c = 1/exp(-0.5*chi2inv(p,m));    % Ref. 1 Eq. 38
      
   otherwise
      error('Finding the scale constant c requires -method- 1, 2 or 3');
end

%% BUS-SuS step
N  = 3e3;    % number of samples per level
p0 = 0.1;    % probability of each subset

fprintf('\nBUS with SUBSET SIMULATION: \n');
[h, samplesU, samplesX, logcE, sigma] = BUS_SuS(N, p0, c, log_likelihood, prior_pdf);

%% extract the samples
nsub = length(h);
u1p  = cell(nsub,1);  u0p = cell(nsub,1);
x1p  = cell(nsub,1);  pp  = cell(nsub,1);
for i = 1:nsub
   % samples in standard
   u1p{i} = samplesU.total{i}(:,1);             
   u0p{i} = samplesU.total{i}(:,2); 
   % samples in physical
   x1p{i} = samplesX{i}(:,1);
   pp{i}  = samplesX{i}(:,2);
end

%% reference and BUS solutions
mu_exact    = 4.81;      % for x_1
sigma_exact = 0.196;     % for x_1
cE_exact    = 2.36e-6;

% show results
fprintf('\nExact model evidence = %g', cE_exact);
fprintf('\nModel evidence BUS-SuS = %g\n', exp(logcE));
fprintf('\nExact posterior mean = %g', mu_exact);
fprintf('\nMean value of samples = %g\n', mean(x1p{end}));
fprintf('\nExact posterior std = %g', sigma_exact);
fprintf('\nStd of samples = %g\n\n', std(x1p{end}));
%%END