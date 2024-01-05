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
prior_pdf = ERANataf(prior_pdf, 1);

%% likelihood PDF (mixture of gaussians in 2D)
s = RandStream.create('mt19937ar','seed',12001);   % fixed the seed to get the same data points
RandStream.setGlobalStream(s);
%
mu             = 5;
sigma          = 0.2;
likelihood     = @(theta) normpdf(theta,mu,sigma);
log_likelihood = @(theta) log(likelihood(theta)+realmin);   % realmin to avoid Inf values in log(0)

%% iTMCMC
Ns = 1e3;        % number of samples per level
Nb = 0.1*Ns;     % burn-in period

fprintf('\niTMCMC: \n');
[samplesU, samplesX, q, logcE] = iTMCMC(Ns, Nb, log_likelihood, prior_pdf);

%% extract the samples
m = length(q);   % number of stages (intermediate levels)
u1p  = cell(m,1);   u0p = cell(m,1);
x1p  = cell(m,1);   pp  = cell(m,1);
for i = 1:m
   % samples in standard
   u1p{i} = samplesU{i}(:,1);
   % samples in physical
   x1p{i} = samplesX{i}(:,1);   
end

%% reference and TMCMC solutions
mu_exact    = 4.81;      % for x_1
sigma_exact = 0.196;     % for x_1
cE_exact    = 2.36e-6;

% show results
fprintf('\nExact model evidence = %g', cE_exact);
fprintf('\nModel evidence = %g\n', exp(logcE));
fprintf('\nExact posterior mean = %g', mu_exact);
fprintf('\nMean value of samples = %g\n', mean(x1p{end}));
fprintf('\nExact posterior std = %g', sigma_exact);
fprintf('\nStd of samples = %g\n\n', std(x1p{end}));
%%END
