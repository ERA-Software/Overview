%% example 1: 1D posterior
%{
---------------------------------------------------------------------------
By:                Date:          Topic:
Fong-Lin Wu        July.2019      
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

%% SMC-SuS
N  = 1e3;        % number of samples per level
p = 1;        % N/number of chains per level
burn = 2;       % Burn-in length (per chain)
tarCoV = 1.5;     % target coefficient of variation of the weights
k_init = 2;

% fprintf('\nSequential Monte Carlo with aCS: \n\n');
% [samplesU, samplesX, q, logcE] = SMC_aCS(N, p, log_likelihood, prior_pdf, burn, tarCoV);
fprintf('\nSequential Monte Carlo with GM: \n\n');
[samplesU, samplesX, q, k_fin, logcE] = SMC_GM(N, p, log_likelihood, prior_pdf, k_init, burn, tarCoV);

%% extract the samples
nsub = length(q);
u1p  = cell(nsub,1); u0p = cell(nsub,1);
x1p  = cell(nsub,1); pp  = cell(nsub,1);
for i = 1:nsub
   % samples in standard
   u1p{i} = samplesU{i}(1,:);              
   % samples in physical
   x1p{i} = samplesX{i}(1,:);   
end

%% reference and aBUS solutions
mu_exact    = 4.81;
sigma_exact = 0.196;
cE_exact    = 2.36e-6;

% show results
fprintf('\nExact model evidence = %g', cE_exact);
fprintf('\nModel evidence SMC-aCS = %g\n', exp(logcE));
fprintf('\nExact posterior mean = %g', mu_exact);
fprintf('\nMean value of samples = %g\n', mean(x1p{end}));
fprintf('\nExact posterior std = %g', sigma_exact);
fprintf('\nStd of samples = %g\n\n', std(x1p{end}));
%%END
