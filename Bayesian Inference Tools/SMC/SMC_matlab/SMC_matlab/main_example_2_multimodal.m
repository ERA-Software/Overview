%% example 2: multi-modal posterior
%{
---------------------------------------------------------------------------
By:                Date:          Topic:
Fong-Lin Wu        June.2019      
---------------------------------------------------------------------------
References:
1."Asymptotically independent Markov sampling: a new MCMC scheme for Bayesian inference"
   James L. Beck and Konstantin M. Zuev
   International Journal for Uncertainty Quantification, 3.5 (2013) 445-474.
2."Bayesian inference with subset simulation: strategies and improvements"
   Wolfgang Betz et al.
   Computer Methods in Applied Mechanics and Engineering 331 (2018) 72-93.
---------------------------------------------------------------------------
%}
clear; clc; close all;

% add path to ERADist and ERANataf classes
% Source: https://www.bgu.tum.de/era/software/eradist/
% addpath('../../../ERA_Dist/ERADistNataf_MATLAB/')

%% initial data
d = 2;      % number of dimensions (number of uncertain parameters)

%% prior PDF for theta (bivariate uniform)
a     = 10;
bound = repmat([0 a],d,1);   % boundaries of the uniform distribution box

% define the prior
prior_pdf(1) = ERADist('uniform','PAR',[bound(1,1),bound(1,2)]);
prior_pdf(2) = ERADist('uniform','PAR',[bound(2,1),bound(2,2)]);

% correlation matrix
R = eye(d);   % independent case

% object with distribution information
prior_pdf = ERANataf(prior_pdf, R);

%% likelihood PDF (mixture of gaussians in 2D)
s = RandStream.create('mt19937ar','seed',123);   % fixed the seed to get the same data points
RandStream.setGlobalStream(s);
%% likelihood PDF (mixture of gaussians in 2D)
s = RandStream.create('mt19937ar','seed',12001);   % fixed the seed to get the same data points
RandStream.setGlobalStream(s);

M     = 10;                          % number of mixture
mu    = prior_pdf.random(M);        % Here using ERDist.random
sigma = 0.1;
Cov   = (sigma^2)*eye(d);
w     = 0.1*ones(M,1);

likelihood     = @(theta) sum(w.*mvnpdf(repmat(theta,M,1),mu,Cov)); 
log_likelihood = @(theta) log(likelihood(theta)+realmin); 


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
u1p  = cell(nsub,1);   u2p = cell(nsub,1);  
x1p  = cell(nsub,1);   x2p = cell(nsub,1); 
for i = 1:nsub
    % samples in standard
    u1p{i} = samplesU{i}(1,:);
    u2p{i} = samplesU{i}(2,:);
    % samples in physical
    x1p{i} = samplesX{i}(1,:);
    x2p{i} = samplesX{i}(2,:);
end

% show results
fprintf('\nModel evidence BUS-SuS = %g', exp(logcE));
fprintf('\nMean value of x_1 = %g', mean(x1p{end}));
fprintf('\tStd of x_1 = %g', std(x1p{end}));
fprintf('\nMean value of x_2 = %g', mean(x2p{end}));
fprintf('\tStd of x_2 = %g\n', std(x2p{end}));

%% plot samples
figure;
for i = 1:nsub
    subplot(2,3,i); plot(u1p{i},u2p{i},'r.');
    xlabel('$u_1$','Interpreter','Latex','FontSize', 18);
    ylabel('$u_2$','Interpreter','Latex','FontSize', 18);
    set(gca,'FontSize',15); axis equal;
end
annotation('textbox', [0, 0.9, 1, 0.1],'String', '\bf Standard space', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center');

figure;
for i = 1:nsub
    subplot(2,3,i); plot(x1p{i},x2p{i},'b.');
    xlabel('$\theta_1$','Interpreter','Latex','FontSize', 18);
    ylabel('$\theta_2$','Interpreter','Latex','FontSize', 18);
    set(gca,'FontSize',15); axis equal;
end
annotation('textbox', [0, 0.9, 1, 0.1],'String', '\bf Original space', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center');
%%END