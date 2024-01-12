%% example 2: multi-modal posterior
%{
---------------------------------------------------------------------------
By:                Date:          Topic:
Fong-Lin Wu        June.2019      
---------------------------------------------------------------------------
Current version 2021-03
* use of log-evidence logcE
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
prior_pdf = ERANataf(prior_pdf,R);

%% likelihood PDF (mixture of gaussians in 2D)
s = RandStream.create('mt19937ar','seed',123);   % fixed the seed to get the same data points
RandStream.setGlobalStream(s);

% data set
m     = 10;                          % number of mixture
mu    = prior_pdf.random(m);         % Here using ERADist.random
sigma = 0.1;
Cov   = (sigma^2)*eye(d);
w     = 0.1*ones(m,1);

% define likelihood
likelihood_opt   = @(theta) sum(w.*mvnpdf(theta,mu,Cov));               % Modified likelihood function for sum
likelihood       = @(theta) sum(w.*mvnpdf(repmat(theta,m,1),mu,Cov));   % Likelihood function for M mixtures
log_likelihood   = @(theta) log(likelihood(theta)+realmin);            % Log of likelihood function for M mixtures

%% find scale constant c
method = 1;
switch method
    % use MLE to find c
    case 1
        f_start = log(mu);
        fun     = @(lnF) -log(likelihood_opt(exp(f_start))+realmin);
        options = optimset('MaxIter',1e7,'MaxFunEvals',1e7);
        MLE_ln  = fminsearch(fun, f_start, options);
        MLE     = exp(MLE_ln);          % likelihood(MLE) = 1
        c       = 1/likelihood_opt(MLE);
        
        % some likelihood evaluations to find c
    case 2
        K  = 5e3;                       % number of samples
        u  = randn(d,K)';                % samples in standard space
        x  = prior_pdf.U2X(u);          % samples in physical space
        % likelihood function evaluation
        L_eval = zeros(K,1);
        for i = 1:K
            L_eval(i) = likelihood(x(i,:));
        end
        c = 1/max(L_eval);              % Ref. 1 Eq. 34
        
        % use approximation to find c
    case 3
        fprintf('This method requires a large number of measurements\n');
        m = length(mu);
        p = 0.05;
        c = 1/exp(-0.5*chi2inv(p,m));   % Ref. 1 Eq. 38
        
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
u1p  = cell(nsub,1);   u2p  = cell(nsub,1);   u0p = cell(nsub,1);
x1p  = cell(nsub,1);   x2p  = cell(nsub,1);   pp  = cell(nsub,1);
for i = 1:nsub
    % samples in standard
    u1p{i} = samplesU.total{i}(:,1);
    u2p{i} = samplesU.total{i}(:,2);
    u0p{i} = samplesU.total{i}(:,3);
    % samples in physical
    x1p{i} = samplesX{i}(:,1);
    x2p{i} = samplesX{i}(:,2);
    pp{i}  = samplesX{i}(:,3);
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
