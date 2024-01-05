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
prior_pdf = ERANataf(prior_pdf, R);

%% likelihood PDF (mixture of gaussians in 2D)
s = RandStream.create('mt19937ar','seed',123);   % fixed the seed to get the same data points
RandStream.setGlobalStream(s);
%
M     = 10;
mu    = prior_pdf.random(M);
sigma = 0.1;
Cov   = (sigma^2)*eye(d);
w     = 0.1*ones(M,1);
%
likelihood     = @(theta) sum(w.*mvnpdf(repmat(theta,M,1),mu,Cov));
log_likelihood = @(theta) log(likelihood(theta)+realmin);

%% aBUS-SuS
N  = 3e3;        % number of samples per level
p0 = 0.1;        % probability of each subset

fprintf('\naBUS with SUBSET SIMULATION: \n\n');
[h, samplesU, samplesX, logcE, c, sigma] = aBUS_SuS(N, p0, log_likelihood, prior_pdf);

%% extract the samples
nsub = length(h);
u1p  = cell(nsub,1);   u2p = cell(nsub,1);   u0p = cell(nsub,1);
x1p  = cell(nsub,1);   x2p = cell(nsub,1);   pp  = cell(nsub,1);
for i = 1:nsub
    % samples in standard
    u1p{i} = samplesU{i}(:,1);
    u2p{i} = samplesU{i}(:,2);
    u0p{i} = samplesU{i}(:,3);
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
%%END