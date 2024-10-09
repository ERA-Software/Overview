%% example 2: multi-modal posterior
%{
---------------------------------------------------------------------------
By:                Date:          Topic:
Fong-Lin Wu        June.2019      Complete preliminary code
---------------------------------------------------------------------------
Current version 2021-03
* Adaptation to new ERANataf class
---------------------------------------------------------------------------
References:
1."Asymptotically independent Markov sampling: a new MCMC scheme for Bayesian inference"
   James L. Beck and Konstantin M. Zuev
   International Journal for Uncertainty Quantification, 3.5 (2013) 445-474.
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
s = RandStream.create('mt19937ar','seed',12001);   % fixed the seed to get the same data points
RandStream.setGlobalStream(s);
%
M     = 10;                          % number of mixture
mu    = prior_pdf.random(M);        % Here using ERDist.random
sigma = 0.1;
Cov   = (sigma^2)*eye(d);
w     = 0.1*ones(M,1);

% define likelihood
likelihood     = @(theta) sum(w.*reshape(mvnpdf(repelem(theta,M,1), repmat(mu',size(theta,1),1), Cov), M, size(theta,1))); 
log_likelihood = @(theta) log(likelihood(theta)+realmin); 

%% CEBU step
N = 1e3;        % number of samples per level
Nlast = N;     % burn-in period
max_steps = 100;
tarCoV = 1.5;
k_init = 5;
rng('shuffle')
fprintf('\nCEBU: \n');

method = "GM";
%method = "vMFNM";

if strcmp(method, "GM")
    [samplesU, samplesX, v_tot, beta_tot, k_fin, evidence, Wlast_normed, ...
     f_s_iid] = CEBU_GM(N, log_likelihood, prior_pdf, max_steps, tarCoV, k_init, 2, Nlast);
elseif strcmp(method, "vMFNM")
    [samplesU, samplesX, v_tot, beta_tot, k_fin, evidence, Wlast_normed, ...
     f_s_iid] = CEBU_vMFNM(N, log_likelihood, prior_pdf, max_steps, tarCoV, k_init, 2, Nlast);
end

%%
% extract samples
nsub = length(samplesU);   % number of stages (intermediate levels)
if nsub == 0
    fprintf("\nNo samples returned, hence no visualization and reference solutions.\n");
    return
end

% reference and CEBU solutions
fprintf('\nModel evidence = %g', evidence);
fprintf('\nMean value of x_1 = %g', mean(samplesX{nsub}(1,:)));
fprintf('\nStd of x_1 = %g', std(samplesX{nsub}(1,:)));
fprintf('\nMean value of x_2 = %g', mean(samplesX{nsub}(2,:)));
fprintf('\nStd of x_2 = %g\n\n', std(samplesX{nsub}(2,:)));

%% plot samples
nrows = ceil(sqrt(nsub));
ncols = ceil(nsub/nrows);

figure;
for k = 1:nsub
    subplot(nrows,ncols,k); plot(samplesU{k}(:,1),samplesU{k}(:,2),'r.');
    title(sprintf('Step %d',k),'Interpreter','Latex','FontSize', 18);
    xlabel('$u_1$','Interpreter','Latex','FontSize', 18);
    ylabel('$u_2$','Interpreter','Latex','FontSize', 18);
    set(gca,'FontSize',15);  axis equal;
end
annotation('textbox', [0, 0.9, 1, 0.1],'String', '\bf Standard space', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center');

figure;
for k = 1:nsub
    subplot(nrows,ncols,k); plot(samplesX{k}(:,1),samplesX{k}(:,2),'b.');
    title(sprintf('Step %d',k),'Interpreter','Latex','FontSize', 18);
    xlabel('$\theta_1$','Interpreter','Latex','FontSize', 18);
    ylabel('$\theta_2$','Interpreter','Latex','FontSize', 18);
    set(gca,'FontSize',15);  axis equal;
end
annotation('textbox', [0, 0.9, 1, 0.1],'String', '\bf Original space', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center');
