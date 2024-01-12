%% example 2: multi-modal posterior
%{
---------------------------------------------------------------------------
By:                Date:          Topic:
Fong-Lin Wu        June.2019      Complete preliminary code
---------------------------------------------------------------------------
Current version 2021-03
* use of log-evidence logcE
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
%
% likelihood     = @(theta) sum(w.*mvnpdf(theta,mu,Cov)); 
likelihood     = @(theta) sum(w.*mvnpdf(repmat(theta,M,1),mu,Cov)); 
log_likelihood = @(theta) log(likelihood(theta)+realmin); 

%% iTMCMC step
Ns = 1e3;        % number of samples per level
Nb = 0.1*Ns;     % burn-in period
rng('shuffle')
fprintf('\niTMCMC: \n');
[samplesU, samplesX, q, logcE] = iTMCMC(Ns, Nb, log_likelihood, prior_pdf);

%% plot samples
m = length(q);   % number of stages (intermediate levels)

% show results
fprintf('\nModel evidence = %g', exp(logcE));
fprintf('\nMean value of x_1 = %g', mean(samplesX{m}(1,:)));
fprintf('\nStd of x_1 = %g', std(samplesX{m}(1,:)));
fprintf('\nMean value of x_2 = %g', mean(samplesX{m}(2,:)));
fprintf('\nStd of x_2 = %g\n\n', std(samplesX{m}(2,:)));

% plot q values
figure;
plot(0:m-1,q,'ro-');
xlabel('Intermediate levels $j$','Interpreter','Latex','FontSize', 18);
ylabel('$q_j$','Interpreter','Latex','FontSize', 18);
set(gca,'FontSize',15); axis tight;

% plot samples increasing q
idx = [1, round(m/3), round(2*m/3), m];
figure;
for i = 1:4
    subplot(2,2,i); plot(samplesU{idx(i)}(:,1),samplesU{idx(i)}(:,2),'r.');
    title(sprintf('$q_j$=%4.3f',q(idx(i))),'Interpreter','Latex','FontSize', 18);
    xlabel('$u_1$','Interpreter','Latex','FontSize', 18);
    ylabel('$u_2$','Interpreter','Latex','FontSize', 18);
    set(gca,'FontSize',15);  axis equal;
end
annotation('textbox', [0, 0.9, 1, 0.1],'String', '\bf Standard space', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center');

figure;
for i = 1:4
    subplot(2,2,i); plot(samplesX{idx(i)}(:,1),samplesX{idx(i)}(:,2),'b.');
    title(sprintf('$q_j$=%4.3f',q(idx(i))),'Interpreter','Latex','FontSize', 18);
    xlabel('$\theta_1$','Interpreter','Latex','FontSize', 18);
    ylabel('$\theta_2$','Interpreter','Latex','FontSize', 18);
    set(gca,'FontSize',15);  axis equal;
end
annotation('textbox', [0, 0.9, 1, 0.1],'String', '\bf Original space', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center');
