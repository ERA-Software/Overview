%% Subset Simulation: Ex. 2 Ref. 2 - linear function of independent exponential
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe
Matthias Willer
Daniel Koutas
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Current version 2022-04
* Inclusion of sensitivity analysis
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SuS"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
2."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
---------------------------------------------------------------------------
%}
clear; close all; clc;

%% definition of the random variables
% d      = 100;         % number of dimensions
d      = 2;         % number of dimensions
pi_pdf = repmat(ERADist('exponential','PAR',1),d,1);   % n independent rv

% % correlation matrix
% R = eye(d);   % independent case
% 
% % object with distribution information
% pi_pdf = ERANataf(pi_pdf,R);    % if you want to include dependence

%% limit state function
% Ca = 140;
Ca = 10;
g  = @(x) Ca - sum(x, 2);

%% Implementation of sensitivity analysis: 1 - perform, 0 - not perform
sensitivity_analysis = 1;

%% Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 1;

%% subset simulation
N  = 2000;         % Total number of samples for each level
p0 = 0.1;          % Probability of each subset, chosen adaptively

fprintf('SUBSET SIMULATION: \n');
[Pf_SuS, delta_SuS, b, Pf, b_sus, pf_sus, samplesU, samplesX, S_F1] = SuS(N, p0, g, pi_pdf, sensitivity_analysis, samples_return);

%% Reference values
% The reference values for the first order indices
S_F1_ref   = [0.2021, 0.1891];

% Print reference values for the first order indices
fprintf("***Reference first order Sobol' indices: ***\n");
disp(S_F1_ref);

% exact solution
lambda   = 1;
pf_ex    = 1 - gamcdf(Ca, d, lambda);
Pf_exact = @(gg) 1-gamcdf(Ca-gg, d, lambda);
gg       = 0:0.1:30;

% show p_f results
fprintf('\n***Exact Pf: %g ***', pf_ex);
fprintf('\n***SuS Pf: %g ***\n\n', Pf_SuS);

%% Plots
% Plot failure probability: Exact
figure; 
semilogy(gg, Pf_exact(gg),'b-'); axis tight; 
title('Failure probability estimate','Interpreter','Latex','FontSize', 20);
xlabel('Limit state function, $g$','Interpreter','Latex','FontSize', 18);   
ylabel('Failure probability, $P_f$','Interpreter','Latex','FontSize', 18);

% Plot failure probability: SuS
hold on;
semilogy(b_sus, pf_sus, 'r--');           % curve
semilogy(b, Pf, 'ko', 'MarkerSize', 5);   % points
semilogy(0, Pf_SuS, 'b*', 'MarkerSize', 6);
semilogy(0, pf_ex, 'ro', 'MarkerSize', 8);
set(gca,'yscale','log'); axis tight;
hl = legend('Exact', 'SuS', 'Intermediate levels', 'Pf SuS', 'Pf Exact', 'Location', 'SE');
set(hl,'Interpreter','latex'); set(gca,'FontSize',18);