%% Subset Simulation: Ex. 1 Ref. 2 - Strongly nonlinear and non-monotonic LSF
%{
---------------------------------------------------------------------------
Created by:
Daniel Koutas
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2022-04
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SuS"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
2."Global reliability sensitivity estimation based on failure samples"
   Luyi et al.
   Structural Safety 81 (2019) 101871
---------------------------------------------------------------------------
%}
clear; close all; clc;
rng(0)

%% definition of the random variables
d      = 3;          % number of dimensions
pi_pdf = repmat(ERADist('standardnormal','PAR'), 3, 1);   % n independent rv

% % correlation matrix
% R = eye(d);   % independent case
% 
% % object with distribution information
% pi_pdf = ERANataf(pi_pdf, R);    % if you want to include dependence

%% limit state function
g    = @(x) x(:,1).^3+10*x(:,2).^2+0.1*sin(pi*x(:,2))+10*x(:,3).^2+40*sin(pi*x(:,3))+38;

%% subset simulation
N  = 2000;        % Total number of samples for each level
p0 = 0.1;         % Probability of each subset, chosen adaptively

fprintf('SUBSET SIMULATION: \n');
[Pf_SuS, delta_SuS, b, Pf, b_sus, pf_sus, samplesU, samplesX, SuS_S_F1] = SuS(N,p0,g,pi_pdf);
% exact solution
pf_ref    = 0.0062;
MC_S_F1  = [0.0811, 0.0045, 0.0398]; % approximately read and extracted from paper
 
% show results
fprintf('\n***Reference Pf: %g ***', pf_ref);
fprintf('\n***SuS Pf: %g ***\n\n', Pf_SuS);

fprintf('\n***MC sensitivity indices:');
disp(MC_S_F1);
fprintf('\n***SuS sensitivity indices:');
disp(SuS_S_F1);