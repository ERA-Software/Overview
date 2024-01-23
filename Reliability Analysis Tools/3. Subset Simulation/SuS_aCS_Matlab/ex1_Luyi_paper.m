%% Subset Simulation: Ex. 1 Ref. 2 - Strongly nonlinear and non-monotonic LSF
%{
---------------------------------------------------------------------------
Created by:
Daniel Koutas

Developed by:
Ivan Olarte-Rodriguez

Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Current version 2023-12
* Modification to Sensitivity Analysis Calls
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

%% Samples return
samples_return = 1; 

%% subset simulation
N  = 2000;        % Total number of samples for each level
p0 = 0.1;         % Probability of each subset, chosen adaptively

fprintf('SUBSET SIMULATION: \n');
[Pf_SuS, delta_SuS, b, Pf, b_sus, pf_sus, samplesU, samplesX, fs_iid] = SuS(N,p0,g,pi_pdf, samples_return);

%% Implementation of sensitivity analysis

% Computation of Sobol Indices
sensitivity_analysis = true;

% Computation of EVPPI (based on standard cost of failure (10^8) and cost
% of replacement (10^5)
compute_EVPPI = false;

[SuS_S_F1,~] = Sim_Sensitivity(fs_iid,Pf_SuS,pi_pdf,sensitivity_analysis,compute_EVPPI);

%% exact solution
pf_ref    = 0.0062;
MC_S_F1  = [0.0811, 0.0045, 0.0398]; % approximately read and extracted from paper
 
%% Show results
fprintf('\n***Reference Pf: %g ***', pf_ref);
fprintf('\n***SuS Pf: %g ***\n\n', Pf_SuS);

fprintf('***MC sensitivity indices:');
disp(MC_S_F1);
fprintf('***SuS sensitivity indices:');
disp(SuS_S_F1);