%% Cross entropy method: Ex. 2 Ref. 2 - linear function of independent exponential
%{
---------------------------------------------------------------------------
Improved cross entropy method: Ex. 2 Ref. 2 - linear function of independent exponential
---------------------------------------------------------------------------
Created by:
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Fong-Lin Wu
Matthias Willer
Peter Kaplan
Daniel Koutas
Ivan Olarte-Rodriguez

Engineering Risk Analysis Group
Technische Universitaet Muenchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Current version 2023-12
* Modification to Sensitivity Analysis Calls
---------------------------------------------------------------------------
Based on:
1. Papaioannou, I., Geyer, S., & Straub, D. (2019).
   Improved cross entropy-based importance sampling with a flexible mixture model.
   Reliability Engineering & System Safety, 191
2. Geyer, S., Papaioannou, I., & Straub, D. (2019).
   Cross entropy-based importance sampling using Gaussian densities revisited. 
   Structural Safety, 76, 15–27
---------------------------------------------------------------------------
%}
clear; close all; clc;

%% definition of the random variables
% d      = 100;         % number of dimensions
d      = 2;           % number of dimensions
pi_pdf = repmat(ERADist('exponential','PAR',1),d,1);   % n independent rv

% correlation matrix
R = eye(d);   % independent case
 
% object with distribution information
pi_pdf = ERANataf(pi_pdf,R);    % if you want to include dependence

%% limit state function
% Ca = 140;
Ca = 10;
g  = @(x) Ca - sum(x, 2);

% Definition of additional values
max_it    = 100;     % maximum number of iteration steps per simulation
N         = 2.0e3;   % definition of number of samples per level
CV_target = 2.5;     % target CV


%% Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 1;

%% CE method
p      = 0.1;     % quantile value to select samples for parameter update
k_init = 1;       % initial number of distributions in the Mixture models (GM/vMFNM)

% exact solution
lambda   = 1;
pf_ex    = 1 - gamcdf(Ca,d,lambda);
Pf_exact = @(gg) 1-gamcdf(Ca-gg,d,lambda);
gg       = 0:0.1:30;

fprintf('Cross-Entropy based IS stage: \n');

% method = 'iCE_SG';
% method = 'iCE_GM';
% method = 'iCE_vMFNM';
% method = 'CE_SG';
method = 'CE_GM';
% method = 'CE_vMFNM';

fprintf('Chosen method: %s\n', method);
switch method
    case 'iCE_SG'        % improved CE single with single gaussian
      [Pf_CE, lv, N_tot, samplesU, samplesX, W_final, fs_iid] = iCE_SG(N, g, pi_pdf, max_it, CV_target, samples_return); 
      
    case 'iCE_GM'        % improved CE single with gaussian mixture
      [Pf_CE, lv, N_tot, samplesU, samplesX, k_fin, W_final, fs_iid] = iCE_GM(N, g, pi_pdf, max_it, CV_target, k_init, samples_return); 
      
    case 'iCE_vMFNM'     % improved CE with adaptive vMFN mixture        
      [Pf_CE, lv, N_tot, samplesU, samplesX, k_fin, W_final, fs_iid] = iCE_vMFNM(N, g, pi_pdf, max_it, CV_target, k_init,samples_return); 
      
    case 'CE_SG'         % single gaussian 
      [Pf_CE, lv, N_tot, gamma_hat, samplesU, samplesX, k_fin, W_final, fs_iid] = CEIS_SG(N, p, g, pi_pdf, samples_return); 
      
    case 'CE_GM'         % gaussian mixture
      [Pf_CE, lv, N_tot, gamma_hat, samplesU, samplesX, k_fin, W_final, fs_iid] = CEIS_GM(N, p, g, pi_pdf, k_init, samples_return);
      
    case 'CE_vMFNM'      % adaptive vMFN mixture
      [Pf_CE, lv, N_tot, gamma_hat, samplesU, samplesX, k_fin, W_final, fs_iid] = CEIS_vMFNM(N, p, g, pi_pdf, k_init, samples_return);
      
    otherwise
      error('Choose iCE_SG, SG, or ... methods');
end

%% Implementation of sensitivity analysis

% Computation of Sobol Indices
compute_Sobol = true;

% Computation of EVPPI (based on standard cost of failure (10^8) and cost
% of replacement (10^5)
compute_EVPPI = true;

[S_F1, S_EVPPI] = Sim_Sensitivity(fs_iid, Pf_CE, pi_pdf, compute_Sobol, compute_EVPPI);

%% Reference values
% The reference values for the first order indices
S_F1_ref   = [0.2021, 0.1891];

% Print reference values for the first order indices
fprintf("***Reference first order Sobol' indices: ***\n");
disp(S_F1_ref);

% show p_f results
fprintf('\n***Exact Pf: %g ***', pf_ex);
fprintf('\n***CE-based IS Pf: %g ***\n\n', Pf_CE);