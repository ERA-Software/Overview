%% Improved cross entropy method: Ex. 1 Ref. 3 - strongly nonlinear and non-monotonic LSF
%{
---------------------------------------------------------------------------
Created by:
Daniel Koutas
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version 2022-04
---------------------------------------------------------------------------
Based on:
1. Papaioannou, I., Geyer, S., & Straub, D. (2019).
   Improved cross entropy-based importance sampling with a flexible mixture model.
   Reliability Engineering & System Safety, 191
2. Geyer, S., Papaioannou, I., & Straub, D. (2019).
   Cross entropy-based importance sampling using Gaussian densities revisited. 
   Structural Safety, 76, 15–27
3. Li, L., Papaioannou, I., & Straub, D. (2019)
   Global reliability sensitivity estimation based on failure samples"
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

% Definition of additional values
max_it    = 100;     % maximum number of iteration steps per simulation
N         = 2.0e3;   % definition of number of samples per level
CV_target = 2.0;     % target CV

%% CE method
p      = 0.1;     % quantile value to select samples for parameter update
k_init = 1;       % initial number of distributions in the Mixture models (GM/vMFNM)

fprintf('Cross-Entropy based IS stage: \n');

method = 'iCE_SG';
%method = 'iCE_GM';
%method = 'iCE_vMFNM';
%method = 'CE_SG';
%method = 'CE_GM';
%method = 'CE_vMFNM';

fprintf('Chosen method: %s\n', method);
switch method
    case 'iCE_SG'        % improved CE single with single gaussian
      [Pf_CE, lv, N_tot, samplesU, samplesX, S_F1] = iCE_SG(N, g, pi_pdf, max_it, CV_target); 
      
    case 'iCE_GM'        % improved CE single with gaussian mixture
      [Pf_CE, lv, N_tot, samplesU, samplesX, k_fin, S_F1] = iCE_GM(N, g, pi_pdf, max_it, CV_target, k_init); 
      
    case 'iCE_vMFNM'     % improved CE with adaptive vMFN mixture        
      [Pf_CE, lv, N_tot, samplesU, samplesX, k_fin, S_F1] = iCE_vMFNM(N, g, pi_pdf, max_it, CV_target, k_init); 
      
    case 'CE_SG'         % single gaussian 
      [Pf_CE, lv, N_tot, gamma_hat, samplesU, samplesX, k_fin, S_F1] = CEIS_SG(N, p, g, pi_pdf); 
      
    case 'CE_GM'         % gaussian mixture
      [Pf_CE, lv, N_tot, gamma_hat, samplesU, samplesX, k_fin, S_F1] = CEIS_GM(N, p, g, pi_pdf, k_init);
      
    case 'CE_vMFNM'      % adaptive vMFN mixture
      [Pf_CE, lv, N_tot, gamma_hat, samplesU, samplesX, k_fin, S_F1] = CEIS_vMFNM(N, p, g, pi_pdf, k_init);
      
    otherwise
      error('Choose iCE_SG, SG, or ... methods');
end

% reference solution
pf_ref    = 0.0062;
MC_S_F1  = [0.0811, 0.0045, 0.0398]; % approximately read and extracted from paper
 
% show results
fprintf('\n***Reference Pf: %g ***', pf_ref);
fprintf('\n***CE Pf: %g ***\n\n', Pf_CE);

fprintf('\n***MC sensitivity indices:');
disp(MC_S_F1);
fprintf('\n***CE sensitivity indices:');
disp(S_F1);