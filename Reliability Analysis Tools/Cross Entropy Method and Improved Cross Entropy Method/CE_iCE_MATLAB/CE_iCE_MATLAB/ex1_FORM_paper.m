%% improved Cross entropy method: Ex. 1 Ref. 3 - steel column
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
2. Papaioannou, I., Geyer, S., & Straub, D. (2019).
   Improved cross entropy-based importance sampling with a flexible mixture model.
   Reliability Engineering & System Safety, 191
3. Geyer, S., Papaioannou, I., & Straub, D. (2019).
   Cross entropy-based importance sampling using Gaussian densities revisited. 
   Structural Safety, 76, 15–27
3. Papaioannou, I., & Straub, D. (2021)
   Variance-based reliability sensitivity analysis and the FORM a-factors."    
   Reliability Engineering and System Safety 210 107496
---------------------------------------------------------------------------
%}
clear; close all; clc;

%% Initialization
b   = 250;
h   = 250;
t_b = 15;
t_h = 10;
L   = 7.5e+3;

A_s = 2*b*t_b + h*t_h;
W_s = h*t_h^3/(6*b) + t_b*b^2/3;
I_s = h*t_h^3/12 + t_b*b^3/6;

%% definition of the random variables
d       = 5;          % number of dimensions
P_p     = ERADist('normal','MOM',[200e+3, 20e+3]);
P_e     = ERADist('gumbel','MOM',[400e+3, 60e+3]);
delta_0 = ERADist('normal','MOM',[30, 10]);
f_y     = ERADist('lognormal','MOM',[400, 32]);
E       = ERADist('lognormal','MOM',[2.1e+5, 8.4e+3]);
pi_pdf  = [P_p; P_e; delta_0; f_y; E];
          

% correlation matrix
R = eye(d);   % independent case

% object with distribution information
pi_pdf = ERANataf(pi_pdf,R);    % if you want to include dependence

%% limit state function in the original space
g  = @(x) 1 - ((x(:,1)+x(:,2))./(x(:,4).*A_s) + ...
         (x(:,1)+x(:,2)).*x(:,3)./(x(:,4)*W_s) .* ...
          pi^2.*x(:,5).*I_s./L^2 ./ (pi^2.*x(:,5).*I_s./L.^2-(x(:,1)+x(:,2))));

% Definition of additional values
max_it    = 100;     % maximum number of iteration steps per simulation
N         = 2.0e3;     % definition of number of samples per level
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

% MC solution given in paper
% The MC results for S_F1_MC have the following COVs in the given order:
% [16.1%, 0.2%, 1.8%, 7.4%, 15.1%]
% Hence the first order indices (except for the second one) have quite high
% uncertainty and should not be considered as exact.
S_F1_MC   = [1.7e-4, 0.1974, 0.0044, 4.8e-4, 1.6e-4];

% The MC results for the total-effect indices have all COVs <= 0.2% so they 
% can be considered as more accurate than the first-order indices
S_F1_T_MC = [0.2365, 0.9896, 0.7354, 0.3595, 0.2145];

% MC probability of failure
Pf_MC = 8.35e-4;

% show indices results
fprintf("\n***MC first order Sobol' indices: ***\n");
disp(S_F1_MC);
fprintf("***CE first order Sobol' indices: ***\n");
disp(S_F1);

fprintf("\n***MC Pf: %g ***\n", Pf_MC);
fprintf("\n***CE Pf: %g***\n", Pf_CE);
