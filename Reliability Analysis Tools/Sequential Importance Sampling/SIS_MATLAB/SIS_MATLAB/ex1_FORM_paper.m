%% Sequential importance sampling: Ex. 1 Ref. 2 - steel column
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
Comments:
* The SIS method in combination with a Gaussian Mixture model can only be
  applied for low-dimensional problems, since its accuracy decreases
  dramatically in high dimensions.
---------------------------------------------------------------------------
Based on:
1."Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
2."Variance-based reliability sensitivity analysis and the FORM a-factors."
   Papaioannou & Straub    
   Reliability Engineering and System Safety 210 (2021) 107496
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

%% Sequential importance sampling
N      = 2000;    % total number of samples for each level
p      = 0.1;     % N/number of chains per level
k_init = 3;       % initial number of Gaussians in the Mixture Model (GM)
burn   = 0;       % burn-in period
tarCOV = 1.5;     % target COV of weights

fprintf('\nSIS method: \n');
method = 'GM';
switch method
   case 'GM'
      [Pf_SIS, lv, samplesU, samplesX, k_fin, S_F1] = SIS_GM(N, p, g, pi_pdf, k_init, burn, tarCOV);
   case 'aCS'
      [Pf_SIS, lv, samplesU, samplesX, S_F1] = SIS_aCS(N, p, g, pi_pdf, burn, tarCOV);
   otherwise
      error('Choose GM or aCS methods');
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
fprintf("***%s SIS first order Sobol' indices: ***\n", method);
disp(S_F1);

fprintf("\n***MC Pf: %g ***\n", Pf_MC);
fprintf("\n***SIS Pf: %g***\n", Pf_SIS);
