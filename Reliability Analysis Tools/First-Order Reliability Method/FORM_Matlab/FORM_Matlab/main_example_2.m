%% FORM using HLRF algorithm and fmincon: Ex. 2 Ref. 3 - linear function of independent exponential
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
First version: 2018-05
---------------------------------------------------------------------------
Current version: 2022-04
* Inclusion of sensitivity analysis
---------------------------------------------------------------------------
Based on:
1."Structural reliability under combined random load sequences."
   Rackwitz, R., and B. Fiessler (1979).    
   Computers and Structures, 9.5, pp 489-494
2."Lecture Notes in Structural Reliability"
   Straub (2016)
---------------------------------------------------------------------------
%}
clear; close all; clc;

%% definition of the random variables
d      = 100;         % number of dimensions

pi_pdf = repmat(ERADist('exponential','PAR',1),d,1);   % n independent rv

% correlation matrix
R = eye(d);   % independent case

% object with distribution information
pi_pdf = ERANataf(pi_pdf,R);    % if you want to include dependence

%% limit state function and its gradient in the original space
Ca = 140;
g  = @(x) Ca - sum(x,2);
dg = @(x) repmat(-1,d,1);

%% Implementation of sensitivity analysis: 1 - perform, 0 - not perform
sensitivity_analysis = 1;

%% Solve the optimization problem of the First Order Reliability Method

% OPC 1. FORM using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[u_star_hlrf, x_star_hlrf, beta_hlrf, Pf_hlrf, S_F1_hlrf, S_F1_T_hlrf] = FORM_HLRF(g, dg, pi_pdf, sensitivity_analysis);

% OPC 2. FORM using MATLAB fmincon (without analytical gradient)
[u_star_fmc, x_star_fmc, beta_fmc, Pf_fmc, S_F1_fmc, S_F1_T_fmc] = FORM_fmincon(g, [], pi_pdf, sensitivity_analysis);

% OPC 3. FORM using MATLAB fmincon (with analytical gradient)
%[u_star_fmc, x_star_fmc, beta_fmc, Pf_fmc, S_F1_fmc, S_F1_T_fmc] = FORM_fmincon(g, dg, pi_pdf, sensitivity_analysis);

% exact solution
lambda   = 1;
pf_ex    = 1 - gamcdf(Ca,d,lambda);

% show p_f results
fprintf('\n***Exact Pf: %g ***\n', pf_ex);
fprintf('***FORM HLRF Pf: %g ***\n', Pf_hlrf);
fprintf('***FORM fmincon Pf: %g ***\n\n', Pf_fmc);

% The reference values for the first order indices
S_F1_ref   = [0.2021, 0.1891];

% Print reference values for the first order indices
fprintf("***Reference first order Sobol' indices: ***\n");
disp(S_F1_ref);

%%END