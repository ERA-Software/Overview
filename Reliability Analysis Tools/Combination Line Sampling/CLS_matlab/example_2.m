%% Noisy limit-state function
%{
---------------------------------------------------------------------------
Created by:
Daniel Koutas
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Current version 2021-06
---------------------------------------------------------------------------
Based on:
1."Combination line sampling for structural reliability analysis"
   Iason Papaioannou & Daniel Straub.
   Structural Safety 88 (2021) 102025.
2."Optimization algorithms for structural reliability"
   Pei-Ling Liu & Armen Der Kiureghian. 
   Structural safety 9 (1991) 161-177.
---------------------------------------------------------------------------
%}
clear; close all; clc;

%% Fix Seed with true or deactivate with false
fix_seed = false;
if fix_seed
    s = RandStream.create('mt19937ar','seed',2021);   % fixed the seed to get the same data points
    RandStream.setGlobalStream(s);
end

%% definition of the random variables
d      = 6;          % number of dimensions
pi_pdf = repmat(ERADist('lognormal','MOM', [120, 8]), 4, 1);  
pi_pdf = cat(1, pi_pdf, ERADist('lognormal','MOM', [50, 10]));
pi_pdf = cat(1, pi_pdf, ERADist('lognormal','MOM', [40, 8]));


%% correlation matrix
R = eye(d);   % independent case

% object with distribution information
pi_pdf = ERANataf(pi_pdf, R);

%% limit state function
g = @(x) x(:,1) + 2*x(:,2) + 2*x(:,3) + x(:,4) - 5*x(:,5) - 5*x(:,6) + 0.1*sum(sin(100*x(:,1:4)), 2);


%% line sampling
N  = 100;        % Total number of samples
u_star = [0.35, 0.24, 0.24, 0.35, 0.63, 0.50];

fprintf('LINE SAMPLING: \n');
[Pf_LS, delta_LS, u_new, x_new, n_eval] = CLS(u_star, N, g, pi_pdf);

% reference solution
pf_ref = 5.29e-4;

% show p_f results
fprintf('\n***Reference Pf: %g ***', pf_ref);
fprintf('\n***CLS Pf: %g ***\n\n', Pf_LS);


