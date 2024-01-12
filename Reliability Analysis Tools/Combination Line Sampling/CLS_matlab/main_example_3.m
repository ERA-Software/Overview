%% Subset Simulation: Ex. 4 Ref. 3 - linear and convex limit state function
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe
Matthias Willer
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Current version 2020-10
* Adaptation to new ERANataf class
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SuS"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
2."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
3."Cross entropy-based importance sampling using Gaussian densities revisited"
   Geyer et al.
   To appear in Structural Safety (2018)
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
d      = 2;          % number of dimensions
pi_pdf = repmat(ERADist('standardnormal','PAR'),d,1);   % n independent rv

% % correlation matrix
% R = eye(d);   % independent case
% 
% % object with distribution information
% pi_pdf = ERANataf(pi_pdf,R);    % if you want to include dependence

%% limit state function
g = @(x) 0.1*(x(:,1)-x(:,2)).^2 - (x(:,1)+x(:,2))./sqrt(2) + 2.5;

%% line sampling
N  = 100;        % Total number of samples
u_star = [0.5,1];

fprintf('LINE SAMPLING: \n');
[Pf_LS, delta_LS, u_new, x_new, n_eval] = CLS(u_star, N, g, pi_pdf);

% reference solution
pf_ref = 4.21e-3;

% show p_f results
fprintf('\n***Reference Pf: %g ***', pf_ref);
fprintf('\n***CLS Pf: %g ***\n\n', Pf_LS);

