%% Nonlinear bi-dimensional limit-state function
%{
---------------------------------------------------------------------------
Created by:
Daniel Koutas
Engineering Risk Analysis Group   
Technische Universitat Muenchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Current version 2021-06
---------------------------------------------------------------------------
Based on:
1."Advanced Line Sampling for efficient robust reliability analysis"
   Marco de Angelis, Edoardo Patelli & Michael Beer.
   Structural Safety 52 (2015) 170–182.
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
pi_pdf = cat(1, ERADist('normal','MOM', [5,2]), ...
                ERADist('normal','MOM', [2,2]));

%% correlation matrix
R = eye(d);   % independent case

% object with distribution information
pi_pdf = ERANataf(pi_pdf, R);

%% limit state function
% you can manually choose between examples 1 to 5
example = 1;

if round(example) < 1 || round(example) > 5
    error("Selected example is not available. Please choose an integer between 1 and 5")
end

% vector of constants to choose from
a_vec = [10, 10.2, 10.5, 12, 14];
% constant for the limit-state function
a = a_vec(round(example));
% limit state function
g = @(x) -sqrt(x(:,1).^2 + x(:,2).^2) + a;


%% line sampling
N  = 100;        % Total number of samples
u_star = [0.5, 1];

fprintf('LINE SAMPLING: \n');
[Pf_LS, delta_LS, u_new, x_new, n_eval] = CLS(u_star, N, g, pi_pdf);

% vector of reference solutions
pf_ref_vec = [1.49e-2, 1.16e-2, 7.40e-3, 7.06e-4, 1.42e-5];

% reference solution
pf_ref = pf_ref_vec(round(example));

% show p_f results
fprintf('\n***Reference Pf: %g ***', pf_ref);
fprintf('\n***CLS Pf: %g ***\n\n', Pf_LS);