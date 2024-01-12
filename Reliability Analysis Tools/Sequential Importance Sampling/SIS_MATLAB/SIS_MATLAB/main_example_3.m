%% Sequential importance sampling: linear and convex limit state function
%{
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
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
Comments:
* The SIS method in combination with a Gaussian Mixture model can only be
  applied for low-dimensional problems, since its accuracy decreases
  dramatically in high dimensions.
---------------------------------------------------------------------------
Based on:
1."Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
---------------------------------------------------------------------------
%}
clear; close all; clc;

%% definition of the random variables
d      = 2;          % number of dimensions
pi_pdf = repmat(ERADist('standardnormal','PAR'),d,1);   % n independent rv

% % correlation matrix
% R = eye(d);   % independent case
% 
% % object with distribution information
% pi_pdf = ERANataf(pi_pdf,R);    % if you want to include dependence

%% limit state function
g_fun = @(x) min([ 3.2 + (1/sqrt(d))*(x(:,1)+x(:,2)), ...
                   0.1*(x(:,1)-x(:,2)).^2 - (x(:,1)+x(:,2))./sqrt(d) + 2.5 ], [], 2);
g     = @(x) g_fun(x);

%% Implementation of sensitivity analysis: 1 - perform, 0 - not perform
sensitivity_analysis = 0;

%% Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 1;

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
      [Pf_SIS, lv, samplesU, samplesX, k_fin, S_F1] = SIS_GM(N, p, g, pi_pdf, k_init, burn, tarCOV, sensitivity_analysis, samples_return);
   case 'aCS'
      [Pf_SIS, lv, samplesU, samplesX, S_F1] = SIS_aCS(N, p, g, pi_pdf, burn, tarCOV, sensitivity_analysis, samples_return);
   otherwise
      error('Choose GM or aCS methods');
end

%% Reference values
% The reference values for the first order indices
S_F1_ref   = [0.0526, 0.0526];

% Print reference values for the first order indices
fprintf("***Reference first order Sobol' indices: ***\n");
disp(S_F1_ref);

% reference solution
pf_ref = 4.90e-3;

% show p_f results
fprintf('\n***Reference Pf: %g ***', pf_ref);
fprintf('\n***SIS Pf: %g ***\n\n', Pf_SIS);

%% Plots
% plot samplesU
if ~isempty(samplesU{1})
    if d == 2
       xx    = -5:0.05:5;
       nnp   = length(xx);
       [X,Y] = meshgrid(xx);
       xnod  = cat(2,reshape(X',nnp^2,1),reshape(Y',nnp^2,1));
       Z     = g(xnod);
       Z     = reshape(Z,nnp,nnp);
       figure; hold on;
       contour(X,Y,Z,[0,0],'r','LineWidth',3);  % LSF
       for j = 1:length(samplesU)
          u_j_samples= samplesU{j};
          plot(u_j_samples(:,1),u_j_samples(:,2),'.');
       end
       axis equal tight;
    end
end