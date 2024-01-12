%% Sequential importance sampling: Ex. 5 Ref. 2 - points outside of hypersphere
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe
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
2."Bayesian inference of engineering models"
   Wolfgang Betz
   Ph.D. Thesis.
---------------------------------------------------------------------------
%}
clear; close all; clc;

rng(10);

%% definition of the random variables
d      = 2;          % number of dimensions
pi_pdf = repmat(ERADist('standardnormal','PAR'), d, 1);   % n independent rv

% % correlation matrix
% R = eye(d);   % independent case
% 
% % object with distribution information
% pi_pdf = ERANataf(pi_pdf,R);    % if you want to include dependence

%% limit state function
r = 5.26;    % radius
m = 1;       % m in [0,4]
g = @(u) 1 - (sqrt(sum(u.^2,2))/r).^2 - (u(:,1)/r).*((1-(sqrt(sum(u.^2,2))/r).^m)./(1+(sqrt(sum(u.^2,2))/r).^m));

%% Implementation of sensitivity analysis: 1 - perform, 0 - not perform
sensitivity_analysis = 1;

%% Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 2;

%% Sequential importance sampling
N      = 3000;    % total number of samples for each level
p      = 0.1;     % N/number of chains per level
k_init = 3;       % initial number of Gaussians in the Mixture Model (GM)
burn   = 0;       % burn-in period
tarCOV = 1.5;     % target COV of weights

fprintf('\nSIS method: \n');
method = 'aCS';
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
S_F1_ref   = [0.1857, 0.1857];

% Print reference values for the first order indices
fprintf("***Reference first order Sobol' indices: ***\n");
disp(S_F1_ref);

% reference solution
pf_ref = 1e-6;

% show p_f results
fprintf('\n***Reference Pf: %g ***', pf_ref);
fprintf('\n***SIS Pf: %g ***\n\n', Pf_SIS);

%% Plots
% plot samplesU
if ~isempty(samplesU{1})
    if d == 2
       xx    = -6:0.05:6; 
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