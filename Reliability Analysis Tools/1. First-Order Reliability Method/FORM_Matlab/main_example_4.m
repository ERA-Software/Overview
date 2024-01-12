%% FORM using HLRF algorithm and fmincon: Ex. 2 Ref. 3 - parabolic/concave limit-state function
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
3. "Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
---------------------------------------------------------------------------
%}
clear; close all; clc;

%% definition of the random variables
d      = 2;          % number of dimensions
pi_pdf = repmat(ERADist('standardnormal','PAR'),d,1);   % n independent rv

% correlation matrix
R = eye(d);   % independent case

% object with distribution information
pi_pdf = ERANataf(pi_pdf,R);    % if you want to include dependence

%% limit state function and its gradient in the original space
b     = 5; 
kappa = 0.5; 
e     = 0.1;
g     = @(u) b - u(:,2) - kappa*(u(:,1)-e).^2;
dg    = @(u) [-2*kappa*(u(:,1)-e); -1];

%% Implementation of sensitivity analysis: 1 - perform, 0 - not perform
sensitivity_analysis = 1;

%% Solve the optimization problem of the First Order Reliability Method
% OPC 1. FORM using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[u_star_hlrf, x_star_hlrf, beta_hlrf, Pf_hlrf, S_F1_hlrf, S_F1_T_hlrf] = FORM_HLRF(g, dg, pi_pdf, sensitivity_analysis);

% OPC 2. FORM using MATLAB fmincon (without analytical gradient)
[u_star_fmc, x_star_fmc, beta_fmc, Pf_fmc, S_F1_fmc, S_F1_T_fmc] = FORM_fmincon(g, [], pi_pdf, sensitivity_analysis);

% OPC 3. FORM using MATLAB fmincon (with analytical gradient)
%[u_star_fmc, x_star_fmc, beta_fmc, Pf_fmc, S_F1_fmc, S_F1_T_fmc] = FORM_fmincon(g, dg, pi_pdf, sensitivity_analysis);

%% Reference values
% The reference values for the first order indices
S_F1_ref   = [0.4377, 0.0078];

% Print reference values for the first order indices
fprintf("***Reference first order Sobol' indices: ***\n");
disp(S_F1_ref);

% reference solution
pf_ref = 3.01e-3;

% show p_f results
fprintf('\n***Reference Pf: %g ***', pf_ref);
fprintf('\n***FORM HLRF Pf: %g ***', Pf_hlrf);
fprintf('\n***FORM fmincon Pf: %g ***\n\n', Pf_fmc);

%% plot HLRF results
if d == 2
  % grid points
  uu      = -6:0.05:6;
  [U1,U2] = meshgrid(uu,uu);
  nnu     = length(uu);
  unod    = cat(2, reshape(U1,nnu^2,1),reshape(U2,nnu^2,1));
  ZU      = g(unod);
  ZU      = reshape(ZU,nnu,nnu);

  figure;  hold on; 
  pcolor(U1,U2,ZU); shading interp;
  contour(U1,U2,ZU,[0 0],'r'); axis equal tight;
  plot(0,0,'r*',u_star_hlrf(1),u_star_hlrf(2),'r*');   % design point in standard
  line([0, u_star_hlrf(1)],[0, u_star_hlrf(2)]);       % reliability index beta
  %
  plot(0,0,'ko',u_star_fmc(1),u_star_fmc(2),'ko');   % design point in standard
  line([0, u_star_fmc(1)],[0, u_star_fmc(2)]);       % reliability index beta
  title('Standard space');
end
%%END