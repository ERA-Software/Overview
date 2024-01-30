%% FORM using HLRF algorithm and fmincon: Ex. 1 Ref. 3 - linear function of independent standard normal
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe
Matthias Willer
Daniel Koutas
Ivan Olarte-Rodriguez

Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2018-05
---------------------------------------------------------------------------
Current version: 2023-10
* Modification of Sensitivity Analysis Calls
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
d      = 2;          % number of dimensions
pi_pdf = repmat(ERADist('standardnormal','PAR'),d,1);   % n independent rv

% correlation matrix
R = eye(d);   % independent case

% object with distribution information
pi_pdf = ERANataf(pi_pdf,R);    % if you want to include dependence

%% limit state function and its gradient in the original space
beta = 3.5;
g    = @(x) -sum(x,2)/sqrt(d) + beta;
dg   = @(x) repmat(-1/sqrt(d),d,1);



%% Solve the optimization problem of the First Order Reliability Method
% OPC 1. FORM using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[u_star_hlrf, x_star_hlrf, beta_hlrf, alpha_hlrf, Pf_hlrf ] = FORM_HLRF(g, dg, pi_pdf);

% OPC 2. FORM using MATLAB fmincon (without analytical gradient)
[u_star_fmc, x_star_fmc, beta_fmc, alpha_fmc, Pf_fmc]= FORM_fmincon(g, [], pi_pdf);

% OPC 3. FORM using MATLAB fmincon (with analytical gradient)
%[u_star_fmc, x_star_fmc, beta_fmc, alpha_fmc, Pf_fmc] = FORM_fmincon(g, dg, pi_pdf);

% exact solution
pf_ex = normcdf(-beta);

%% Implementation of sensitivity analysis

% Computation of Sobol Indices
compute_Sobol = true;

% Computation of EVPPI (based on standard cost of failure (10^8) and cost
% of replacement (10^5)
compute_EVPPI = true;

% using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[S_F1_hlrf, S_F1_T_hlrf, S_EVPPI_hlrf] = FORM_Sensitivity(Pf_hlrf, pi_pdf,beta_hlrf, alpha_hlrf, ...
                                                          compute_Sobol, compute_EVPPI);

% using MATLAB fmincon
[S_F1_fmc, S_F1_T_fmc, S_EVPPI_fmc] = FORM_Sensitivity(Pf_fmc, pi_pdf, beta_fmc, alpha_fmc, ...
                                                       compute_Sobol,compute_EVPPI);
                                       
%% Reference values
% The reference values for the first order indices
S_F1_ref   = [0.0315, 0.0315];

% Print reference values for the first order indices
fprintf("***\n\nReference first order Sobol' indices: ***\n");
disp(S_F1_ref);

% show p_f results
fprintf('***Exact Pf: %g ***\n', pf_ex);
fprintf('***FORM HLRF Pf: %g ***\n', Pf_hlrf);
fprintf('***FORM fmincon Pf: %g ***\n\n', Pf_fmc);

%% plot HLRF results
if d == 2
    % grid points
    uu      = 0:0.05:5;
    [U1,U2] = meshgrid(uu,uu);
    nnu     = length(uu);
    unod    = cat(2, reshape(U1,nnu^2,1),reshape(U2,nnu^2,1));
    ZU      = g(unod);
    ZU      = reshape(ZU,nnu,nnu);

    figure;  hold on; 
    pcolor(U1,U2,ZU); shading interp;
    contour(U1,U2,ZU,[0 0],'r');
    plot(0,0,'r*',u_star_hlrf(1),u_star_hlrf(2),'r*');   % design point in standard
    line([0, u_star_hlrf(1)],[0, u_star_hlrf(2)]);       % reliability index beta
    %
    plot(0,0,'ko',u_star_fmc(1),u_star_fmc(2),'ko');   % design point in standard
    line([0, u_star_fmc(1)],[0, u_star_fmc(2)]);       % reliability index beta
    xlabel('$u_1$','Interpreter','Latex','FontSize', 18);
    ylabel('$u_2$','Interpreter','Latex','FontSize', 18); 
    set(get(gca,'ylabel'),'rotation',0);
    axis equal tight;
    title('Standard space');
end
%%END