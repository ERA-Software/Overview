%% FORM using HLRF algorithm and fmincon: Ex. 3 Ref. 3 - series system reliability problem
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
g = @(x) min([0.1.*(x(:,1)-x(:,2)).^2-(x(:,1)+x(:,2))./sqrt(2)+3,...
                  0.1.*(x(:,1)-x(:,2)).^2+(x(:,1)+x(:,2))./sqrt(2)+3,...
                  x(:,1)-x(:,2)+7./sqrt(2),...
                  x(:,2)-x(:,1)+7./sqrt(2)],[],2);



%% Solve the optimization problem of the First Order Reliability Method 
% OPC 1. FORM using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[u_star_hlrf, x_star_hlrf, beta_hlrf, alpha_hlrf, Pf_hlrf ] = FORM_HLRF(g, [], pi_pdf);

% OPC 2. FORM using MATLAB fmincon (without analytical gradient)
[u_star_fmc, x_star_fmc, beta_fmc, alpha_fmc, Pf_fmc]= FORM_fmincon(g, [], pi_pdf);

% OPC 3. FORM using MATLAB fmincon (with analytical gradient)
%[u_star_fmc, x_star_fmc, beta_fmc, alpha_fmc, Pf_fmc] = FORM_fmincon(g, dg, pi_pdf);

%% Computation of Sensitivity (First Order Indices)

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
S_F1_ref   = [0.0481, 0.0481];

% Print reference values for the first order indices
fprintf("\n\n***Reference first order Sobol' indices: ***\n");
disp(S_F1_ref);

% reference solution
pf_ref = 2.26e-3;

% show p_f results
fprintf('***Reference Pf: %g ***', pf_ref);
fprintf('\n***FORM HLRF Pf: %g ***', Pf_hlrf);
fprintf('\n***FORM fmincon Pf: %g ***\n\n', Pf_fmc);


%% Plot
if d == 2
    % grid points
    uu      = -7:0.05:7;
    [U1,U2] = meshgrid(uu,uu);
    nnu     = length(uu);
    unod    = cat(2, reshape(U1,nnu^2,1),reshape(U2,nnu^2,1));
    ZU      = g(unod);
    ZU      = reshape(ZU,nnu,nnu);

    figure;  hold on; 
    pcolor(U1,U2,ZU); shading interp;
    contour(U1,U2,ZU,[0 0],'r');
    plot(0,0,'ko',u_star_fmc(1),u_star_fmc(2),'ko');   % design point in standard
    line([0, u_star_fmc(1)],[0, u_star_fmc(2)]);       % reliability index beta
    xlabel('$u_1$','Interpreter','Latex','FontSize', 18);
    ylabel('$u_2$','Interpreter','Latex','FontSize', 18); 
    set(get(gca,'ylabel'),'rotation',0);
    axis equal tight;
    title('Standard space');
end
%%END