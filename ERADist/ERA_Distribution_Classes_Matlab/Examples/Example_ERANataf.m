%% Example file: Definition and use of ERADist objects
%{
 In this script the definition of a joint distribution object with the
 ERANataf class and the use of its methods are shown.
 For more information on ERANataf please have a look at the provided
 documentation or execute the command "help ERANataf" 
---------------------------------------------------------------------------
Developed by:
Antonios Kamariotis (antonis.kamariotis@tum.de)
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Luca Sardi
Nicola Bronzetti
Alexander von Ramm
Matthias Willer
Peter Kaplan

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2020-10
---------------------------------------------------------------------------
References:
1. Documentation of the ERA Distribution Classes
---------------------------------------------------------------------------
%}
clear; clc; close all;

%% Definition of the ERANataf object
rng(5)
% definition of the marginal distributions
M(1) = ERADist('normal','PAR',[4,2]);
M(2) = ERADist('gumbel','MOM',[1,2]);
M(3) = ERADist('exponential','PAR',4);

% definition of the correlation matrix
Rho  = [ 1.0 0.5 0.5;
         0.5 1.0 0.5;
         0.5 0.5 1.0 ];

% definition of the joint distribution    
T_Nataf = ERANataf(M,Rho)

%% Methods

% generation of five random samples to work with
X = T_Nataf.random(5)

% computation of joint PDF
PDF_X = T_Nataf.pdf(X)

% computation of joint CDF
CDF_X = T_Nataf.cdf(X)

% transformation from physical space X to the standard normal space U and
% Jacobian of the transformation of the first sample
[U,Jac_X2U] = T_Nataf.X2U(X,'Jac')

% transformation from standard normal space U to physical space X and
% Jacobian of the transformation of the first sample
[X_backtransform,Jac_U2X] = T_Nataf.U2X(U,'Jac')

%% Creation of samples in physical space, transformation to standard normal
% space and plot of the samples to show the isoprobabilistic transformation

n=1000; % number of samples

% generation of n random samples
X_plot = T_Nataf.random(n);
% transformation from physical space X to the standard normal space U 
U_plot = T_Nataf.X2U(X_plot);

figure;
subplot(121); plot3(X_plot(:,1),X_plot(:,2),X_plot(:,3),'b.');
title('Physical space','Interpreter','Latex','FontSize', 18);
xlabel('$X_1$','Interpreter','Latex','FontSize', 18);
ylabel('$X_2$','Interpreter','Latex','FontSize', 18);
zlabel('$X_3$','Interpreter','Latex','FontSize', 18);
set(gca,'FontSize',15);
subplot(122); plot3(U_plot(:,1),U_plot(:,2),U_plot(:,3),'r.'); axis equal;
title('Standard space','Interpreter','Latex','FontSize', 18);
xlabel('$U_1$','Interpreter','Latex','FontSize', 18);
ylabel('$U_2$','Interpreter','Latex','FontSize', 18);
zlabel('$U_3$','Interpreter','Latex','FontSize', 18);
set(gca,'FontSize',15);
