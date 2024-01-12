function [w] = shear_building_2DOF(m1,m2,k1,k2)
%% two-degree-of-freedom shear building model
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
EXAMPLE:
m1 = 16.531e3;   m2 = 16.131e3;
k1 = 29.7e6;     k2 = 29.7e6;
[w,Phi] = shear_building_2DOF(m1,m2,k1,k2)
---------------------------------------------------------------------------
References:
1."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2014) 1-13.
---------------------------------------------------------------------------
%}

%% Mass and Stiffness matrices
M = [ m1,  0; 
       0, m2  ];        % mass matrix
K = [ k1 + k2, -k2;
       -k2   ,  k2 ];   % stiffness matrix

%% Free response - Modal analysis 
[V,L]   = eig(M\K);               % eigenvalue solution
w       = sqrt(diag(L))/(2*pi);   % natural frequencies [Hz]
[w,idx] = sort(w);                % ordering (ascendent)
% Phi     = V(:,idx);               % vibration modes

% Normalizing modes
% q   = Phi'*M*Phi;                 % orthogonality property
% Phi = Phi./repmat(sqrt(diag(q))',2,1);

return;
%%END