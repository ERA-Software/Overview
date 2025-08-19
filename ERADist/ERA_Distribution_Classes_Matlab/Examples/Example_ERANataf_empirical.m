%% Example file: Use of ERADist empirical distribution in ERANataf
%{
In this script the definition of a joint distribution object including an
empirical distribution based on a dataset with the ERANataf class and the
use of its methods are shown. For more information on ERANataf please have
a look at the provided documentation. 
---------------------------------------------------------------------------
Developed by: 
Michael Engel

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Initial Version 2025-07
---------------------------------------------------------------------------
References:
1. Documentation of the ERA Distribution Classes
---------------------------------------------------------------------------
%}

clear all; close all; clc;

%% Main script
rng(2025);   % initialize random number generator
n         = 2000;   % number of datapoints
n_plot = 1000;   % number of samples for plotting

% --- generate a bimodal Gaussian mixture dataset
data = sample_bimodal_gaussian(n, [0.2, 0.8], [-2, 3], [0.5, 1.0]);

%% Definition of the ERANataf object

% 1. Define marginal distributions
M(1) = ERADist('normal', 'PAR', [4,2]);
M(2) = ERADist('gumbel', 'MOM', [1,2]);
M(3) = ERADist('empirical', 'DATA', {data, [], 'linear', [], struct()});

% 2. Define correlation matrix
Rho = [1.0, 0.5, 0.5;
       0.5, 1.0, 0.5;
       0.5, 0.5, 1.0];

% 3. Define joint distribution
T_Nataf = ERANataf(M, Rho);

%% Methods

% Generate random samples
X = T_Nataf.random(5);
disp('X = '), disp(X)

% Joint PDF
PDF_X = T_Nataf.pdf(X);
disp('PDF = '), disp(PDF_X)

% Joint CDF
CDF_X = T_Nataf.cdf(X);
disp('CDF = '), disp(CDF_X)

% Transformation X->U with Jacobian
[U, Jac_X2U] = T_Nataf.X2U(X, 'Jac');
disp('U = '), disp(U)
disp('Jac_X2U = '), disp(Jac_X2U)

% Transformation U->X with Jacobian
[X_back, Jac_U2X] = T_Nataf.U2X(U, 'Jac');
disp('X backtransformed = '), disp(X_back)
disp('Jac_U2X = '), disp(Jac_U2X)

%% Plot: physical space vs. standard normal space
X_plot = T_Nataf.random(n_plot);
U_plot = T_Nataf.X2U(X_plot);

figure('Position',[100 100 1600 800]);

subplot(1,2,1)
scatter3(X_plot(:,1), X_plot(:,2), X_plot(:,3), 20, 'b', 'filled');
title('Physical space');
xlabel('$X_1$','Interpreter','latex');
ylabel('$X_2$','Interpreter','latex');
zlabel('$X_3$','Interpreter','latex');
grid on; axis equal;

subplot(1,2,2)
scatter3(U_plot(:,1), U_plot(:,2), U_plot(:,3), 20, 'r', 'filled');
title('Standard normal space');
xlabel('$U_1$','Interpreter','latex');
ylabel('$U_2$','Interpreter','latex');
zlabel('$U_3$','Interpreter','latex');
grid on; axis equal;


%% Helper function: sample from bimodal Gaussian
function data = sample_bimodal_gaussian(n_samples, mix_weights, means, stds)
    comps = randsample([0, 1], n_samples, true, mix_weights);
    data = zeros(n_samples,1);
    idx0 = comps == 0;
    idx1 = comps == 1;
    data(idx0) = normrnd(means(1), stds(1), sum(idx0), 1);
    data(idx1) = normrnd(means(2), stds(2), sum(idx1), 1);
end

