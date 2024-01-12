%% example 3: parameter identification two-DOF shear building
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe
Luca Sardi
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Current version 2021-03
* use of log-evidence logcE
---------------------------------------------------------------------------
References:
1."Transitional Markov Chain Monte Carlo: Observations and improvements". 
   Wolfgang Betz et al.
   Journal of Engineering Mechanics. 142.5 (2016) 04016016.1-10
2."Transitional Markov Chain Monte Carlo method for Bayesian model updating, 
   model class selection and model averaging". 
   Jianye Ching & Yi-Chun Chen
   Journal of Engineering Mechanics. 133.7 (2007) 816-832
3."Bayesian inference with subset simulation: strategies and improvements"
   Wolfgang Betz et al.
   Computer Methods in Applied Mechanics and Engineering 331 (2018) 72-93.
---------------------------------------------------------------------------
%}
clear; clc; close all;

% add path to ERADist and ERANataf classes
% Source: https://www.bgu.tum.de/era/software/eradist/
% addpath('../../../ERA_Dist/ERADistNataf_MATLAB/')

%% model data
% shear building data
m1 = 16.5e3;     % mass 1st story [kg]
m2 = 16.1e3;     % mass 2nd story [kg]
kn = 29.7e6;     % nominal values for the interstory stiffnesses [N/m]

%% prior PDF for X1 and X2 (product of lognormals)
mod_log_X1 = 1.3;   % mode of the lognormal 1
std_log_X1 = 1.0;   % std of the lognormal 1
mod_log_X2 = 0.8;   % mode of the lognormal 2
std_log_X2 = 1.0;   % std of the lognormal 2

% find lognormal X1 parameters
var_fun = @(mu) std_log_X1^2 - (exp(mu-log(mod_log_X1))-1)...
                              .*exp(2*mu+(mu-log(mod_log_X1)));
mu_X1  = fzero(var_fun,0);             % mean of the associated Gaussian
std_X1 = sqrt(mu_X1-log(mod_log_X1));  % std of the associated Gaussian

% find lognormal X2 parameters
var_X2 = @(mu) std_log_X2^2 - (exp(mu-log(mod_log_X2))-1)...
                             .*exp(2*mu+(mu-log(mod_log_X2)));
mu_X2  = fzero(var_X2,0);              % mean of the associated Gaussian
std_X2 = sqrt(mu_X2-log(mod_log_X2));  % std of the associated Gaussian

%% definition of the random variables
n = 2;         % number of dimensions

% assign data: 1st and 2nd variables are Lognormal
pi_pdf(1) = ERADist('lognormal','PAR',[mu_X1, std_X1]);
pi_pdf(2) = ERADist('lognormal','PAR',[mu_X2, std_X2]); 

% correlation matrix
R = eye(n);   % independent case

% object with distribution information
pi_pdf = ERANataf(pi_pdf,R);

%% likelihood function
lambda  = [1, 1]';            % means of the prediction error
i       = 9;                  % simulation level
var_eps = 0.5^(i-1);          % variance of the prediction error
f_tilde = [3.13, 9.83]';      % measured eigenfrequencies [Hz]

% shear building model 
f = @(x) shear_building_2DOF(m1, m2, kn*x(1), kn*x(2));

% modal measure-of-fit function
J = @(x) sum((lambda.^2).*(((f(x).^2)./f_tilde.^2) - 1).^2);   

% likelihood function
likelihood     = @(x) exp(-J(x)/(2*var_eps));
log_likelihood = @(x) -J(x)/(2*var_eps);

%% TMCMC
Ns = 1e3;        % number of samples per level
Nb = 0.1*Ns;     % burn-in period

% run the iTMCMC.m function
fprintf('\niTMCMC: \n');
[samplesU, samplesX, q, logcE] = iTMCMC(Ns, Nb, log_likelihood, pi_pdf);

%% show results
% reference solutions
mu_exact    = 1.12;   % for x_1
sigma_exact = 0.66;   % for x_1
cE_exact    = 1.52e-3;

fprintf('\nExact model evidence = %g',cE_exact);
fprintf('\nModel evidence = %g\n',exp(logcE));
fprintf('\nExact posterior mean x_1 = %g',mu_exact);
fprintf('\nMean value of x_1 = %g\n',mean(samplesX{end}(:,1)));
fprintf('\nExact posterior std x_1 = %g',sigma_exact);
fprintf('\nStd of x_1 = %g\n\n',std(samplesX{end}(:,1)));

%% plots
m = length(q);   % number of stages (intermediate levels)
% plot p values
figure;
plot(0:m-1,q,'ro-');
xlabel('Intermediate levels $j$','Interpreter','Latex','FontSize', 18);
ylabel('$q_j$','Interpreter','Latex','FontSize', 18);
set(gca,'FontSize',15); axis tight;
   
% plot samples increasing q
idx = [1, round(m/3), round(2*m/3), m];
figure;
for i = 1:4
   subplot(2,2,i); plot(samplesX{idx(i)}(:,1),samplesX{idx(i)}(:,2),'b.');
   title(sprintf('$q_j$=%4.3f',q(idx(i))),'Interpreter','Latex','FontSize', 18);
   xlabel('$\theta_1$','Interpreter','Latex','FontSize', 18);
   ylabel('$\theta_2$','Interpreter','Latex','FontSize', 18);
   set(gca,'FontSize',15);  axis equal;  xlim([0 3]); ylim([0 2]);
end
%%END
