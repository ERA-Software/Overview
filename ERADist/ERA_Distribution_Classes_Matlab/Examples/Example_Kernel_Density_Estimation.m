%% Example file: Computing and Plotting PDFs and CDFs with Kernel Density Estimation
%{
The purpose of this MATLAB script is to show the user how to compute and plot
PDFs and CDFs with kernel density estimation. It should be noted that these
functions were not developed by the ERA Group but come with the ‘Statistics
and Machine Learning Toolbox’ by MATLAB. The reason for presenting these
functions here is that it is not possible to implement an accurate joint
CDF method for the ERARosen class. Therefore, one can use these Kernel
Density Estimation functions. Nonetheless the shown methods are also not
completely accurate. The accuracy depends on a proper choice of the kernel,
the bandwidths, and the distribution bounds, which vary for the different 
distributions that should be estimated. Since kernel density estimation is 
a sampling based approach the estimates always have certain variability. 
These functions should therefore only be used by users that have enough
experience and knowledge about the problem the problem they are dealing with.
Additionally, the presented functions can be used to estimate the
marginalized PDFs and CDFs of the different variables of a multivariate 
distribution, like the ones created by ERARosen and ERANataf.
For a more detailed explanation of the kernel density estimation functions
of MATLAB, including bandwidths and bounds, please consult the respective
MATLAB documentation (https://mathworks.com/help/stats/ksdensity.html 
and https://mathworks.com/help/stats/mvksdensity.html).
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
Version 2022-01
---------------------------------------------------------------------------
References:
1. Documentation of the ERA Distribution Classes
---------------------------------------------------------------------------
%}
clear; clc; close all;

%% Creation of the ERARosen object
% for a more detailed explanation on the definition of the ERARosen class
% have a look at the MATLABscript 'Example_ERARosen_and_ERACond'.

% marginal distributions defined by ERADist classes
A = ERADist('normal','PAR',[3,2],'A');
B = ERADist('normal','PAR',[5,4],'B');
C = ERADist('normal','PAR',[7,1],'C');

% conditional distributions defined by ERACond classes 
D = ERACond('normal','PAR',{@(x)x(:,1).*x(:,2)-x(:,3),2},'D');
E = ERACond('normal','PAR',{@(x)x(:,1)-x(:,2)+3*x(:,3),@(x)abs(x(:,1))},'E');

% joint distribution defined by an ERARosen class
dist = {A,B,C,D,E};
depend = {[],[],[],[1,2,3],[1,2,3]};
X_dist = ERARosen(dist,depend);

% plot of the dependency between the different marginal and conditional distributions
X_dist.plotGraph();

%% Generation of joint samples
% Since kernel density estimation is a sampling based approach the user
% must generate a finite amount of samples first.

n = 100000; % number of samples
X_samples = X_dist.random(n); % joint samples

%% Multivariate density estimation
% In MATLAB the kernel density estimation of a multivariate distribution
% can be carried out by the function 'mvksdensity'. The input of the
% bandwidth of every variable of the joint distribution is mandatory. A
% method to estimate an appropriate bandwidth is Silberman's rule of thumb.
% The different bandwidths according to this rule depend on the number of
% samples, the number of dimension and the standard deviation of the
% respective variable. The joint CDF is evaluated at the point
% x = (10,10,10,10,10).

ev_pt = [10,10,10,10,10]; % evaluation point

% bandwidths according to Silberman's rule of thumb
n_dim = 5; % number of dimensions of the joint distribution
bw = std(X_samples,1)*(4/((n_dim+2)*n)).^(1/(n_dim+4)); % computation of all five bandwidths

% estimate of the joint CDF at the evaluation point
jointCDF = mvksdensity(X_samples,ev_pt,'Bandwidth',bw,'Function','cdf')

%% Marginalization of single variables of the joint distribution
% The following approaches are shown with single variables of a joint
% distribution defined by an ERARosen objects. Anyhow, the approaches could
% also be interesting for single variables of a joint distributions defined
% by an ERANataf object.

% The distributions D and E were initially defined as conditional
% distributions depending on the distributions A, B and C. By using the
% samples of the variables D and E, which were obtained by sampling the 
% joint distribution X with the help of kernel density estimation, the
% marginal PDFs and CDFs of D and E can be obtained.
% The input of a bandwidth is not mandatory when using the function
% 'ksdensity' for the density estimation of an univariate distribution.

D_samples = X_samples(:,4);     % samples of variable D obtained from X
E_samples = X_samples(:,5);     % samples of variable E obtained from X

% plotting the marginal PDFs and CDFs of D and E
figure()
subplot(1,2,1)
ksdensity(D_samples)
hold on
ksdensity(E_samples)
xlabel('$X$','Interpreter','Latex','FontSize', 18);
ylabel('PDF','Interpreter','Latex','FontSize', 18);
hl = legend({'D','E'},'Location','Best'); set(hl,'Interpreter','latex'); 
set(gca,'FontSize',15);
subplot(1,2,2)
ksdensity(D_samples,'Function','cdf')
hold on
ksdensity(E_samples,'Function','cdf')
xlabel('$X$','Interpreter','Latex','FontSize', 18);
ylabel('CDF','Interpreter','Latex','FontSize', 18);
hl = legend({'D','E'},'Location','Best'); set(hl,'Interpreter','latex'); 
set(gca,'FontSize',15);

% It is also possible to evaluate the PDFs and CDFs values at specific 
% points that are of interest for the user. In the following the marginal
% PDFs and CDFs for both variables D and E is estimated at the points
% [0,10,20,30,40].

pts = [0,10,20,30,40]; % evaluation points

% evaluation of the PDFs and CDFs
PDF_D = ksdensity(D_samples,pts)
CDF_D = ksdensity(D_samples,pts,'Function','cdf')
PDF_E = ksdensity(E_samples,pts)
CDF_E = ksdensity(E_samples,pts,'Function','cdf')
