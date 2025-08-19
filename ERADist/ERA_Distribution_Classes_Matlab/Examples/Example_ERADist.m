%% Example file: Definition and use of ERADist objects
%{
In this example a lognormal distribution is defined by its parameters,
moments and data.Furthermore the different methods of ERADist are
illustrated.
For other distributions and more information on ERADist please have a look
at the provided documentation or execute the command "help ERADist".
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

%% Definition of an ERADist object by the distribution parameters

dist = ERADist('lognormal','PAR',[2,0.5])

% computation of the first two moments
mean_dist = dist.mean
std_dist = dist.std

% generation of n random samples
n = 10000;
samples = dist.random(n,1);

%% Definition of an ERADist object by the first moments
% Based on the just determined moments a new distribution object with the
% same properties is created..

dist_mom = ERADist('lognormal','MOM',[mean_dist,std_dist])

%% Definition of an ERADist object by data fitting
% Using maximum likelihood estimation a new distribution object is created
% from the samples which were created above.

dist_data = ERADist('lognormal','DATA',samples)

%% Other methods

% generation of five samples x to work with
x = dist.random(5,1)

% computation of the PDF for the samples x
pdf = dist.pdf(x)

% computation of the CDF for the samples x
cdf = dist.cdf(x)

% computation of the inverse CDF based on the CDF values (-> initial x)
icdf = dist.icdf(cdf)

%% Plot of the PDF and CDF

x_plot = 0:0.1:40;          % values for which the PDF and CDF are evaluated 
PDF = dist.pdf(x_plot);     % computation of PDF
CDF = dist.cdf(x_plot);     % computation of CDF

figure()
% plot of the PDF
subplot(1,2,1)
plot(x_plot,PDF)
xlabel('$X$','Interpreter','Latex','FontSize', 18);
ylabel('PDF','Interpreter','Latex','FontSize', 18);
set(gca,'FontSize',15);

subplot(1,2,2)
% plot of the CDF
plot(x_plot,CDF)
xlabel('$X$','Interpreter','Latex','FontSize', 18);
ylabel('CDF','Interpreter','Latex','FontSize', 18);
set(gca,'FontSize',15);