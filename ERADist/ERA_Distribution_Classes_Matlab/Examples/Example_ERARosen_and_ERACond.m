%% Example file: Definition and use of ERARosen and ERACond objects
%{
 In this script the definition of a multivariate distribution with the
 ERARosen class using ERACond and ERADist distribution objects is shown.
 For more information on ERARosen and ERACond please have a look at the
 provided documentation or execute the commands "help ERARosen" or 
 "help ERACond" respectively.
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

%% Creating the marginal and conditional distribution objects

% marginal distributions defined by ERADist classes
x1_dist = ERADist('normal','PAR',[3,2],'A');
x2_dist = ERADist('normal','PAR',[5,4],'B');

% conditional distributions defined by ERACond classes
a=3; % use of a constant in function handle for demonstration purposes
x3_dist = ERACond('normal','PAR',{@(X)X(:,1).*X(:,2)+a, 2},'C');
% use of a user-defined function for demonstration purposes in x4_dist
x4_dist = ERACond('normal','PAR',{@(X)subtraction(X(:,1),X(:,2)), @(X)abs(X(:,1))},'D');
% both alternatives for x5_dist work (definition as function handle and definition as 1x1 cell array)
x5_dist = ERACond('exponential','PAR',@(X)abs(X(:,1).^2-X(:,2)),'E');
%x5_dist = ERACond('exponential','PAR',{@(X)abs(X(:,1).^2-X(:,2))},'E'); % both alternatives for x5_dist work (definition as function handle and definition as 1x1 cell array)
x6_dist = ERACond('normal','PAR',{@(X)3*X, 4},'F');
x7_dist = ERACond('normal','PAR',{@(X)X(:,1)+X(:,2)-X(:,3), 1},'G');

% % alternative for conditional distributions with more complicated parameter functions  
% x3_dist = ERACond('normal','PAR',{@(X)max(X,[],2), 2},'C');
% x4_dist = ERACond('normal','PAR',{@(X)sqrt(sum(X.^2,2)), @(X)abs(X(:,1))},'D');
% x5_dist = ERACond('normal','PAR',{@(X)mean(X.^2,2).*mean(X,2), @(X)X(:,1).^2},'E');
% x6_dist = ERACond('normal','PAR',{@(X)3*X, 4},'F');
% A=1; B=[1;2;3]; C=[2,1,0;1,2,1;0,1,2]; D=[3;4;5];
% x7_dist = ERACond('normal','PAR',{@(Y)A+B'/C*(Y'-D), 1},'G');

 % collecting all the distribution objects in a cell array
dist = {x1_dist,x2_dist,x3_dist,x4_dist,x5_dist,x6_dist,x7_dist}

%% Describing the dependency and creating the ERARosen class

% describing the dependency by parents using a cell array
depend = {[],[],[1,2],[1,3],[3,2],4,[3,4,5]};

% creation of the ERARosen class
X_dist= ERARosen(dist,depend)

%% Methods of the ERARosen class

% plot of the graph definining the dependency in the distribution

% ...with naming of the nodes according to their order in input dist
% (overwrites the ID of the distribution)
figure_numb = X_dist.plotGraph('numbering');

% ... with naming of the nodes according to their ID
figure = X_dist.plotGraph();

% creation of n samples of the joint distribution
rng(10) % initializing random number generator
n = 5;
X = X_dist.random(n)

% transformation from physical space X to the standard normal space U
U = X_dist.X2U(X)
 
% transformation from standard normal space U to physical space X
X_backtransform = X_dist.U2X(U)

% computation of joint PDF
PDF = X_dist.pdf(X)

%% user-defined function for demonstration purposes

function c=subtraction(a,b)
c=a-b;
end