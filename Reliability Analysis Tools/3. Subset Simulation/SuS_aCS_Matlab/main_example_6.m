%% Subset Simulation: Ex. 5 Ref. 2 - points outside of hypersphere
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe
Daniel Koutas
Ivan Olarte-Rodriguez

Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Current version 2023-12
* Modification to Sensitivity Analysis Calls
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SuS"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
2."Bayesian inference of engineering models"
   Wolfgang Betz
   Ph.D. Thesis.
---------------------------------------------------------------------------
%}
clear; close all; clc;

%% definition of the random variables
d      = 2;          % number of dimensions
pi_pdf = repmat(ERADist('standardnormal','PAR'), d, 1);   % n independent rv

% % correlation matrix
% R = eye(d);   % independent case
% 
% % object with distribution information
% pi_pdf = ERANataf(pi_pdf, R);    % if you want to include dependence

%% limit state function
r = 5.26;    % radius
m = 1;       % m in [0,4]
g = @(u) 1 - (sqrt(sum(u.^2,2))/r).^2 - (u(:,1)/r).*((1-(sqrt(sum(u.^2,2))/r).^m)./(1+(sqrt(sum(u.^2,2))/r).^m));


%% Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 1;

%% subset simulation
N  = 2000;        % Total number of samples for each level
p0 = 0.1;         % Probability of each subset, chosen adaptively

fprintf('SUBSET SIMULATION: \n');
[Pf_SuS, delta_SuS, b, Pf, b_sus, pf_sus, samplesU, samplesX, fs_iid] = SuS(N,p0,g,pi_pdf, samples_return);

%% Implementation of sensitivity analysis

% Computation of Sobol Indices
compute_Sobol = true;

% Computation of EVPPI (based on standard cost of failure (10^8) and cost
% of replacement (10^5)
compute_EVPPI = true;

[S_F1, S_EVPPI] = Sim_Sensitivity(fs_iid, Pf_SuS, pi_pdf, compute_Sobol,compute_EVPPI);

%% Reference values
% The reference values for the first order indices
S_F1_ref   = [0.1857, 0.1857];

% Print reference values for the first order indices
fprintf("\n\n***Reference first order Sobol' indices: ***\n");
disp(S_F1_ref);

% reference solution
pf_ref = 1e-6;

% show p_f results
fprintf('\n***Reference Pf: %g ***', pf_ref);
fprintf('\n***SuS Pf: %g ***\n\n', Pf_SuS);

%% Plots
if ~isempty(samplesU.total{1})
    m     = length(Pf);
    xx    = -6:0.05:6;
    nnp   = length(xx);
    [X,Y] = meshgrid(xx);
    xnod  = cat(2,reshape(X',nnp^2,1),reshape(Y',nnp^2,1));
    Z     = g(xnod);
    Z     = reshape(Z,nnp,nnp);
    figure; hold on;
    contour(X, Y, Z, [0,0], 'r', 'LineWidth',3);  % LSF
    for j = 1:m+1
       u_j_samples = samplesU.total{j};
       plot(u_j_samples(:,1), u_j_samples(:,2), '.');
       if samples_return == 1
           break
       end
    end
    xlabel('$u_1$','Interpreter','Latex','FontSize', 18);
    ylabel('$u_2$','Interpreter','Latex','FontSize', 18);
    set(get(gca,'ylabel'),'rotation',0);
    axis equal tight;
end

% Plot failure probability
figure; 
title('Failure probability estimate','Interpreter','Latex','FontSize', 20);
xlabel('Limit state function, $g$','Interpreter','Latex','FontSize', 18);   
ylabel('Failure probability, $P_f$','Interpreter','Latex','FontSize', 18);

% Plot failure probability: SuS
hold on;
semilogy(b_sus,pf_sus,'b--');           % curve
semilogy(b,Pf,'ko','MarkerSize',5);     % points
semilogy(0,Pf_SuS,'b*','MarkerSize',6);
semilogy(0,pf_ref,'ro','MarkerSize',8);
set(gca,'yscale','log'); axis tight;
hl = legend('SuS','Intermediate levels','Pf SuS','Pf Ref.','Location','SE');
set(hl,'Interpreter','latex'); set(gca,'FontSize',18);