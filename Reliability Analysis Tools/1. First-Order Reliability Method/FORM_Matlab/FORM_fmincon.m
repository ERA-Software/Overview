function [u_star, x_star, beta, Pf, S_F1, S_F1_T] = FORM_fmincon(g, dg, distr, sensitivity_analysis, u0, tol, maxit)
%% optimization using the fmincon function
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe
Matthias Willer
Daniel Koutas
Max Ehre
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2018-05
---------------------------------------------------------------------------
Changelog 
2022-04: * included first order and total-effect Sobol' indices computation
2023-05: * added optional inputs to control design point search parameters
         * added option to pass analytical LSF gradient
---------------------------------------------------------------------------
Comment:
* The FORM method uses a first order approximation of the LSF and is 
  therefore not accurate for non-linear LSF's
---------------------------------------------------------------------------
Input:
* g                    : limit state function in the original space
* dg                   : anonynmous function containing the analytical 
                         gradient of the limit state function (optional)
* distr                : ERANataf-Object containing the distribution
* sensitivity_analysis : implementation of sensitivity analysis: 
                         1 - perform, 
                         0 - not perform

Optional (optimization parameters for design point search):

* u0                   : initial guess 
* tol                  : tolerance 
* maxit                : maximum number of iterations
---------------------------------------------------------------------------
Output:
* u_star : design point in the standard space
* x_star : design point in the original space
* beta   : reliability index
* Pf     : probability of failure
* S_F1   : vector of first-order indices
* S_F1_T : vector of total-effect indices
---------------------------------------------------------------------------
%}

%% initial check if there exists a Nataf object
if ~(any(strcmp('Marginals',fieldnames(distr))) == 1)
	return;
end
d = length(distr.Marginals); 

%% set optional inputs if not passed
if ~exist('u0', 'var')
    u0  = repmat(0.1,d,1);   % default starting point point
end

if ~exist('tol', 'var')
    tol = 1e-6; % default tolerance
end

if ~exist('maxit', 'var')
    maxit = 5e2; % default max. number of iterations
end

%% objective function
dist_fun = @(u) norm(u);

%% parameters of the fmincon function
A   = [];                % linear equality constraints
b   = [];                % linear equality constraints
Aeq = [];                % linear inequality constraints
beq = [];                % linear inequality constraints
lb  = [];                % lower bound constraints
ub  = [];                % upper bound constraints

% nonlinear constraint: H(u) <= 0
H      = @(u) g(distr.U2X(u));

if isempty(dg)
    lsfcon = @(u) deal([], H(u'));
else
    dG_dU  = @(u) dG_dU_fun(u,distr,dg);
    lsfcon = @(u) deal(dG_dU(u'), H(u'));
end

%% use fmincon
options = optimoptions('fmincon','Display','off','Algorithm','sqp','StepTolerance',tol,'MaxIterations',maxit);
[u_star,beta,~,output] = fmincon(dist_fun,u0,A,b,Aeq,beq,lb,ub,lsfcon,options);

iter = output.iterations;
alg  = output.algorithm;

% compute design point in orignal space and failure probability
x_star = distr.U2X(u_star);
Pf     = normcdf(-beta);

%% sensitivity analysis
if sensitivity_analysis == 1
    [S_F1, S_F1_T, exitflag, errormsg] = FORM_Sobol_indices(u_star/beta, beta, Pf);
else
    S_F1 = [];
    S_F1_T = [];
end

%% print results
fprintf('*fmincon with %s Method\n',alg);
fprintf(' %g iterations... Reliability index = %g --- Failure probability = %g\n\n',iter,beta,Pf);

% print first order and total-effect indices
if sensitivity_analysis == 1
    if any(strcmp('Marginals',fieldnames(distr)))
        if ~isequal(distr.Rho_X, eye(length(distr.Marginals)))
            fprintf("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING: !!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            fprintf("Results of sensitivity analysis do not apply for dependent inputs.")
            fprintf("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
        end
    end
    
    if exitflag == 1
        fprintf(" First order indices: \n");
        disp(S_F1);
        fprintf(" Total-effect indices: \n");
        disp(S_F1_T);
    else
        fprintf('Sensitivity analysis could not be performed, because: \n')
        fprintf(errormsg);
    end
end

return

function grad = dG_dU_fun(u,distr,dg)
   [x, J] = distr.U2X(u, 'Jac');
   grad    = J * dg(x);
return
