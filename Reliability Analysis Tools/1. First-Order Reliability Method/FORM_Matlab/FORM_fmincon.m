function [u_star, x_star, beta, alpha, Pf] = FORM_fmincon(g, dg, distr, u0, tol, maxit)
%% optimization using the fmincon function
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe
Matthias Willer
Daniel Koutas
Max Ehre
Ivan Olarte-Rodriguez
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2018-05
---------------------------------------------------------------------------
Changelog 
2023-08: * added optional inputs to control design point search parameters
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
* alpha  : vector with the values of FORM indices
* Pf     : probability of failure

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
dist_fun = @(u) norm(u,2);

%% parameters of the fmincon function
A   = [];                % linear equality constraints
b   = [];                % linear equality constraints
Aeq = [];                % linear inequality constraints
beq = [];                % linear inequality constraints


% nonlinear constraint: H(u) <= 0
H      = @(u) g(distr.U2X(u));
lsfcon = @(u) deal(H(u'),[]);

if isempty(dg)
    % autograd gradient evaluation
    dg    = @(x) extractdata(dlfeval(@(x) dlgradient(g(x),x),dlarray(x)))';
    fd_grad = 0;
    % test if autograd works on given LSF
    try
        test_dg = dg(distr.U2X(randn(d,1)));
        dG_dU  = @(u) dG_dU_fun(u,distr,dg);
    catch err_msg
        % use finite differences if autograd fails
        fd_grad = 1;
        eps     = @(gg) 1e-4*max(abs(gg),1e-6);
        dg      = @(x,gg) (g(x + diag(eps(gg)*ones(d,1))) - gg) / eps(gg);
    end

    if fd_grad
        dG_dU  = @(u) dG_dU_fun(u,distr,dg(u,H(u')));    
    end
    

else
    dG_dU  = @(u) dG_dU_fun(u,distr,dg);

end

% Clear some memory
clear test_dg;

%% use fmincon

init_alg = "sqp";
options = optimoptions('fmincon','Display','off','Algorithm',init_alg,...
    'StepTolerance',tol,'MaxIterations',maxit,'CheckGradients',true,...
    "ConstraintTolerance",1e-10,"FiniteDifferenceType","central");
[u_star,beta,~,output] = fmincon(dist_fun,u0,A,b,Aeq,beq,repelem(-10,d),repelem(10,d),lsfcon,options);


iter = output.iterations;
alg  = output.algorithm;

% compute design point in original space and failure probability
x_star = distr.U2X(u_star);

Pf     = normcdf(-beta);


%alpha = u_star/beta;
alpha = dG_dU(u_star)./(norm(dG_dU(u_star),2));


%% print results
fprintf('*fmincon with %s Method\n',alg);
fprintf(' %g iterations... Reliability index = %g --- Failure probability = %g\n\n',iter,beta,Pf);

end

%---------- Nested Functions ----------------------------------------------

% Gradient Function
function grad = dG_dU_fun(u,distr,dg)
   [x, J] = distr.U2X(u, 'Jac');
   grad    = J * dg(x);
end
