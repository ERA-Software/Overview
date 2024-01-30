function [u_star,x_star,beta,alpha,Pf] = FORM_HLRF(g, dg, distr, u0, tol, maxit)
%% HLRF function
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
2023-05: * added optional inputs to control design point search parameters
         * include autograd and finite difference methods for LSF gradient
---------------------------------------------------------------------------
Comment:
* The FORM method uses a first order approximation of the LSF and is 
  therefore not accurate for non-linear LSF's
---------------------------------------------------------------------------
Input:
* g                    : limit state function
* dg                   : anonynmous function containing the analytical 
                         gradient of the limit state function (optional)
                         - if not provided, automatic differentiation or
                         finite differences are used
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
* S_F1   : vector of first-order indices
* S_F1_T : vector of total-effect indices
---------------------------------------------------------------------------
References:
1. "Structural reliability under combined random load sequences."
   Rackwitz, R., and B. Fiessler (1979).    
   Computers and Structures, 9.5, pp 489-494
---------------------------------------------------------------------------
%}

%% initial check if there exists a Nataf object
if ~(any(strcmp('Marginals',fieldnames(distr))))   % use Nataf transform (dependence)
	return;
end
d = length(distr.Marginals);    % number of random variables (dimension)

%% set optional inputs if not passed
if ~exist('u0', 'var') || isempty(u0)
    u0  = repmat(0.1,d,1);   % default starting point point
end

if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-6; % default tolerance
end

if ~exist('maxit', 'var') || isempty(maxit)
    maxit = 5e2; % default max. number of iterations
end

%% Get the length
d = length(distr.Marginals);    % number of random variables (dimension)

%% determine how to evaluate LSF gradients
fd_grad = 0;

if isempty(dg)
    % autograd gradient evaluation
    dg    = @(x) extractdata(dlfeval(@(x) dlgradient(g(x),x),dlarray(x)))';
    
    % test if autograd works on given LSF
    try
        test_dg = dg(distr.U2X(randn(d,1)));
        clear test_dg;
    catch err_msg
        % use finite differences if autograd fails
        fd_grad = 1;
        eps     = @(gg) 1e-4*max(abs(gg),1e-6);
        dg      = @(x,gg) (g(x + diag(eps(gg)*ones(d,1))) - gg) / eps(gg);
    end
end
   

%% initialization
beta  = zeros(1,maxit);

%% HLRF method
k = 1;

u = u0;

while true
   % 0. Get x and Jacobi from u (important for transformation)
   [xk, J] = distr.U2X(u(:,k), 'Jac');
   
   % 1. evaluate LSF at point u_k
   H_uk = g(xk);
   
   % 2. evaluate LSF gradient at point u_k and direction cosines
   if fd_grad
       DH_uk      = J * dg(xk,H_uk);
   else
       DH_uk      = J * dg(xk);
   end

   norm_DH_uk = norm(DH_uk,2);
   alpha      = DH_uk/norm_DH_uk;
   
   % 3. calculate beta
   beta(k) = -u(:,k)'*alpha + (H_uk/norm_DH_uk);
   
   % 4. calculate u_{k+1}
   u(:,k+1) = -beta(k)*alpha;
   
   % next iteration
   if (norm(u(:,k+1)-u(:,k),2) <= tol)  || (k == maxit)
      break;
   else
      k = k+1;
   end
end

% delete unnecessary data
u(:,k+2:end) = [];

% compute design point, reliability index and Pf
u_star = u(:,end);
x_star = distr.U2X(u_star);
beta   = beta(k);
Pf     = normcdf(-beta,0,1);


%% print results
fprintf('*FORM Method\n');
fprintf(' %g iterations... Reliability index = %g --- Failure probability = %g\n',k,beta,Pf);
% Check if final design point lies on LSF
if ~(abs(g(x_star)) <= 1e-6)
    fprintf('Warning! HLRF may have converged to wrong value! The LSF of the design point is: %g\n', g(x_star));
end
fprintf('\n')


return;