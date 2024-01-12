function [Pr, lv, N_tot, gamma_hat, samplesU, samplesX, k_fin, S_F1] = CEIS_vMFNM(N, p, g_fun, distr, k_init, sensitivity_analysis, samples_return)
%% Cross entropy-based importance sampling with vMFN mixture model
%{
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
Matthias Willer
Fong-Lin Wu
Daniel Koutas

Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2022-04
* Inclusion of sensitivity analysis
---------------------------------------------------------------------------
Comments:
* Remove redundant dimension adjustment of limit state function. It should
  be restricted in the main script
---------------------------------------------------------------------------
Input:
* N                    : number of samples per level
* p                    : quantile value to select samples for parameter update
* g_fun                : limit state function
* distr                : Nataf distribution object or
                         marginal distribution object of the input variables
* k_init               : initial number of distributions in the mixture model
* sensitivity_analysis : implementation of sensitivity analysis: 1 - perform, 0 - not perform
* samples_return       : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr        : probability of failure
* lv        : total number of levels
* N_tot     : total number of samples
* gamma_hat : intermediate levels
* samplesU  : object with the samples in the standard normal space
* samplesX  : object with the samples in the original space
* k_fin     : final number of distributions in the mixture
* S_F1      : vector of first order Sobol' indices
---------------------------------------------------------------------------
Based on:
1."A new flexible mixture model for cross entropy based importance sampling".
   Papaioannou et al. (2018)
   In preparation.
2."Cross entropy-based importance sampling using Gaussian densities revisited"
   Geyer et al.
   To appear in Structural Safety
---------------------------------------------------------------------------
%}
if (N*p ~= fix(N*p)) || (1/p ~= fix(1/p))
    error('N*p and 1/p must be positive integers. Adjust N and p accordingly');
end

%% transform to the standard Gaussian space
if any(strcmp('Marginals',fieldnames(distr))) == 1   % use Nataf transform (dependence)
    dim = length(distr.Marginals);    % number of random variables (dimension)
    u2x = @(u) distr.U2X(u);          % from u to x
    
else   % use distribution information for the transformation (independence)
    % Here we are assuming that all the parameters have the same distribution !!!
    % Adjust accordingly otherwise
    dim = length(distr);                    % number of random variables (dimension)
    u2x = @(u) distr(1).icdf(normcdf(u));   % from u to x
end
if dim < 2
    error('The vMFN-mixture can only be applied to dim > 1');
end

%% LSF in standard space
G_LSF = @(u) g_fun(u2x(u));

%% Initialization of variables and storage
max_it = 50;      % maximum number of iterations
N_tot  = 0;       % total number of samples

% Definition of parameters of the random variables (uncorrelated standard normal)
gamma_hat = zeros(max_it+1,1);   % space for intermediate failure thresholds
samplesU  = cell(1,1);           % space for the samples in the standard space

%% CE procedure
% initial nakagami parameters (make it equal to chi distribution)
omega_init = dim;    % spread parameter
m_init     = dim/2;  % shape parameter

% initial von Mises-Fisher parameters
kappa_init = 0;                    % Concentration parameter (zero for uniform distribution)
mu_init    = hs_sample(1, dim,1);  % Initial mean sampled on unit hypersphere

% initial disribution weight
alpha_init = 1;

% Initializing parameters
mu_hat       = mu_init;
kappa_hat    = kappa_init;
omega_hat    = omega_init;
m_hat        = m_init;
gamma_hat(1) = 1;
alpha_hat    = alpha_init;

% Iteration
for j = 1:max_it
    % save parameters from previous step
    mu_cur    = mu_hat;
    kappa_cur = kappa_hat;
    omega_cur = omega_hat;
    m_cur     = m_hat;
    alpha_cur = alpha_hat;
    
    % Generate samples
    X = vMFNM_sample(mu_cur,kappa_cur,omega_cur,m_cur,alpha_cur,N);
        
    % Count generated samples
    N_tot = N_tot + N;
    
    % Evaluation of the limit state function
    geval = G_LSF(X);
    
    % Calculation of the likelihood ratio
    W_log = likelihood_ratio_log(X,mu_cur,kappa_cur,omega_cur,m_cur,alpha_cur);
        
    % Samples return - all / all by default
    if ~ismember(samples_return, [0 1])
        samplesU{j} = X;
    end

    % Check convergence
    if gamma_hat(j) == 0
        % Samples return - last
        if (samples_return == 1) || (samples_return == 0 && sensitivity_analysis == 1)
            samplesU{1} = X;
        end
        k_fin = length(alpha_cur);
        break;
    end
    
    % obtaining estimator gamma
    gamma_hat(j+1) = max(0, prctile(geval, p*100));
    fprintf('\nIntermediate threshold: %g\n',gamma_hat(j+1));
    
    % Indicator function
    I = (geval <= gamma_hat(j+1));
    
    % EM algorithm
    [mu, kappa, m, omega, alpha] = EMvMFNM(X(I,:)',exp(W_log(I,:)),k_init);
    
    % remove unnecessary components
    if min(alpha) <= 0.01
        ind   = find(alpha>0.01);
        mu    = mu(:,ind);
        kappa = kappa(ind);
        m     = m(ind);
        omega = omega(ind);
        alpha = alpha(ind);
    end
    
    % Assigning updated parameters
    mu_hat    = mu';
    kappa_hat = kappa;
    m_hat     = m;
    omega_hat = omega;
    alpha_hat = alpha;
   
end

% Samples return - all by default message
if ~ismember(samples_return, [0 1 2])
    fprintf('\n-Invalid input for samples return, all samples are returned by default \n');
end

% store the required steps
lv = j;
gamma_hat(lv+1:end) = [];

% adjust the dimension
% [mm,nn] = size(geval);
% if mm > nn
%     geval = geval';
% end

%% Calculation of the Probability of failure
I  = (geval <= gamma_hat(j));
Pr = 1/N*sum(exp(W_log(I,:)));

%% transform the samples to the physical/original space
samplesX = cell(length(samplesU),1);
if (samples_return ~= 0) || (samples_return == 0 && sensitivity_analysis == 1)
	for i = 1:length(samplesU)
		samplesX{i} = u2x(samplesU{i});
	end
end

%% sensitivity analysis
if sensitivity_analysis == 1
    % resample 1e4 failure samples with final weights W
    weight_id = randsample(find(I),1e4,'true',exp(W_log(I,:)));
    f_s = samplesX{end}(weight_id,:);
    
    if size(f_s,1) == 0
        fprintf("\n-Sensitivity analysis could not be performed, because no failure samples are available \n")
        S_F1 = [];
    else
        [S_F1, exitflag, errormsg] = Sim_Sobol_indices(f_s, Pr, distr);
        if exitflag == 1
            fprintf("\n-First order indices: \n");
            disp(S_F1);
        else
            fprintf('\n-Sensitivity analysis could not be performed, because: \n')
            fprintf(errormsg);
        end
    end
	if samples_return == 0
        samplesU = cell(1,1);  % empty return samples U
        samplesX = cell(1,1);  % and X
    end
else 
    S_F1 = [];
end

% Convergence is not achieved message
if j == max_it
    fprintf('-Exit with no convergence at max iterations \n\n');
end

return;


%===========================================================================
%===========================NESTED FUNCTIONS================================
%===========================================================================
function X = hs_sample(N,n,R)
% Returns uniformly distributed samples from the surface of an
% n-dimensional hypersphere
% N: # samples
% n: # dimensions
% R: radius of hypersphere

Y    = randn(n,N);
Y    = Y';
norm = repmat(sqrt(sum(Y.^2,2)),[1 n]);
X    = Y./norm*R;
return;


%===========================================================================
function X = vMFNM_sample(mu,kappa,omega,m,alpha,N)
% Returns samples from the von Mises-Fisher-Nakagami mixture

[k,dim] = size(mu);
if k == 1
    % sampling the radius
    %     pd=makedist('Nakagami','mu',m,'omega',omega);
    %     R=pd.random(N,1);
    R      = sqrt(gamrnd(m,omega./m,N,1));
    X_norm = vsamp(mu',kappa,N);    % sampling on unit hypersphere
    
else
    % Determine number of samples from each distribution
    %    z = sum(dummyvar(randsample(k,N,true,alpha)));
    %    k = length(z);
    ind = randsample(size(mu,1),N,true,alpha);
    z = histcounts(ind,[(1:size(mu,1)) size(mu,1)+1]);
    % Generation of samples
    R      = zeros(N,1);
    R_last = 0;
    X_norm = zeros(N,dim);
    X_last = 0;
    
    for i = 1:k
        % sampling the radius
        R(R_last+1:R_last+z(i)) = sqrt(gamrnd(m(i),omega(i)./m(i),z(i),1));
        R_last                  = R_last + z(i);
        
        % sampling on unit hypersphere
        X_norm(X_last+1:X_last+z(i),:) = vsamp(mu(i,:)',kappa(i),z(i));
        X_last                         = X_last+z(i);
        clear pd;
    end
end

% assign sample vector
X = bsxfun(@times,R,X_norm);
return;


%===========================================================================
function X = vsamp(center, kappa, n)
% Returns samples from the von Mises-Fisher distribution

% only d > 1
d  = size(center,1);			% Dimensionality
l  = kappa;				      % shorthand
t1 = sqrt(4*l*l + (d-1)*(d-1));
b  = (-2*l + t1 )/(d-1);
x0 = (1-b)/(1+b);
X  = zeros(n,d);
m  = (d-1)/2;
c  = l*x0 + (d-1)*log(1-x0*x0);
%
for i = 1:n
    t = -1000;
    u = 1;
    while (t < log(u))
        z = betarnd(m , m);	   % z is a beta rand var
        u = rand;			    	% u is unif rand var
        w = (1 - (1+b)*z)/(1 - (1-b)*z);
        t = l*w + (d-1)*log(1-x0*w) - c;
    end
    v          = hs_sample(1,d-1,1);
    X(i,1:d-1) = sqrt(1-w*w)*v';
    X(i,d)     = w;
end
%
[v,b] = house(center);
Q     = eye(d) - b*(v*v');
for i = 1:n
    tmpv   = Q*X(i,:)';
    X(i,:) = tmpv';
end
return;


%===========================================================================
function y = vMF_logpdf(X,mu,kappa)
% Returns the von Mises-Fisher mixture log pdf on the unit hypersphere

d = size(X,1);
n = size(X,2);
if kappa == 0
    A = log(d) + log(pi^(d/2)) - gammaln(d/2+1);
    y = -A*ones(1,n);
elseif kappa > 0
    c = (d/2-1)*log(kappa)-(d/2)*log(2*pi)-logbesseli(d/2-1,kappa);
    q = bsxfun(@times,mu,kappa)'*X;
    y = bsxfun(@plus,q,c');
else
    error('kappa must not be negative');
end
return;


%===========================================================================
function y = nakagami_logpdf(X,m,om)

y = log(2)+m*(log(m)-log(om)-X.^2./om)+log(X).*(2*m-1)-gammaln(m);
return;


%===========================================================================
function W_log = likelihood_ratio_log(X,mu,kappa,omega,m,alpha)

k       = length(alpha);
[N,dim] = size(X);
R       = sqrt(sum(X.^2,2));
if k == 1
    % log pdf of vMF distribution
    logpdf_vMF = vMF_logpdf((bsxfun(@times,X,1./R))',mu',kappa)';
    % log pdf of Nakagami distribution
    logpdf_N = nakagami_logpdf(R,m,omega);
    % log pdf of weighted combined distribution
    h_log = logpdf_vMF+logpdf_N;
else
    logpdf_vMF = zeros(N,k);
    logpdf_N   = zeros(N,k);
    h_log      = zeros(N,k);
    
    % log pdf of distributions in the mixture
    for i = 1:k
        % log pdf of vMF distribution
        logpdf_vMF(:,i) = vMF_logpdf((bsxfun(@times,X,1./R))',mu(i,:)',kappa(i))';
        % log pdf of Nakagami distribution
        logpdf_N(:,i) = nakagami_logpdf(R,m(i),omega(i));
        % log pdf of weighted combined distribution
        h_log(:,i) = logpdf_vMF(:,i)+logpdf_N(:,i)+log(alpha(i));
    end
    
    % mixture log pdf
    h_log = logsumexp(h_log,2);
end

% unit hypersphere uniform log pdf
A   = log(dim)+log(pi^(dim/2))-gammaln(dim/2+1);
f_u = -A;

% chi log pdf
f_chi = log(2)*(1-dim/2)+log(R)*(dim-1)-0.5*R.^2-gammaln(dim/2);

% logpdf of the standard distribution (uniform combined with chi distribution)
f_log = f_u + f_chi;
W_log = f_log - h_log;
return;


%===========================================================================
function [v,b] = house(x)
% HOUSE Returns the householder transf to reduce x to b*e_n
% [V,B] = HOUSE(X)  Returns vector v and multiplier b so that
% H = eye(n)-b*v*v' is the householder matrix that will transform
% Hx ==> [0 0 0 ... ||x||], where  is a constant.

n = length(x);
s = x(1:n-1)'*x(1:n-1);
v = [x(1:n-1)', 1]';
if (s == 0)
    b = 0;
else
    m = sqrt(x(n)*x(n) + s);
    if (x(n) <= 0)
        v(n) = x(n)-m;
    else
        v(n) = -s/(x(n)+m);
    end
    b = 2*v(n)*v(n)/(s + v(n)*v(n));
    v = v/v(n);
end
return;


%===========================================================================
function logb = logbesseli(nu,x)
% log of the Bessel function, extended for large nu and x approximation
% from Eq. 9.7.7 of Abramowitz and Stegun
% http://www.math.sfu.ca/~cbm/aands/page_378.htm

if nu == 0   % special case when nu=0
    logb = log(besseli(nu,x));
else   % normal case
    n      = size(x,1);
    frac   = x./nu;
    square = ones(n,1) + frac.^2;
    root   = sqrt(square);
    eta    = root + log(frac) - log(ones(n,1)+root);
    logb   = - log(sqrt(2*pi*nu)) + nu.*eta - 0.25*log(square);
end
return;


%===========================================================================
function s = logsumexp(x, dim)
% Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
% By default dim = 1 (columns).
% Written by Michael Chen (sth4nth@gmail.com).

if nargin == 1
    % Determine which dimension sum will use
    dim = find(size(x)~=1,1);
    if isempty(dim)
        dim = 1;
    end
end

% subtract the largest in each column
y = max(x,[],dim);
x = bsxfun(@minus,x,y);
s = y + log(sum(exp(x),dim));
i = find(~isfinite(y));
if ~isempty(i)
    s(i) = y(i);
end
return;
%%END