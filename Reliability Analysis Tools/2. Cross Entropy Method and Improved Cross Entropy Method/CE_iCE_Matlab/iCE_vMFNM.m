function [Pr, lv, N_tot, samplesU, samplesX, k_fin,W_final, f_s_iid] = ...
    iCE_vMFNM(N, g_fun, distr, max_it, CV_target, k_init, samples_return)
%% Improved cross entropy-based importance sampling with vMFN mixture model
%{
---------------------------------------------------------------------------
Improved cross entropy-based importance sampling with vMFN mixture model
---------------------------------------------------------------------------
Created by:
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Fong-Lin Wu
Matthias Willer
Peter Kaplan
Daniel Koutas
Ivan Olarte-Rodriguez

Engineering Risk Analysis Group
Technische Universitaet Muenchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2023-08
* Generation of i.i.d. samples for Sensitivity Analysis
---------------------------------------------------------------------------
Comments:
* Adopt draft scripts from Sebastian and reconstruct the code to comply
  with the style of the published codes
* W is the original importance weight (likelihood ratio)
* W_t is the transitional importance weight of samples
* W_approx is the weight (ratio) between real and approximated indicator functions
---------------------------------------------------------------------------
Input:
* N                    : number of samples per level
* g_fun                : limit state function
* max_it               : maximum number of iterations
* distr                : Nataf distribution object or marginal distribution object of the input variables
* CV_target            : taeget correlation of variation of weights
* k_init               : initial number of Gaussians in the mixture model
* samples_return       : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr        : probability of failure
* lv        : total number of levels
* N_tot     : total number of samples
* samplesU  : object with the samples in the standard normal space
* samplesX  : object with the samples in the original space
* W_final   : Final Weights
* f_s       : i.i.d failure samples
---------------------------------------------------------------------------
Based on:
1. Papaioannou, I., Geyer, S., & Straub, D. (2019).
   Improved cross entropy-based importance sampling with a flexible mixture model.
   Reliability Engineering & System Safety, 191, 106564
2. Geyer, S., Papaioannou, I., & Straub, D. (2019).
   Cross entropy-based importance sampling using Gaussian densities revisited. 
   Structural Safety, 76, 15–27
---------------------------------------------------------------------------
%}

%% Transform to the standard Gaussian space
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
G_LSF = @(u) g_fun(reshape(u2x(u),[],dim));

%% Initialization of variables and storage
N_tot  = 0;        % total number of samples

% initial nakagami parameters (make it equal to chi distribution)
omega_init = dim;    % spread parameter
m_init     = dim/2;  % shape parameter;

% initial von Mises-Fisher parameters
kappa_init = 0;                    % Concentration parameter (zero for uniform distribution)
mu_init    = hs_sample(1, dim,1);  % Initial mean sampled on unit hypersphere

Pi_init   = 1;                   % weight of the mixtures
sigma_t   = zeros(max_it,1);     % squared difference between COV and target COV
samplesU  = cell(1,1);           % space for the samples in the standard space

%% iCE procedure
% initial disribution weight and numbers
Pi_hat       = Pi_init;
k            = k_init;

% Initializing parameters
mu_hat       = mu_init;
kappa_hat    = kappa_init;
omega_hat    = omega_init;
m_hat        = m_init;

% Iteration
for j = 1:max_it
    % save parameters from previous step
    mu_cur    = mu_hat;
    kappa_cur = kappa_hat;
    omega_cur = omega_hat;
    m_cur     = m_hat;
    
    % generate samples
    X = vMFNM_sample(mu_cur,kappa_cur,omega_cur,m_cur,Pi_hat,N);
    
    % count generated samples
    N_tot = N_tot + N;
    
    % evaluation of the limit state function
    geval = zeros(size(X,1),1);
    for ii = 1:numel(geval)
        geval(ii) = G_LSF(X(ii,:));
    end
    
    % initialize sigma_0
    if j==1,    sigma_t(1) = 10*mean(geval);    end
    
    % calculation of the likelihood ratio
    W_log = likelihood_ratio_log(X,mu_cur,kappa_cur,omega_cur,m_cur,Pi_hat);
    W     = exp(W_log);
    
    % indicator function
    I = (geval <= 0);
    
    % Samples return - all / all by default
    if ~ismember(samples_return, [0 1])
        samplesU{j} = X;
    end

    % check convergence
    % transitional weight W_t = I*W when sigma_t approches 0 (smooth approximation:)
    W_approx = I ./ approx_normCDF(-geval/sigma_t(j)); % weight of indicator approximations
    %Cov_x   = std(I.*W_log) / mean(I.*W_log);
    Cov_x    = std(W_approx) / mean(W_approx);
    if Cov_x <= CV_target
        % Samples return - last
        if (samples_return == 1) 
            samplesU{1} = X;
        end
        break;
    end
    
    % compute sigma and weights for distribution fitting
    % minimize COV of W_t (W_t=normalCDF*W)
    fmin      = @(x) abs(std(approx_normCDF(-geval/x) .* W) / mean(approx_normCDF(-geval/x).*W) - CV_target);
    sigma_new = fminbnd(fmin,0,sigma_t(j));
    
    % update parameters
    sigma_t(j+1) = sigma_new;
    W_t = approx_normCDF(-geval/sigma_t(j+1)).*W;
    
    % normalize weights
    W_t = W_t./sum(W_t);
    
    % parameter update with EM algorithm
    % improved IS: Use all samples and
    [mu, kappa, m, omega, Pi] = EMvMFNM(X', W_t, k);
        
    % remove unnecessary components
    if min(Pi) <= 0.01
        ind   = find(Pi>0.01);
        mu    = mu(:,ind);
        kappa = kappa(ind);
        m     = m(ind);
        omega = omega(ind);
        Pi = Pi(ind);
    end
    
    % Assigning updated parameters
    mu_hat    = mu';
    kappa_hat = kappa;
    m_hat     = m;
    omega_hat = omega;
    Pi_hat    = Pi;
    k         = length(Pi);
    
end

% Store the weights
W_final = W;

% Samples return - all by default message
if ~ismember(samples_return, [0 1 2])
    fprintf('\n-Invalid input for samples return, all samples are returned by default \n');
end

% store the required steps
lv=j;
k_fin = k;

%% Calculation of the Probability of failure
Pr      = 1/N*sum(W(I));

%% transform the samples to the physical/original space
samplesX = cell(length(samplesU),1);
f_s_iid = [];
if (samples_return ~= 0)
	for m = 1:length(samplesU)
        if ~isempty(samplesU{m})
            samplesX{m} = u2x(samplesU{m});
        end
    end

    %% Output for Sensitivity Analysis

    % resample 1e4 failure samples with final weights W
    weight_id = randsample(find(I),1e4,'true',W(I));
    if ~isempty(samplesX{end})
        f_s_iid = samplesX{end}(weight_id,:);
    end
end


%% Error Messages
% Convergence is not achieved message
if j == max_it
    fprintf('-Exit with no convergence at max iterations \n\n');
end

return;

%===========================================================================
%===========================NESTED FUNCTIONS================================
%===========================================================================
function p = approx_normCDF(x)

% Returns an approximation for the standard normal CDF based on a
% polynomial fit of degree 9

erfun=zeros(size(x));

idpos=x>0;
idneg=x<0;

t=(1+0.5*abs(x/sqrt(2))).^-1;

tau=t.*exp(-(x/sqrt(2)).^2-1.26551223+1.0000236*t+0.37409196*t.^2+0.09678418*t.^3-0.18628806*t.^4+0.27886807*t.^5-1.13520398*t.^6+1.48851587*t.^7-0.82215223*t.^8+0.17087277*t.^9);
erfun(idpos)=1-tau(idpos);
erfun(idneg)=tau(idneg)-1;

p=0.5*(1+erfun);

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
function X = vMFNM_sample(mu,kappa,omega,m,Pi,N)
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
    %    z = sum(dummyvar(randsample(k,N,true,Pi)));
    %    k = length(z);
    ind = randsample(size(mu,1),N,true,Pi);
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
function W_log = likelihood_ratio_log(X,mu,kappa,omega,m,Pi)

k       = length(Pi);
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
        h_log(:,i) = logpdf_vMF(:,i)+logpdf_N(:,i)+log(Pi(i));
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