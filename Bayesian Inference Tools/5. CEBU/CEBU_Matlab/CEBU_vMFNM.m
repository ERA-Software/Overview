function [samplesU, samplesX, v_tot, beta_tot, k_fin, evidence, ...
          Wlast_normed, f_s_iid] = CEBU_vMFNM(N, g_fun, distr, ...
          max_it, CV_target, k_init, samples_return, varargin)
%% Comments
%{
---------------------------------------------------------------------------
Bayesian Updating and Marginal Likelihood Estimation by Cross Entropy based 
Importance Sampling
---------------------------------------------------------------------------

Created by:
Michael Engel
Oindrila Kanjilal 
Iason Papaioannou
Daniel Straub

Assistant Developers:
Daniel Koutas
Ivan Olarte-Rodriguez

Engineering Risk Analysis Group
Technische Universitaet Muenchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2023-09
* 
---------------------------------------------------------------------------
Comments:
* Adopted draft scripts from Michael Engel. Reconstruction of the code made 
  to comply with the style of the published codes

---------------------------------------------------------------------------
Input:
* N                    : number of samples per level
* g_fun                : limit state function
* max_it               : maximum number of iterations
* distr                : Nataf distribution object or marginal distribution object of the input variables
* CV_target            : taeget correlation of variation of weights
* k_init               : initial number of Gaussians in the mixture model
* samples_return       : return of samples: 0 - none, 1 - final sample, 2 - all samples

Optional:
* N_last               : set a number of independent identically 
                         distributed of samples to generate at the end.
                         (Default value: 1e+04).
* sampling_method      : method used to sample the independent identically
                         distributed samples. 
---------------------------------------------------------------------------
Output:
* samplesU      : list of the samples in the standard normal space
* samplesX      : list of the samples in the original space
* v_tot         : list of distribution parameters
* beta_tot      : list of tempering parameters
* k_fin         : final number of Gaussians in the mixture
* evidence      : evidence
* W_last_normed : normed weights of the last level samples
* f_s_iid       : iid samples
---------------------------------------------------------------------------
Based on:
1. Engel, M., Kanjilal, O., Papaioannou, I., Straub, D. (2022)
   Bayesian Updating and Marginal Likelihood Estimation by 
   Cross Entropy based Importance Sampling.(Preprint submitted to Journal 
   of Computational Physics) 
---------------------------------------------------------------------------
%}

%% Input assurance

% Initialize input parser
i_parser = inputParser;

% Insert conditions on required inputs
addRequired(i_parser,"N",@(x) isscalar(x) && x>0);

addRequired(i_parser,"g_fun", @(x) strcmp(class(x),"function_handle"));

addRequired(i_parser,"distr",@(x) (strcmp(class(x),"ERANataf")) || ...
    strcmp(class(x),"ERARosen") || strcmp(class(x),"ERADist" ));

addRequired(i_parser,"max_it", @(x) isscalar(x) && x>0);

addRequired(i_parser,"CV_target", @(x) isscalar(x) && isfinite(x));

addRequired(i_parser,"k_init", @(x) isscalar(x) && x>0);

addRequired(i_parser,"samples_return", @(x) isscalar(x) && ismember(x,[0,1,2]));

addOptional(i_parser,"N_last", 10000,@(x) isscalar(x) && x>0);

addOptional(i_parser,"sampling_method","stratified", @(x) any(validatestring(x,{'stratified', ...
    'multinomial','residual','systematic','matlab'})));

% Parse Inputs (error is thrown if the validation functions are not
% fulfilled
parse(i_parser,N,g_fun,distr,max_it, CV_target, k_init, samples_return,varargin{:});


%% Set transformation into Standard Gaussian Space
if any(strcmp('Marginals',fieldnames(distr)))    % use Nataf transform (dependence)
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
G_LSF = @(u) g_fun(u2x(u)');

%% Initialization of variables and storage

N_tot  = 0;        % total number of samples

mu_init = 1/sqrt(dim).*ones(k_init,dim); % Initial mean sampled on unit hypersphere
kappa_init = zeros(k_init,1);  % Concentration parameter (zeroes for uniform distribution)
omega_init = ones(k_init,1)*(dim); % spread parameter
m_init = ones(k_init,1).*(dim/2); % shape parameter
alpha_init = ones(k_init,1)./k_init; % normalized weight of the mixtures
beta_init = 0;
                   
%sigma_t_init   = zeros(max_it,1);     % squared difference between COV and target COV
samplesU  = cell(max_it,1);           % space for the samples in the standard space

%% initialize storage parameters

mu_hat = mu_init;  % Initial mean sampled on unit hypersphere
kappa_hat = kappa_init;  % Concentration parameter
omega_hat = omega_init;  % spread parameter
m_hat = m_init; % shape parameter
alpha_hat = alpha_init; % normalized weight of the mixtures

% Generate structure array to store the parameters
v_tot = struct("mu_hat",cell(max_it,1),"kappa_hat",cell(max_it,1),"omega_hat",cell(max_it,1),...
               "m_hat",cell(max_it,1),"alpha_hat",cell(max_it,1));

v_tot(1) =  struct("mu_hat",mu_hat,"kappa_hat",kappa_hat,"omega_hat",omega_hat,...
               "m_hat",m_hat,"alpha_hat",alpha_hat);

beta_hat = beta_init;

beta_tot = zeros(max_it,1);
cv_tot = zeros(max_it,1);


%% Iteration Section
% Start the counter

for j = 1:max_it 
    fprintf(['\n\nstep ',num2str(j),' for beta=',num2str(beta_hat)])
    
    % Generate samples
    X = vMFNM_sample(v_tot(j).mu_hat,...
                     v_tot(j).kappa_hat,...
                     v_tot(j).omega_hat,...
                     v_tot(j).m_hat,...
                     v_tot(j).alpha_hat,N);

    samplesU{j} = X;

    % Add the number of samples generated
    N_tot = N_tot+N;
    
    % evaluation of the limit state function
    geval = G_LSF(X);

    % Replace the "NaN" values from the estimation of the limit state
    % function
    geval(isnan(geval)) = min(geval,[],"all");  

    % Compute the prior pdf function
    logPriorpdf_ = jointpdf_prior(X,dim);
    
    % Update beta (tempering parameter)
    beta_hat = new_beta(beta_hat,X,v_tot(j),geval,logPriorpdf_,N,CV_target);
    
    % Handler (if beta is greater than 1, set to 1)
    if beta_hat > 1
        beta_hat = 1;
    end

    % Save the new beta (tempering parameter)
    beta_tot(j+1) = beta_hat;
    
    % likelihood ratio
    logWconst_ = log_W_const(X,v_tot(j),logPriorpdf_);
    logW = logwl(beta_hat,geval',logWconst_); 
    
    cv_tot(j) = std(exp(logW))/mean(exp(logW)); % only for diagnostics, hence not optimized for precision
    
    % update parameter
    v_new = vnew(X,logW,k_init);
    v_tot(j+1) = v_new;
    
    if beta_hat>=1
        fprintf('\nbeta is equal to 1')

        break;
    end
        
end

%% Adjust storing variables
v_tot = v_tot(1:j+1);

samplesU = samplesU(1:j,1);

beta_tot = beta_tot(1:j+1);
cv_tot = cv_tot(1:j,1);

%% results
fprintf('\n\nstart final sampling')

% transform the samples to the physical/original space
samplesX = cell(length(samplesU),1);
if samples_return ~= 0
	for m = 1:numel(samplesU)
		samplesX{m} = u2x(samplesU{m})';
	end
end

% Set the sample output based on the return parameter value
if i_parser.Results.samples_return == 0
    samplesU = {};
    samplesX = {};
elseif i_parser.Results.samples_return == 1
    samplesU = samplesU(end);
    samplesX = samplesX(end);
end

% Produce new failure samples after end step
ulast = vMFNM_sample(v_tot(end).mu_hat,v_tot(end).kappa_hat,...
    v_tot(end).omega_hat,v_tot(end).m_hat,v_tot(end).alpha_hat,...
    i_parser.Results.N_last);

N_tot = N_tot+i_parser.Results.N_last;

logWlast = logwl2(ulast,v_tot(end),G_LSF,1);
Wlast_normed = exp(logWlast-logsumexp(logWlast,2));

% Compute nESS
nESS = exp(2*logsumexp(logWlast)-logsumexp(2*logWlast)-log(i_parser.Results.N_last));

evidence = exp(logsumexp(logWlast)-log(i_parser.Results.N_last)); % Evidence
 
xlast = u2x(ulast);

k_fin = size(v_tot(end).mu_hat,1);

%% Generate Random Samples
f_s_iid = MErandsample(xlast,i_parser.Results.N_last,true,Wlast_normed,...
                       i_parser.Results.sampling_method);
fprintf(['\nfinished after ',num2str(j-1),' steps at beta=',num2str(beta_hat),'\n'])


end


%% NESTED FUNCTIONS

%===========================================================================
function X = vMFNM_sample(mu,kappa,omega,m,alpha,N)
% Returns samples from the von Mises-Fisher-Nakagami mixture
[k,dim] = size(mu);
if k == 1
   % sampling the radius
   R = sqrt(gamrnd(m,omega./m,N,1));
   % sampling direction on unit hypersphere
   X_norm = vsamp(mu',kappa,N);    
else   
   % Determine number of samples from each distribution
   z = sum(dummyvar(randsample(k,N,true,alpha)));
   k = length(z);
   
   % Generation of samples
   R = zeros(N,1);
   R_last = 0;
   X_norm = zeros(N,dim);
   X_last = 0;
   for i = 1:k      
      % sampling the radius
      R(R_last+1:R_last+z(i)) = sqrt(gamrnd(m(i),omega(i)./m(i),z(i),1));
      R_last = R_last + z(i);
      % sampling direction on unit hypersphere
      X_norm(X_last+1:X_last+z(i),:) = vsamp(mu(i,:)',kappa(i),z(i));
      X_last = X_last+z(i);
      clear pd;
   end
end
% assign sample vector
X = bsxfun(@times,R,X_norm);
end

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

end

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

end

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

end

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
end

%===========================================================================
function h = mixture_log_pdf(X,mu,kappa,omega,m,alpha)
    
    [N,dim] = size(X);
    if dim > N
        X=X';
        [N,~] = size(X);
    end

    k = length(alpha);

    % Convert from standard normal space to radial space
    R      = sqrt(sum(X'.^2,1));
    X_norm = X'./R;

    h = zeros(k,N);
    if k == 1   
       % log pdf of vMF distribution
       logpdf_vMF = vMF_logpdf(X_norm,mu(1,:),kappa(1))';
       % log pdf of Nakagami distribution
       logpdf_N = nakagami_logpdf(R',m(1),omega(1));
       % log pdf of weighted combined distribution
       h(1,:) = logpdf_vMF+logpdf_N;
    else   
       logpdf_vMF = zeros(N,k);
       logpdf_N   = zeros(N,k);
       h_log      = zeros(k,N);

       % log pdf of distributions in the mixture
       for i = 1:k
          % log pdf of vMF distribution
          logpdf_vMF(:,i) = vMF_logpdf(X_norm,mu(i,:),kappa(i))';
          % log pdf of Nakagami distribution
          logpdf_N(:,i) = transpose(nakagami_logpdf(R,m(i),omega(i)));
          % log pdf of weighted combined distribution
          h_log(i,:) = logpdf_vMF(:,i)+logpdf_N(:,i)+log(alpha(i));
       end

       % mixture log pdf
       h = logsumexp(h_log,1);
    end
end

% =========================================================================
function y = vMF_logpdf(X,mu,kappa)
    % Returns the von Mises-Fisher mixture log pdf on the unit hypersphere
    d = size(X,1);
    n = size(X,2);
    if kappa == 0
        % unit hypersphere
        A = log(d) + log(pi^(d/2)) - gammaln(d/2+1);
        y = -A*ones(1,n);
    elseif kappa > 0
        % concentrated direction
        c = (d/2-1)*log(kappa)-(d/2)*log(2*pi)-logbesseli(d/2-1,kappa);
        q = bsxfun(@times,mu,kappa)*X;
        y = bsxfun(@plus,q,c');
    else
        error('kappa<0 or NaN');
    end
end

%==========================================================================
function y = nakagami_logpdf(X,m,om)

y = log(2)+m*(log(m)-log(om)-X.^2./om)+log(X).*(2*m-1)-gammaln(m);
end

%==========================================================================
function logb = logbesseli(nu,x)
% log of the Bessel function, extended for large nu and x
% approximation from Eqn 9.7.7 of Abramowitz and Stegun
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


end


%===========================================================================
function pdf = jointpdf_prior(X,dim)
% Computes the joint pdf of the prior distribution

% Convert the physical space from "rectilinear" to "spherical"
R      = sqrt(sum(X'.^2,1));
X_norm = X'./R;

% numerical stability; 
% should actually be equal to number closest to zero without cancellation 
% error
R(R==0)=1e-300; 

% uniform hypersphere
A   = log(dim)+dim/2*log(pi)-gammaln(dim/2+1);
f_u = -A;

% chi distribution
f_chi = log(2)*(1-dim/2)+log(R)*(dim-1)-0.5*R.^2-gammaln(dim/2);
pdf = f_u + f_chi;

end

%===========================================================================

function newbeta = new_beta(betaold,samples_U,v,logLikelihood_,logpriorpdf_,Nstep,cvtarget)
% Function which performs the update of the new parameter beta  

% Compute the logarithmic weights
logWconst_ = log_W_const(samples_U,v,logpriorpdf_); 

% necessary as new set of v given as computation after parameter update
logstep_ = logstep(Nstep,cvtarget); % could be computed beforehand (once) but computational cost is negligible

% variant a: optimize beta with respect to the ISdensity -> depending on parameters
% logESSoptimize_ = @(dbeta) logESSoptimize(betaold+abs(dbeta),logLikelihood_,logWconst_,logstep_);
% variant b: optimize beta with respect to the likelihood ratio assuming a 
% perfect fit of the previous ISdensity -> not depending on the parameters
logESSoptimize_ = @(dbeta) logESSoptimize2(abs(dbeta),logLikelihood_,logstep_);

options = optimset('TolFun',1e-100,'TolX',1e-100,'MaxFunEvals',1000,'MaxIter',1000);

%,'Display','iter');%,'PlotFcns',@optimplotfval);
[dbeta,~] = fminbnd(logESSoptimize_,-1,1e-6,options);
dbeta = abs(dbeta);
newbeta = betaold+dbeta;

% for convergence
if newbeta >=1-1e-3
    newbeta=1;
end


end



%===========================================================================
function newv = vnew(X,logWold,k)
% This function updates the parameters of the parametric density


% RESHAPE STEP (avoid confusion on row or column vectors)
[N,dim] = size(X);
if N > dim
    X = X';
    dim = size(X,1);
end

% END RESHAPE STEP (avoid confusion on row or column vectors)
if k == 1
    
    % Transform coordinates
    R      = sqrt(sum(X.^2,1));
    X_norm = X./R;

    logsumwold = logsumexp(logWold,2);
    
    % mean direction mu
    signX = sign(X);
    logX = log(abs(X));
    
    [mu_unnormed,signmu] = signedlogsumexp(logX+logWold,2,signX);
    lognorm_mu = 0.5*logsumexp(2*mu_unnormed,1);
    mu = exp(mu_unnormed-lognorm_mu).*signmu;
    mu(isinf(mu)) = sqrt(dim);
    
    v0 = mu;
        
    % concentration parameter kappa
    xi = min(exp(lognorm_mu-logsumwold),0.999999999);
    kappa = abs((xi*dim-xi^3)/(1-xi^2)); 
        
    v1 = kappa;

    % spread parameter omega
    % for numerical stability; 
    % should actually be equal to the number closest to zero without 
    % cancellation error by machine
    R(R==0)=1e-300; 
    
    logR = log(R);
    logRsquare = 2*logR;
    omega = exp(logsumexp(logWold+logRsquare,2) - logsumwold);
    
    v2 = omega;
    
    % shape parameter m
    logRpower = 4*logR;
    mu4       = exp(logsumexp(logWold+logRpower,2) - logsumwold);
    m         = omega.^2./(mu4-omega.^2);
    
    m(m<0.5)  = dim/2;
    m(isinf(m)) = dim/2;
    
    v3 = m;
    
    % distribution weights alpha
    alpha = [1];
    
    v4 = alpha;
    
    newv_var = {transpose(v0),v1,v2,v3,v4};
else
%             % EMvMFNM from ERA
    fprintf('\nusing EMvMFNM_log ')
    % for sake of consistency I let m and omega computed in the same order 
    % as within the EMvMFNM-script -> therefore the transposed digits
    [v0,v1,v3,v2,v4] = EMvMFNM_log(X,logWold,k); 

    % Perform Flips
    newv_var = {transpose(v0),v1,v2,v3,v4};
end

% Restructure the output as a dynamic memory variable (STRUCTURE)
newv = struct("mu_hat",newv_var{1},"kappa_hat",newv_var{2},"omega_hat",newv_var{3},...
               "m_hat",newv_var{4},"alpha_hat",newv_var{5});


end


% =========================================================================

function [s,sign] = signedlogsumexp(x, dim, b)
    % Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
    % By default dim = 1 (columns).
    % Written by Michael Chen (sth4nth@gmail.com).
    
    % Adapted by Michael Engel such that log(sum(exp(x)*b,dim)) and
    % negative results are supported.
    % Only recommended for b working as a sign-vector of exp(x).

    if nargin == 1
       % Determine which dimension sum will use
       dim = find(size(x)~=1,1);
       if isempty(dim)
          dim = 1;
       end
    end
    
    if isempty(b)
        b = ones(1,size(x,dim));
    end

    % subtract the largest in each column (rescaling x to (0,1] 
    % where log offers better precision)
    y = max(x,[],dim);
    x = bsxfun(@minus,x,y);
    term = sum(exp(x).*b,dim);
    sign = ones(size(term));
    sign(term<0) = -1;

    % return nonfinite value if existing
    s = y + log(abs(term));
    i = find(~isfinite(y));
    if ~isempty(i)
       s(i) = y(i);
    end
end


% =========================================================================

function logw = logESSoptimize(beta,logLikelihood,logwconst,logstep)
    logw = beta.*logLikelihood+logwconst;
    logw = ((2*logsumexp(logw)-logsumexp(2*logw))-(logstep))^2;
end


% =========================================================================
function logdiff = logESSoptimize2(dbeta,logLikelihood,logstep)
    logw = dbeta.*logLikelihood;
    logdiff = ((2*logsumexp(logw)-logsumexp(2*logw))-(logstep))^2;
end

% =========================================================================
function logconstw = log_W_const(samples_U,v,logpriorpdf)
    logconstw = logpriorpdf-mixture_log_pdf(samples_U,v.mu_hat,...
        v.kappa_hat, v.omega_hat,v.m_hat,v.alpha_hat);
end

% =========================================================================
function logS = logstep(Nstep,cvtarget)
    logS = log(Nstep)-log(1+cvtarget^2);
end

% =========================================================================
function logW = logwl(beta,logL,logWconst)
    % Compute the logarithm of the transition weights (W_t)
    if size(logL,2) == 1
        logL = logL';
    end

    if size(logWconst,2) ==1
        logWconst = logWconst';
    end
    logW = beta.*logL+logWconst;
end

% =========================================================================
function logw = logwl2(X,v,G_LSF_U,beta)

% Perform evaluation of the 
logL_ = G_LSF_U(X);

% Correct the evaluation
logL_(isnan(logL_)) = min(logL_,[],"all");  

%fprintf(['\nmaximum of final logLikelihood: ',num2str((max(logL_)))]);
%fprintf(['\nminimum of final logLikelihood: ',num2str((min(logL_)))]);

if size(logL_,2) == 1
        logL_ = logL_';
end

[~,dim] = size(X);
logP_ = jointpdf_prior(X,dim);
logwconst_ = log_W_const(X,v,logP_);

if size(logwconst_,2) ==1
    logwconst_ = logwconst_';
end

logw = beta*logL_+logwconst_;

end


%% NESTED SAMPLING METHODS
function randsamples = MErandsample(u,n,replace,W,method)
%{
This function is a workaround so that the MATLAB and python versions are
consistent. See MATLAB-function randsample for documentation.
%}

% If the number of arguments is less than 5, then proceed to set the
% 'matlab' method by default
    if nargin<5
        method = 'matlab';
    end

    if strcmp(method,'stratified') % in fact the same as systematic (due to implementation -> should not be the case)
        randsamples = u(:,resampleStratified(W));
    elseif strcmp(method,'multinomial')
        randsamples = u(:,resampleMultinomial(W));
    elseif strcmp(method,'residual')
        randsamples = u(:,resampleResidual(W));
    elseif strcmp(method,'systematic')
        randsamples = u(:,resampleSystematic(W));
    elseif strcmp(method,'matlab')
        randsamples = u(:,randsample(1:length(W),n,replace,W)); % multinomial set
    else
        randsamples = nan;
    end
end

% ------------------------------------------------------------------------
% Stratified Sampling
% ------------------------------------------------------------------------
function [ indx ] = resampleStratified( w )
    N = length(w);
    Q = cumsum(w);
    for i=1:N
        T(i) = rand(1,1)/N + (i-1)/N;
    end
    T(N+1) = 1;
    i=1;
    j=1;
    while (i<=N)
        if (T(i)<Q(j))
            indx(i)=j;
            i=i+1;
        else
            j=j+1;        
        end
    end
end
% ------------------------------------------------------------------------
% Multinomial Sampling
% ------------------------------------------------------------------------
function [ indx ] = resampleMultinomial( w )
M = length(w);
Q = cumsum(w);
Q(M)=1; % Just in case...
i=1;
while (i<=M)
    sampl = rand(1,1);  % (0,1]
    j=1;
    while (Q(j)<sampl)
        j=j+1;
    end
    indx(i)=j;
    i=i+1;
end
end

% ------------------------------------------------------------------------
% Residual Sampling
% ------------------------------------------------------------------------
function [ indx ] = resampleResidual( w )
M = length(w);
% "Repetition counts" (plus the random part, later on):
Ns = floor(M .* w);
% The "remainder" or "residual" count:
R = sum( Ns );
% The number of particles which will be drawn stocastically:
M_rdn = M-R;
% The modified weights:
Ws = (M .* w - floor(M .* w))/M_rdn;
% Draw the deterministic part:
i=1;
for j=1:M
    for k=1:Ns(j)
        indx(i)=j;
        i = i +1;
    end
end

% And now draw the stocastic (Multinomial) part:
Q = cumsum(Ws);
Q(M)=1; % Just in case...
while (i<=M)
    sampl = rand(1,1);  % (0,1]
    j=1;
    while (Q(j)<sampl)
        j=j+1;
    end
    indx(i)=j;
    i=i+1;
end

end

% ------------------------------------------------------------------------
% Systematic Sampling
% ------------------------------------------------------------------------
function [ indx ] = resampleSystematic( w )
N = length(w);
Q = cumsum(w);
T = linspace(0,1-1/N,N) + rand(1)/N;
T(N+1) = 1;
i=1;
j=1;
while (i<=N)
    if (T(i)<Q(j))
        indx(i)=j;
        i=i+1;
    else
        j=j+1;        
    end
end

end