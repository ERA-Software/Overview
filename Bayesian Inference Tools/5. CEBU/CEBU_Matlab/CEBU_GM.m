function [samplesU, samplesX, v_tot, beta_tot, k_fin, evidence, ...
          Wlast_normed, f_s_iid] = CEBU_GM(N, g_fun, distr, ...
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

%if dim < 2
%    error('The vMFN-mixture can only be applied to dim > 1');
%end

%% LSF in standard space
G_LSF = @(u) g_fun(u2x(u)');

%% Initialization of variables and storage

N_tot  = 0;        % total number of samples

mu_init = zeros(k_init,dim); % Initial mean
Si_init = zeros(dim,dim,k_init);  % covariance
for i=1:k_init
    Si_init(:,:,i) = Si_init(:,:,i)+1*eye(dim);
end

Pi_init = ones(k_init,1)./k_init;

clear i

% Initial Beta
beta_init = 0;
                   
%sigma_t_init   = zeros(max_it,1);     % squared difference between COV and target COV
samplesU  = cell(max_it,1);           % space for the samples in the standard space

%% initialize storage parameters

mu_hat = mu_init;  % Initial mean sampled on unit hypersphere
Si_hat = Si_init;  % Concentration parameter
Pi_hat = Pi_init;

% Generate structure array to store the parameters
v_tot = struct("mu_hat",cell(max_it,1),"Si_hat",cell(max_it,1),"Pi_hat",cell(max_it,1));
v_tot(1) =  struct("mu_hat",mu_hat,"Si_hat",Si_hat,"Pi_hat",Pi_hat);

beta_hat = beta_init;
beta_tot = zeros(max_it,1);

cv_tot = zeros(max_it,1);

%% Iteration Section
% Start the counter

for j = 1:max_it 
    fprintf(['\n\nstep ',num2str(j),' for beta=',num2str(beta_hat)])
    
    % Generate samples
    X = GM_sample(v_tot(j).mu_hat,...
                  v_tot(j).Si_hat,...
                  v_tot(j).Pi_hat,N);

    samplesU{j} = X;

    % Add the number of samples generated
    N_tot = N_tot+N;
    
    % evaluation of the limit state function
    geval = G_LSF(X);

    % Replace the "NaN" values from the estimation of the limit state
    % function
    geval(isnan(geval)) = min(geval,[],"all");  
    logPriorpdf_ = jointpdf_prior(X,dim);

    % Update beta
    beta_hat = new_beta(beta_hat,X,v_tot(j),geval,logPriorpdf_,N,CV_target);
    
    % Handler (if beta is greater than 1, set to 1)
    if beta_hat > 1
        beta_hat = 1;
    end
    beta_tot(j+1) = beta_hat;
    
    % likelihood ratio
    logWconst_ = log_W_const(X,v_tot(j),logPriorpdf_);
    logW = logwl(beta_hat,geval,logWconst_); 
    
    cv_tot(j) = std(exp(logW))/mean(exp(logW)); % only for diagnostics, hence not optimized for precision
    
    % update parameter
    v_new = vnew(X,logW,k_init);
    v_tot(j+1) = v_new;
    
    if beta_hat>=1
        break;
    end
        
end

%% Adjust storing variables
v_tot = v_tot(1:j+1);

samplesU = samplesU(1:j,1);

beta_tot = beta_tot(1:j+1);
cv_tot = cv_tot(1:j);


%% results
% transform the samples to the physical/original space
samplesX = cell(length(samplesU),1);
if samples_return ~= 0
	for m = 1:numel(samplesU)
		samplesX{m} = u2x(samplesU{m});
        if ~prod(size(samplesX{m}) == size(samplesU{m}))
            samplesX{m} = samplesX{m}';
        end
	end
end

% Set the sample output based on the return parameter value
if samples_return == 0
    samplesU = {};
    samplesX = {};
elseif samples_return == 1
    samplesU = samplesU(end);
    samplesX = samplesX(end);
end

% Produce new failure samples after end step
ulast = GM_sample(v_tot(end).mu_hat,v_tot(end).Si_hat,...
    v_tot(end).Pi_hat,i_parser.Results.N_last);


N_tot = N_tot+i_parser.Results.N_last;

logWlast = logwl2(ulast,v_tot(end),G_LSF,1);
Wlast = exp(logWlast);
Wlast_normed = exp(logWlast-logsumexp(logWlast,2));

% Compute nESS
nESS = exp(2*logsumexp(logWlast)-logsumexp(2*logWlast)-log(i_parser.Results.N_last));

evidence = exp(logsumexp(logW)-log(i_parser.Results.N_last)); % Evidence
 
xlast = u2x(ulast);

if prod(size(xlast) == [N, 1])
    xlast = xlast';
end

k_fin = size(v_tot(end).mu_hat,1);

%% Generate Random Samples

f_s_iid = MErandsample(xlast,i_parser.Results.N_last,true,Wlast_normed,...
                       i_parser.Results.sampling_method)';
fprintf(['\nfinished after ',num2str(j-1),' steps at beta=',num2str(beta_hat),'\n'])

if j == max_it
    disp("\n-Exit with no convergence at max iterations \n")
end
end


%% NESTED FUNCTIONS

%===========================================================================
function X = GM_sample(mu,Si,Pi,N)
% Returns samples from the Gaussian mixture
if size(mu,1) == 1
    X = mvnrnd(mu,Si,N);
else
    ind = randsample(size(mu,1),N,true,Pi);
    z = histcounts(ind,[(1:size(mu,1)) size(mu,1)+1]);
    X = ones(N,size(mu,2));
    ind = 1;
    for i = 1:size(mu,1)
        np = z(i);
        X(ind:ind+np-1,:) = mvnrnd(mu(i,:),Si(:,:,i),np);
        ind = ind+np;
    end
end

end


%===========================================================================
function y = loggausspdf(X, mu, Sigma)
% Computes the logarithm of a Multi-Gaussian Probability Density function
d = size(X,1);
[U,~] = chol(Sigma);
% Try this error handle to check the error in the dimensions
try
    X = bsxfun(@minus,X,mu);
    Q = U'\X;
catch 
    try
        X = bsxfun(@minus,X,mu');
        Q = U'\X';
    catch
        error("The variables X and mu have mismatching sizes!");
    end
end

q = dot(Q,Q,1); % quadratic term (Mahalanobis distance)
c = d*log(2*pi)+2*sum(log(diag(U))); % normalization constant
y = -(c+q)/2;

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
function h = mixture_log_pdf(X,mu,si,ki)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%     X   samples             [dim x Nsamples]
%     mu  mean                [Nmodes x dim]
%     si  covariance          [dim x dim x Nmodes]
%     ki  weights of modes    [1 x Nmodes]
% Output:
%     h   logpdf of samples   [1 x Nsamples]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Invert the number of samples
[dim,N] = size(X);

if dim > N
    X = transpose(X); % transpose the sample array
end

N = size(X,2); % number of samples
k_tmp = length(ki); % number of modes
if k_tmp == 1
    h = loggausspdf(X,mu',si);
else
    h_pre = zeros(k_tmp,N);
    for q = 1:k_tmp
        mu_ = mu(q,:);
        si_ = si(:,:,q);
        h_pre(q,:) = log(ki(q)) + loggausspdf(X,mu_',si_);
    end
    h = logsumexp(h_pre,1);
end

end


%===========================================================================
function pdf = jointpdf_prior(X,dim)
% Computes the joint pdf of the prior distribution
pdf = loggausspdf(X',zeros(dim,1),eye(dim));

end

%===========================================================================
function newbeta = new_beta(betaold,samples_U,v,logLikelihood_,logpriorpdf_,Nstep,cvtarget)

% Function which performs the update of the new parameter beta    
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
if newbeta >=1-1e-2
    newbeta=1;
end


end



%===========================================================================
function newv = vnew(X,logWold,k)
% This function updates the parameters of the parametric density
if k == 1
    fprintf('\n...using analytical formulas')
    sumwold = exp(logsumexp(logWold,2));

    dim = size(X,1);
    
    signu = sign(X);
    logu = log(abs(X));
    
    [mu_unnormed,signmu] = signedlogsumexp(logu+logWold,2,signu);
    mu = exp(mu_unnormed-logsumexp(logWold,2)).*signmu;

    v0 = mu';
    
    v1 = zeros(dim,dim,k);
    sqrtR = exp(0.5.*logWold);
    Xo = bsxfun(@minus,X,v0(1,:)');
    Xo = bsxfun(@times,Xo,sqrtR);
    v1(:,:,1) = Xo*Xo'./sumwold;
    v1(:,:,1) = v1(:,:,1)+eye(dim)*(1e-6);
    
    v2 = [1];
    
    new_var = {v0,v1,v2};
else
    if all(X==0)
        fprintf('issue with samples')
    end
    fprintf('\n...using EMGM_log ')
    [v0,v1,v2] = EMGM_log(transpose(X),logWold,k);
    new_var = {v0',v1,v2};
end

% Restructure the output as a dynamic memory variable (STRUCTURE)
newv = struct("mu_hat",new_var{1},"Si_hat",new_var{2},"Pi_hat",new_var{3});
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
    logconstw = logpriorpdf-mixture_log_pdf(samples_U,v.mu_hat,v.Si_hat,v.Pi_hat);
end

% =========================================================================
function logS = logstep(Nstep,cvtarget)
    logS = log(Nstep)-log(1+cvtarget^2);
end

% =========================================================================
function logW = logwl(beta,logL,logWconst)

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

