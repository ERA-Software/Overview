function [mu, si, pi] = EMGM_log(X,logW,nGM)
%% Perform soft EM algorithm for fitting the Gaussian mixture model
%{
---------------------------------------------------------------------------
Created by:
Michael Engel (m.engel@tum.de)

Based upon the code of:
Sebastian Geyer (s.geyer@tum.de)
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2020-07-29
---------------------------------------------------------------------------
Input:
* X   : input samples
* logW   : initial guess for the logweights
* nGM : number of Gaussians in the Mixture
---------------------------------------------------------------------------
Output:
* mu : [npi x d]-array of means of Gaussians in the Mixture
* si : [d x d x npi]-array of cov-matrices of Gaussians in the Mixture
* pi : [npi]-array of weights of Gaussians in the Mixture (sum(Pi) = 1)
---------------------------------------------------------------------------
Based on:
1. "EM Demystified: An Expectation-Maximization Tutorial"
   Yihua Chen and Maya R. Gupta
   University of Washington, Dep. of EE (Feb. 2010)
---------------------------------------------------------------------------
%}

%% initialization
logR = initialization(X,nGM);
%
tol       = 1e-5;
maxiter   = 500;
llh       = -inf(1,maxiter);
converged = false;
t         = 1;

%% soft EM algorithm
while ~converged && t < maxiter
   t            = t+1;
   [~,label(:)] = max(logR,[],2);
   u            = unique(label);   % non-empty components
   if size(logR,2) ~= size(u,2)
      logR = logR(:,u);   % remove empty components
   end
   [mu, si, pi] = maximization(X, logW, logR);
   [logR, llh(t)]  = expectation(X, logW, mu, si, pi);
   if t > 2
      converged = abs(llh(t)-llh(t-1)) < tol*abs(llh(t));
   end
end
if converged
   fprintf('Converged in %d steps.\n',t-1);
else
   fprintf('Not converged in %d steps.\n',maxiter);
end

return;


%===========================================================================
%===========================NESTED FUNCTIONS================================
%===========================================================================
function logR = initialization(X,k)
% % Random initialization
% [~,n]       = size(X);
% label       = ceil(k*rand(1,n));
% [u,~,label] = unique(label);
% while k ~= length(u)
%    label       = ceil(init.k*rand(1,n));
%    [u,~,label] = unique(label);
% end


% Check 
% kmeans initialization
[~,n] = size(X);
label = kmeans(X',k); % kmeans from MATLAB

% build R
logR = ones(n,k)*(-1e200);
for i=1:n
    logR(i,label(i))=0;
end

% % visualization
% figure(20)
% close;
% figure(20)
% hold on;
% for k_=1:k
%     scatter(X(1,label==k_),X(2,label==k_))
% end
% hold off;
% pause(0.25);

return;

%===========================================================================
function [logR, llh] = expectation(X, logW, mu, si, pi)

% Check 
n      = size(X,2);
k      = size(mu,2);
logpdf = zeros(n,k);
for i = 1:k
   logpdf(:,i) = loggausspdf(X,mu(:,i),si(:,:,i));
end
%
logpdf = bsxfun(@plus,logpdf,log(pi));
T      = logsumexp(logpdf,2);
llh    = sum(logW'+T)-logsumexp(logW,2);

logR = bsxfun(@minus,logpdf,T);
logR = logR-logsumexp(logR,2);

if isnan(logR)
    fprintf('\nlogR is Nan')
end

return;


%===========================================================================
function [mu, Sigma, w] = maximization(X,logW,logR)
logRW     = logW'+logR;
[d,~] = size(X);
k     = size(logRW,2);
nk = exp(logsumexp(logRW,1));
lognk = logsumexp(logRW,1);
if any(nk == 0) % numerical stability
   nk = nk + 1e-6;
end
% sumW = exp(logsumexp(logW,2));
% w = nk/sumW;
w = exp(logsumexp(logRW,1)-logsumexp(logW));

signX = sign(X);
logX = log(abs(X));

mu = zeros(d,k);
for i=1:k
    [mu_unnormed,signmu] = signedlogsumexp(logX+logRW(:,i)',2,signX);
    mu(:,i) = exp(mu_unnormed-logsumexp(logRW(:,i)')).*signmu;
end

Sigma = zeros(d,d,k);
sqrtRW = exp(0.5.*logRW);
% linear-version as log-version slow
for i = 1:k
    Xo = bsxfun(@minus,X,mu(:,i));
    Xo = bsxfun(@times,Xo,sqrtRW(:,i)');
%     Sigma(:,:,i) = exp(log(Xo*Xo')-lognk(i));
    Sigma(:,:,i) = Xo*Xo'/nk(i);
    Sigma(:,:,i) = Sigma(:,:,i)+eye(d)*(1e-5); % add a prior for numerical stability
%     Sigma(:,:,i) = diag(diag(Sigma(:,:,i))); %%%
    
if (isnan(Sigma))
    fprintf('\nSigma is Nan')
end

if (isinf(w))
    fprintf('\nAlpha is Inf')
end
    
end
return;


%===========================================================================
function y = loggausspdf(X, mu, Sigma)
d     = size(X,1);
X     = bsxfun(@minus,X,mu);
[U,~] = chol(Sigma);
Q     = U'\X;
q     = dot(Q,Q,1);  % quadratic term (M distance)
c     = d*log(2*pi)+2*sum(log(diag(U)));   % normalization constant
y     = -(c+q)/2;
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
return;
%%END