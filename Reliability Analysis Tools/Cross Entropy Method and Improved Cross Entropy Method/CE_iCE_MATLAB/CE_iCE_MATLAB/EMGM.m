function [mu, si, pi] = EMGM(X,W,nGM)
%% Perform soft EM algorithm for fitting the Gaussian mixture model
%{
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
Fong-Lin Wu
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2020-02
---------------------------------------------------------------------------
Input:
* X   : input samples
* W   : initial guess for the weights
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
R = initialization(X,nGM);
%
tol       = 1e-5;
maxiter   = 500;
llh       = -inf(1,maxiter);
converged = false;
t         = 1;

%% soft EM algorithm
while ~converged && t < maxiter
   t            = t+1;
   [~,label(:)] = max(R,[],2);
   u            = unique(label);   % non-empty components
   if size(R,2) ~= size(u,2)
      R = R(:,u);   % remove empty components
   end
   [mu, si, pi] = maximization(X, W, R);
   [R, llh(t)]  = expectation(X, W, mu, si, pi);
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
function R = initialization(X, nGM)
% Random initialization

[~,n]       = size(X);
idx         = randsample(n,nGM);
m           = X(:,idx);
[~,label]   = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
[u,~,label] = unique(label);
while nGM ~= length(u)
   idx         = randsample(n,nGM);
   m           = X(:,idx);
   [~,label]   = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
   [u,~,label] = unique(label);
end
R = full(sparse(1:n,label,1,n,nGM,n));
return;


%===========================================================================
function [R, llh] = expectation(X, W, mu, si, pi)

n      = size(X,2);
k      = size(mu,2);
logpdf = zeros(n,k);
for i = 1:k
   logpdf(:,i) = loggausspdf(X,mu(:,i),si(:,:,i));
end
%
logpdf = bsxfun(@plus,logpdf,log(pi));
T      = logsumexp(logpdf,2);
llh    = sum(W.*T)/sum(W);
logR   = bsxfun(@minus,logpdf,T);
R      = exp(logR);
return;


%===========================================================================
function [mu, Sigma, w] = maximization(X,W,R)

R     = repmat(W,1,size(R,2)).*R;
[d,~] = size(X);
k     = size(R,2);
nk    = sum(R,1);
if any(nk == 0)   % prevent division by zero
   nk = nk + eps;
end
w     = nk/sum(W);
mu    = bsxfun(@times, X*R, 1./nk);
Sigma = zeros(d,d,k);
sqrtR = sqrt(R);
for i = 1:k
   Xo = bsxfun(@minus,X,mu(:,i));
   Xo = bsxfun(@times,Xo,sqrtR(:,i)');
   Sigma(:,:,i) = Xo*Xo'/nk(i);
   Sigma(:,:,i) = Sigma(:,:,i)+eye(d)*(1e-6); % add a prior for numerical stability
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
%%END