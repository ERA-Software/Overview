function [mu, kappa, m, omega, alpha] = EMvMFNM_log(X,logW,k)
%% Perform soft EM algorithm for fitting the von Mises-Fisher-Nakagami mixture model.
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
* X : data matrix (dimensions x Number of samples)
* logW : vector of log-likelihood-ratios for weighted samples
* k : number of vMFN-distributions in the mixture
---------------------------------------------------------------------------
Output:
* mu    : mean directions
* kappa : approximated concentration parameter
* m     : approximated shape parameter
* omega : spread parameter
* alpha : distribution weights
---------------------------------------------------------------------------
Based on:
1. "EM Demystified: An Expectation-Maximization Tutorial"
   Yihua Chen and Maya R. Gupta
   University of Washington, Dep. of EE (Feb. 2010)
---------------------------------------------------------------------------
%}

%% initialization
logM      = initialization(X,k);
R         = sqrt(sum(X.^2))';
X_norm    = bsxfun(@times,X,1./R');
%
tol       = 1e-5;
maxiter   = 500;
llh       = -inf(2,maxiter);
converged = false;
t         = 1;

%% soft EM algorithm
while ~converged && t <= maxiter
   t            = t+1;
   [~,label(:)] = max(logM,[],2);
   u            = unique(label);   % non-empty components
   if size(logM,2) ~= size(u,2)
      logM = logM(:,u);   % remove empty components
   end
   %
   [mu,kappa,m,omega,alpha] = maximization(X_norm,logW,R,logM);
   [logM,llh(:,t)]             = expectation(X_norm,logW,mu,kappa,m,omega,alpha,R);
   %
   if t > 2
      con1      = abs(llh(1,t)-llh(1,t-1)) < tol*abs(llh(1,t));
      con2      = abs(llh(2,t)-llh(2,t-1)) < tol*100*abs(llh(2,t));
      converged = min(con1,con2);
   end
end
%
if converged
   fprintf('Converged in %d steps.\n',t-1);
else
   fprintf('Not converged in %d steps.\n',maxiter);
end

return;


%===========================================================================
%===========================NESTED FUNCTIONS================================
%===========================================================================
function logM = initialization(X,k)
% % Random initialization
% [~,n]       = size(X);
% 
% label       = ceil(k*rand(1,n));
% [u,~,label] = unique(label);
% while k ~= length(u)
%    label       = ceil(init.k*rand(1,n));
%    [u,~,label] = unique(label);
% end

% kmeans initialization
[~,n] = size(X);
label = kmeans(X',k); % kmeans from MATLAB

% % kmeans++ initialization with code from "Laurent S (2020). k-means++ (https://www.mathworks.com/matlabcentral/fileexchange/28804-k-means), MATLAB Central File Exchange. Retrieved June 9, 2020." 
% [~,n] = size(X);
% [label,~] = kmeans_plusplus(X,k); % kmeans++ algorithm; dimensioning of X transposed to the original matlab-code

% % dbscan initialization
% [~,n] = size(X);
% label = (k+1)*ones(n,1);
% epsilon = 0.1;
% while max(label)>k || min(label)<1
%     label = dbscan(X',epsilon,50);
%     epsilon = epsilon*1.01;
% end

% build M
logM = ones(n,k)*(-1e200);
for i=1:n
    logM(i,label(i))=0;
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
function [logM,llh] = expectation(X,logW,mu,kappa,m,omega,alpha,R)
n           = size(X,2);
k           = size(mu,2);
logvMF      = zeros(n,k);
lognakagami = zeros(n,k);
logpdf      = zeros(n,k);

% logpdf
for i = 1:k
   logvMF(:,i)      = (logvMFpdf(X,mu(:,i),kappa(i)))';
   lognakagami(:,i) = lognakagamipdf(R,m(i),omega(i));
   logpdf(:,i)      = logvMF(:,i)+lognakagami(:,i)+log(alpha(i));
end

% Matrix of posterior probabilities
T         = logsumexp(logpdf,2);
logM      = bsxfun(@minus,logpdf,T);
logM = logM-logsumexp(logM,2);

% loglikelihood as tolerance criterion
logvMF_weighted      = bsxfun(@plus,logvMF,log(alpha));
lognakagami_weighted = bsxfun(@plus,lognakagami,log(alpha));
T_vMF                = logsumexp(logvMF_weighted,2);
T_nakagami           = logsumexp(lognakagami_weighted,2);

llh1                 = [sum(logW'+T_vMF,1)-logsumexp(logW,2);sum(logW'+T_nakagami,1)-logsumexp(logW,2)];
llh                  = llh1;
if any(any(isnan(logM))) || any(any(isinf(logM)))
    fprintf('\nEMvMFNM_log: issue with M\n')
end
return;


%===========================================================================
function [mu,kappa,m,omega,alpha] = maximization(X_norm,logW,R,logM)
logMW = logW'+logM;
[d,~] = size(X_norm);
lognk = logsumexp(logMW,1);

if any(isinf(lognk)) || any(isnan(lognk))
    lognk(isinf(lognk)) = -1e50;
    lognk(isnan(lognk)) = -1e50;
end

%  distribution weights
alpha = exp(lognk - logsumexp(logW,2));

% mean directions
signX = sign(X_norm);
logX = log(abs(X_norm));

mu = zeros(d,length(lognk));
norm_mu = zeros(1,length(lognk));
lognorm_mu = zeros(1,length(lognk));
for i=1:length(alpha)
    [mu_unnormed,signmu] = signedlogsumexp(logX+logMW(:,i)',2,signX);
    lognorm_mu(i) = 0.5*logsumexp(2*mu_unnormed,1);
    mu(:,i) = exp(mu_unnormed-lognorm_mu(i)).*signmu;
end

% approximated concentration parameter
xi = min(exp(lognorm_mu-lognk),0.999999999);
% xi = min(exp(lognorm_mu-lognk),0.95);
kappa = (xi.*d-xi.^3)./(1-xi.^2);

% spread parameter
R(R==0)=1e-300; % for numerical stability; should actually be equal to the closest number to zero without cancellation error
logR = log(R);
logRsquare = 2*logR;
omega = exp(logsumexp(logMW+logRsquare,1) - logsumexp(logMW,1));

% approximated shape parameter
logRpower = 4*logR;
mu4       = exp(logsumexp(logMW+logRpower,1) - logsumexp(logMW,1));
m         = omega.^2./(mu4-omega.^2);

m(m<0.5) = d/2;
m(isinf(m)) = d/2;
return;


%===========================================================================
function y = logvMFpdf(X, mu, kappa)

d = size(X,1);
if kappa==0
   % unit hypersphere uniform log pdf
   A = log(d)+log(pi^(d/2))-gammaln(d/2+1);
   y = -A;
elseif kappa>0
   c = (d/2-1)*log(kappa)-(d/2)*log(2*pi)-logbesseli(d/2-1,kappa);
   q = bsxfun(@times,mu,kappa)'*X;
   y = bsxfun(@plus,q,c');
else
   error('ERROR: k<0 or NaN');
end
return;


%===========================================================================
function y = lognakagamipdf(X,m,om)
y = log(2)+m*(log(m)-log(om)-X.^2./om)+log(X).*(2*m-1)-gammaln(m);
return;


%===========================================================================
function logb = logbesseli(nu,x)
% log of the Bessel function, extended for large nu and x
% approximation from Eqn 9.7.7 of Abramowitz and Stegun
% http://www.math.sfu.ca/~cbm/aands/page_378.htm

if nu == 0   % special case when nu=0
   logb = log(besseli(0,nu,1))+abs(nu); % for stability
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
return;

%%END