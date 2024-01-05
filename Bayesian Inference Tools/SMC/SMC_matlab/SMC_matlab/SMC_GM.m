function [samplesU, samplesX, q, k_fin, logcE] = SMC_GM(N, p, log_likelihood, distr, k_init, burn, tarCoV)
%% Sequential Monte Carlo using adaptive conditional sampling (pCN)
%{
---------------------------------------------------------------------------
Created by:
Iason Papaioannou (iason.papaioannou@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2021-03
---------------------------------------------------------------------------
Comments:
* The SMC method in combination with a Gaussian Mixture model can only be
  applied for low-dimensional problems, since its accuracy decreases
  dramatically in high dimensions.---------------------------------------------------------------------------
Input:
* N      : number of samples per level
* p      : N/number of chains per level
* log_likelihood  : log-likelihood function
* distr  : Nataf distribution object or
           marginal distribution object of the input variables
* k_init : initial number of Gaussians in the mixture model
* burn   : burn-in period
* tarCoV : target coefficient of variation of the weights
---------------------------------------------------------------------------
Output:
* logcE       : log-evidence
* q    : tempering parameters
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* k_fin    : final number of Gaussians in the mixture
---------------------------------------------------------------------------
Based on:
1."A sequential particle filter method for static models"
   Chopin
   Biometrika 89 (3) (2002) 539-551

2."Inference for Levy-driven stochastic volatility models via adaptive sequential Monte Carlo"
   Jasra et al.
   Scand. J. Stat. 38 (1) (2011) 1-22

3."Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
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
   % Adjust accordingly otherwise or use an ERANataf object
   dim = length(distr);                    % number of random variables (dimension)
   u2x = @(u) distr(1).icdf(normcdf(u));   % from u to x   
end

%% log likelihood in standard space
loglike_fun = @(u) log_likelihood(u2x(u));

%% Initialization of variables and storage
max_it = 100;    % estimated number of iterations
m      = 0;      % counter for number of levels

% Properties of SMC
nsamlev  = N;                % number of samples
nchain   = nsamlev*p;        % number Markov chains
lenchain = nsamlev/nchain;   % number of samples per Markov chain
tarESS = nsamlev/(1+tarCoV^2); % target effective sample size


% initialize samples
logLk = zeros(1,nsamlev);   % space for evaluations of log likelihood
accrate = zeros(max_it,1);    % space for acceptance rate
q       = zeros(max_it,1);    % space for the tempering parameters
logSk      = ones(max_it,1);     % space for log-expected weights

%% SMC GM
%===== Step 1: Perform the first Monte Carlo simulation
uk = randn(nsamlev,dim);    % initial samples
for k = 1:nsamlev
   logLk(k) = loglike_fun(uk(k,:));   % evaluate likelihood    
end
% save samples
samplesU{m+1} = uk';

while q(m+1) < 1 && (m < max_it)  % adaptively choose q
   m = m+1;
   %===== Step 2 and 3: compute tempering parameter and weights   
   
   fun = @(dq) exp(2*logsumexp(abs(dq)*logLk)-logsumexp(2*abs(dq)*logLk)) - tarESS;   % ESS equation
   [dq,~,flag] = fzero(fun, 0);   
   
   % if fzero does not work try with fsolve
   if flag > 0   % OK
      dq = abs(dq);  % dq is >= 0   
   elseif license('test','optimization_toolbox')
      option = optimset('Display','off');
      [dq,~,flag] = fsolve(fun, 0, option);
      dq          = abs(dq);  % dq is >= 0
      if flag < 0
         error('fzero and fsolve do not converge');
      end
   else 
      error('no optimization_toolbox available');
   end
   %
   if ~isnan(dq)
      q(m+1) = min(1, q(m)+dq);
   else
      q(m+1) = 1;
      fprintf('Variable q was set to %f, since it is not possible to find a suitable value\n',q(m+1));
   end   
   
   % log-weights
   logwk = (q(m+1)-q(m))*logLk;
    
   %===== Step 4: compute estimate of log-expected w
   logwsum=logsumexp(logwk);
   logSk(m) = logwsum-log(nsamlev);

   wnork = exp(logwk-logwsum);          % compute normalized weights
   
   [mu, si, ww] = EMGM(uk',wnork',k_init);    % fit Gaussian Mixture   
   
   %===== Step 5: resample
   % seeds for chains
   ind = randsample(nsamlev,nchain,true,wnork);
   logLk0 = logLk(ind);
   uk0 = uk(ind,:);
   
   %===== Step 6: perform independent M-H
   count   = 0; 
   % initialize chain acceptance rate
   alphak = zeros(nchain,1);
   logLk     = [];                % delete previous samples
   uk     = [];                % delete previous samples
   for k = 1:nchain
      % set seed for chain
      u0 = uk0(k,:);
      logL0 = logLk0(k);
      for j = 1:lenchain+burn
         count = count+1;
         if j == burn+1
            count = count-burn;
         end
         % get candidate sample from the fitted GM distribution
         indw  = randsample(length(ww), 1, true,ww);
         ucand = mvnrnd(mu(:,indw), si(:,:,indw));
         
         % Evaluate log-likelihood function
         logLcand = loglike_fun(ucand);
         
         % compute acceptance probability
         logpdfn = zeros(length(ww),1);
         logpdfd = zeros(length(ww),1);
         for ii = 1:length(ww)
            logpdfn(ii) = log(ww(ii))+loggausspdf(u0',mu(:,ii),si(:,:,ii));
            logpdfd(ii) = log(ww(ii))+loggausspdf(ucand',mu(:,ii),si(:,:,ii));
         end 
         logpdfnsum = logsumexp(logpdfn);
         logpdfdsum = logsumexp(logpdfd);
         logpriorn = -0.5*ucand*ucand';
         logpriord = -0.5*u0*u0';
         alpha     = min(1,exp(q(m+1)*(logLcand-logL0)+logpriorn-logpriord+logpdfnsum-logpdfdsum));
         alphak(k) = alphak(k) + alpha/(lenchain+burn);
         
         
         % check if sample is accepted
         uhelp = rand;
         if uhelp <= alpha
            uk(count,:) = ucand;
            logLk(count) = logLcand;
            u0  = ucand;
            logL0  = logLcand;
         else
            uk(count,:) = u0;
            logLk(count) = logL0;
         end
      end
      
   end
   uk = uk(1:nsamlev,:);
   logLk = logLk(1:nsamlev);
   
   % save samples
   samplesU{m+1} = uk';
   
   % compute mean acceptance rate of all chains in level m
   accrate(m) = mean(alphak);

   fprintf('\t *MH-GM accrate = %g\n', accrate(m));   

end
k_fin = length(ww); 
l_tot = m+1;
q = q(1:l_tot);
logSk = logSk(1:l_tot-1);

%% log-evidence
logcE = sum(logSk);

%% transform the samples to the physical/original space
samplesX = cell(l_tot,1);
for i = 1:l_tot
   samplesX{i} = u2x(samplesU{i}')';
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

%===========================================================================
function y = loggausspdf(X, mu, Sigma)
    d = size(X,1);
    X = bsxfun(@minus,X,mu);
    [U,~] = chol(Sigma);
    Q = U'\X;
    q = dot(Q,Q,1); % quadratic term (Mahalanobis distance)
    c = d*log(2*pi)+2*sum(log(diag(U))); % normalization constant
    y = -(c+q)/2;
return