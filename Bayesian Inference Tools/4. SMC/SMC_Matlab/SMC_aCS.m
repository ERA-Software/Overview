function [samplesU, samplesX, q, logcE] = SMC_aCS(N, p, log_likelihood, distr, burn, tarCoV)
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
* The SMC method in combination with the adaptive conditional M-H (or pCN) sampler
  (aCS) performs well in high-dimensions. For low-dimensional problems, the
  Gaussian mixture proposal should be chosen over aCS.
* The way the initial standard deviation is computed can be changed in line 85.
  By default we use option 'a' (it is equal to one).
  In option 'b', it is computed from the seeds.
---------------------------------------------------------------------------
Input:
* N      : number of samples per level
* p      : N/number of chains per level
* log_likelihood  : log-likelihood function
* distr  : Nataf distribution object or
           marginal distribution object of the input variables
* burn   : burn-in period
* tarCoV : target coefficient of variation of the weights
---------------------------------------------------------------------------
Output:
* logcE       : log-evidence
* q    : tempering parameters
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
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
opc     = 'a';                % way to estimate the std for the aCS (see comments)

% parameters for adaptive MCMC
adapflag   = 1;
adapchains = ceil(100*nchain/nsamlev);  % number of chains after which the proposal is adapted
lambda     = 0.6;

%% SMC aCS
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
   
   %===== Step 5: resample
   % seeds for chains
   ind = randsample(nsamlev,nchain,true,wnork);
   logLk0 = logLk(ind);
   uk0 = uk(ind,:);
   
   %===== Step 6: perform aCS
   % compute initial standard deviation
   switch opc
      case 'a'   % 1a. sigma = ones(n,1);
         sigmaf = 1;
      case 'b'   % 1b. sigma = sigma_hat; (sample standard deviations)
         muf = wnork*uk;
         sigmaf = wnork*(uk-muf).^2;         
      otherwise
         error('Choose a or b');
   end

   % compute parameter rho
   sigmafk = min(1, lambda*sigmaf);
   rhok    = sqrt(1-sigmafk.^2);
   counta  = 0;
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
         
         % get candidate sample from conditional normal distribution
         ucand = normrnd(rhok*u0', sqrt(1-rhok^2))';
         
         % Evaluate log-likelihood function
         logLcand = loglike_fun(ucand);
         
         % compute acceptance probability
         alpha     = min(1,exp(q(m+1)*(logLcand-logL0)));
         alphak(k) = alphak(k)+alpha/(lenchain+burn);
         
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
      
      % adapt the chain correlation
      if adapflag == 1
         % check whether to adapt now
         if mod(k,adapchains) == 0
            % mean acceptance rate of last adap_chains
            alpha_mu = mean(alphak(k-adapchains+1:k));
            counta   = counta+1;
            gamma    = counta^(-0.5);
            lambda   = exp(log(lambda)+gamma*(alpha_mu-0.44));
            
            % compute parameter rho
            sigmafk = min(1,lambda*sigmaf);
            rhok    = sqrt(1-sigmafk.^2);
         end
      end
   end
   uk = uk(1:nsamlev,:);
   logLk = logLk(1:nsamlev);
   
   % save samples
   samplesU{m+1} = uk';
   
   % compute mean acceptance rate of all chains in level m
   accrate(m) = mean(alphak);

   fprintf('\t*aCS sigma = %g \t *aCS accrate = %g\n', sigmafk, accrate(m));

end
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