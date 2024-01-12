function [samplesU, samplesX, q, logcE] = iTMCMC(Ns, Nb, log_likelihood, T_nataf)
%% iTMCMC function
%{
---------------------------------------------------------------------------
Created by:
Fong-Lin Wu
Felipe Uribe
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Current version 2021-03
* use of log-evidence logcE
---------------------------------------------------------------------------
Version 2020-10
* Adaptation to new ERANataf class
Version 2019-06
* Clean up code and modify while loop
Version 2017-04
* License check for optimization toolbox -> Use of fsolve if available in case 
  fzero does not work
---------------------------------------------------------------------------
Input:
* Ns             : number of samples per level
* Nb             : number of samples for burn-in
* log_likelihood : log-likelihood function of the problem at hand
* T_nataf        : Nataf distribution object (probabilistic transformation)
---------------------------------------------------------------------------
Output:
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* q        : array of tempering parameter for each level
* logcE    : log-evidence
---------------------------------------------------------------------------
References:
1."Transitional Markov Chain Monte Carlo: Observations and improvements".
   Wolfgang Betz et al.
   Journal of Engineering Mechanics. 142.5 (2016) 04016016.1-10
2."Transitional Markov Chain Monte Carlo method for Bayesian model updating,
   model class selection and model averaging".
   Jianye Ching & Yi-Chun Chen
   Journal of Engineering Mechanics. 133.7 (2007) 816-832
---------------------------------------------------------------------------
%}

%% transform to the standard Gaussian space
if any(strcmp('Marginals',fieldnames(T_nataf))) == 1   % use Nataf transform (dependence)
    d   = length(T_nataf.Marginals);    % number of random variables (dimension)
    u2x = @(u) T_nataf.U2X(u);          % from u to x
    prior_pdf = @(t) T_nataf.pdf(t);
    
else   % use distribution information for the transformation (independence)
    % Here we are assuming that all the parameters have the same distribution !!!
    % Adjust accordingly otherwise or use an ERANataf object
    d   = length(T_nataf);                    % number of random variables (dimension)
    u2x = @(u) T_nataf(1).icdf(normcdf(u));   % from u to x
    prior_pdf = @(t) T_nataf(1).pdf(t);
end

%% some constants and initialization
beta    = 2.4/sqrt(d);                 % prescribed scaling factor (recommended choice)
t_acr   = 0.21/d + 0.23;               % target acceptance rate
Na      = 100;                         % number of chains to adapt
thres_p = 1;                           % threshold for the c.o.v (100% recommended choice)
max_it  = 20;                          % max number of iterations (for allocation)
j       = 0;                           % initialize counter for intermediate levels

% memory allocation
samplesU = cell(max_it,1);            % space for the samples in the standard space
samplesX = cell(max_it,1);            % space for the samples in the standard space
logS_j    = ones(max_it,1);            % space for factors logSj
q        = zeros(max_it,1);           % store tempering parameters

%% 1. Obtain N samples from the prior pdf and evaluate likelihood
u_j     = randn(Ns,d);    % u_0 (Nxd matrix)
theta_j = u2x(u_j);      % theta_0 (transform the samples)
logL_j  = zeros(Ns,1);
for i = 1:Ns
   logL_j(i) = log_likelihood(theta_j(i,:));
end
samplesU{1} = u_j;           % store initial level samples
samplesX{1} = theta_j;       % store initial level samples

%% iTMCMC
while q(j+1) < 1 && (j < max_it)  % adaptively choose q
   j = j+1;
   fprintf('\niTMCMC intermediate level j = %g, with q_{j} = %g\n', j-1, q(j));
   
   % 2. Compute tempering parameter q_{j+1}
   % dq   = q_{j+1}-q_{j}
   fun = @(dq) std(exp(abs(dq)*logL_j)) - thres_p*mean(exp(abs(dq)*logL_j));   % c.o.v equation
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
      q(j+1) = min(1, q(j)+dq);
   else
      q(j+1) = 1;
      fprintf('Variable q was set to %f, since it is not possible to find a suitable value\n',q(j+1));
   end
   
   % 3. Compute log-'plausibility weights' logw_j and factors logS_j for the evidence
   logw_j = (q(j+1)-q(j))*logL_j;
   logwsum = logsumexp(logw_j);
   logS_j(j) = logwsum-log(Ns);
   
   % 4. Metropolis resampling step to obtain N samples from f_{j+1}(theta)
   % weighted sample mean
   w_j_norm = exp(logw_j-logwsum);          % compute normalized weights
   mu_j     = u_j'*w_j_norm;   % sum inside [Ref. 2 Eq. 17]
   
   % Compute scaled sample covariance matrix of f_{j+1}(theta)
   Sigma_j = zeros(d);
   for k = 1:Ns
      tk_mu   = u_j(k,:) - mu_j';
      Sigma_j = Sigma_j + w_j_norm(k)*(tk_mu'*tk_mu);   % [Ref. 2 Eq. 17]
   end
   
   % target pdf prior(theta)*likelihood(theta)^q(j)
   level_post = @(t,log_L) prior_pdf(t).*exp(q(j+1)*log_L); % [Ref. 2 Eq. 11]
   
   % Start N different Markov chains
   msg     = fprintf('* M-H sampling...\n');
   u_c     = u_j;
   theta_c = theta_j;
   logL_c  = logL_j;
   Nadapt  = 1;
   na      = 0;
   acpt    = 0;
   for k = 1:(Ns+Nb)      
      % select index l with probability w_j
      l = resampling_index(exp(logw_j));
      
      % sampling from the proposal and evaluate likelihood
      u_star     = mvnrnd(u_c(l,:),(beta^2)*Sigma_j);
      theta_star = u2x(u_star);   % transform the sample
      logL_star  = log_likelihood(theta_star);
      
      % compute the Metropolis ratio
      ratio = level_post(theta_star  ,logL_star)/...
              level_post(theta_c(l,:),logL_c(l));
      
      % accept/reject step
      if rand <= ratio 
         u_c(l,:)     = u_star;
         theta_c(l,:) = theta_star;
         logL_c(l)    = logL_star;
         acpt         = acpt+1;
      end
      if k > Nb   % (Ref. 1 Modification 2: burn-in period)
         u_j(k-Nb,:)     = u_c(l,:);
         theta_j(k-Nb,:) = theta_c(l,:);
         logL_j(k-Nb)    = logL_c(l);
      end
      
      % recompute the weights (Ref. 1 Modification 1: update sample weights)
      logw_j(l) = (q(j+1)-q(j))*logL_c(l);
      
      % adapt beta (Ref. 1 Modification 3: adapt beta)
      na = na+1;
      if na >= Na
         p_acr  = acpt/Na;
         ca     = (p_acr - t_acr)/sqrt(Nadapt);
         beta   = beta*exp(ca);
         Nadapt = Nadapt+1;
         na     = 0;
         acpt   = 0;         
      end
   end
   fprintf(repmat('\b',1,msg));
   
   % store samples
   samplesU{j+1} = u_j;
   samplesX{j+1} = theta_j;
end

% delete unnecesary data
samplesU(j+2:end)  = [];
samplesX(j+2:end)  = [];   
q(j+2:end)         = [];
logS_j(j+1:end)     = [];

%% log-evidence
logcE = sum(logS_j);

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