function [Pr, l_tot, samplesU, samplesX,W_final, f_s] = ...
    SIS_aCS(N, p, g_fun, distr, burn, tarCoV, samples_return)
%% Sequential importance sampling using adaptive conditional sampling
%{
---------------------------------------------------------------------------
Created by:
Iason Papaioannou (iason.papaioannou@tum.de)
Max Ehre (max.ehre@tum.de)
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2018-05
---------------------------------------------------------------------------
Current version: 2023-07
* Modification in the Sensitivity Analysis Output
---------------------------------------------------------------------------
Comments:
* The SIS method in combination with the adaptive conditional M-H sampler
  (aCS) performs well in high-dimensions. For low-dimensional problems, the
  Gaussian mixture proposal should be chosen over aCS.
* The way the initial standard deviation is computed can be changed in line 76.
  By default we use option 'a' (it is equal to one).
  In option 'b', it is computed from the seeds.
---------------------------------------------------------------------------
Input:
* N                    : number of samples per level
* p                    : N/number of chains per level
* g_fun                : limit state function
* distr                : Nataf distribution object or
                         marginal distribution object of the input variables
* burn                 : burn-in period
* tarCoV               : target coefficient of variation of the weights
* sensitivity_analysis : implementation of sensitivity analysis: 1 - perform, 0 - not perform
* samples_return       : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr       : probability of failure
* l_tot    : total number of levels
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* W_final  : final weights
* f_s      : i.i.d failure samples
---------------------------------------------------------------------------
Based on:
1."Sequential importance sampling for structural reliability analysis"
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
   u2x = @(u) reshape(distr.U2X(u),[],dim);          % from u to x
   
else   % use distribution information for the transformation (independence)
   % Here we are assuming that all the parameters have the same distribution !!!
   % Adjust accordingly otherwise or use an ERANataf object
   dim = length(distr);                    % number of random variables (dimension)
   u2x = @(u) reshape(distr(1).icdf(normcdf(u)),[],dim);   % from u to x   
end

%% LSF in standard space
g = @(u) g_fun(u2x(u)); 

%% Initialization of variables and storage
max_it = 100;    % estimated number of iterations
m      = 0;      % counter for number of levels

% Properties of SIS
nsamlev  = N;                % number of samples
nchain   = nsamlev*p;        % number Markov chains
lenchain = nsamlev/nchain;   % number of samples per Markov chain

% initialize samples
gk      = zeros(1,nsamlev);   % space for evaluations of g
accrate = zeros(max_it,1);    % space for acceptance rate
sigmak  = zeros(max_it,1);    % space for the standard deviation
Sk      = ones(max_it,1);     % space for expected weights
opc     = 'a';                % way to estimate the std for the aCS (see comments)

% parameters for adaptive MCMC
adapflag   = 1;
adapchains = ceil(100*nchain/nsamlev);  % number of chains after which the proposal is adapted
lambda     = 0.6;

samplesU  = cell(1,1);                  % space for the samples in the standard space

%% SIS aCS
%===== Step 1: Perform the first Monte Carlo simulation
uk = randn(nsamlev,dim);    % initial samples
for k = 1:nsamlev
   gk(k) = g(uk(k,:));
end

% save samples
samplesU{m+1} = uk;

% set initial subset and failure level
gmu = mean(gk);

for m = 1:max_it
   %===== Step 2 and 3: compute sigma and weights
   if m == 1
      sigma2      = fminbnd(@(x) abs( std( normcdf(-gk/x))/...
                                     mean(normcdf(-gk/x)) - tarCoV), 0, 10*gmu);
      sigmak(m+1) = sigma2;
      wk          = normcdf(-gk/sigmak(m+1));
   else
      sigma2      = fminbnd(@(x) abs( std( normcdf(-gk/x)./normcdf(-gk/sigmak(m)))/...
                                     mean(normcdf(-gk/x)./normcdf(-gk/sigmak(m))) - tarCoV ), 0, sigmak(m));
      sigmak(m+1) = sigma2;
      wk          = normcdf(-gk/sigmak(m+1))./normcdf(-gk/sigmak(m));
   end
   
   %===== Step 4: compute estimate of expected w
   Sk(m) = mean(wk);
   % exit algorithm if no convergence is achieved
   if Sk(m) == 0
      break;
   end
   wnork = wk./Sk(m)/nsamlev;          % compute normalized weights
   
   %===== Step 5: resample
   % seeds for chains
   ind = randsample(nsamlev,nchain,true,wnork);
   gk0 = gk(ind);
   uk0 = uk(ind,:);
   
   %===== Step 6: perform aCS
   % compute initial standard deviation
   switch opc
      case 'a'   % 1a. sigma = ones(n,1);
         sigmaf = 1;
      case 'b'   % 1b. sigma = sigma_hat; (sample standard deviations)
         muf    = mean(repmat(wnork,dim,1)'.*uk,1);
         sigmaf = zeros(1,dim);
         for k = 1:nsamlev
            sigmaf = sigmaf + wnork(k)*(uk(k,:)-muf).^2;
         end
      otherwise
         error('Choose a or b');
   end

   % compute parameter rho
   sigmafk = min(lambda*sigmaf, 1);
   rhok    = sqrt(1-sigmafk.^2);
   counta  = 0;
   count   = 0;
   
   % initialize chain acceptance rate
   alphak = zeros(nchain,1);
   gk     = [];                % delete previous samples
   uk     = [];                % delete previous samples
   for k = 1:nchain
      % set seed for chain
      u0 = uk0(k,:);
      g0 = gk0(k);
      for j = 1:lenchain+burn
         count = count+1;
         if j == burn+1
            count = count-burn;
         end
         % get candidate sample from conditional normal distribution
         ucand = normrnd(rhok*u0', sqrt(1-rhok^2))';
         
         % Evaluate limit-state function
         gcand = g(ucand);
         
         % compute acceptance probability
         alpha     = min(1,normcdf(-gcand/sigmak(m+1))/normcdf(-g0/sigmak(m+1)));
         alphak(k) = alphak(k)+alpha/(lenchain+burn);
         
         % check if sample is accepted
         uhelp = rand;
         if uhelp <= alpha
            uk(count,:) = ucand;
            gk(count) = gcand;
            u0  = ucand;
            g0  = gcand;
         else
            uk(count,:) = u0;
            gk(count) = g0;
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
            sigmafk = min(lambda*sigmaf,1);
            rhok    = sqrt(1-sigmafk.^2);
         end
      end
   end
   uk = uk(1:nsamlev,:);
   gk = gk(1:nsamlev);
   
   % Samples return - all / all by default
   if ~ismember(samples_return, [0 1])
       samplesU{m+1} = uk;
   end
   
   % compute mean acceptance rate of all chains in level m
   accrate(m) = mean(alphak);
   COV_Sl     = std( (gk < 0)./normcdf(-gk/sigmak(m+1)))/...
                mean((gk < 0)./normcdf(-gk/sigmak(m+1)));
   
   % it is not a problem if COV_Sl is NaN
   if ~isnan(COV_Sl)
      fprintf('\nCOV_Sl = %d\n', COV_Sl);
      fprintf('\t*aCS sigma = %g \t *aCS accrate = %g\n', sigmafk, accrate(m));
   end   
   if COV_Sl < tarCoV
      % Samples return - last
      if (samples_return == 1) 
          samplesU{1} = uk;
      end
      break;
   end
end

% Samples return - all by default message
if ~ismember(samples_return, [0 1 2])
    fprintf('\n-Invalid input for samples return, all samples are returned by default \n');
end

l_tot = m+1;

%% probability of failure
const = prod(Sk);
I_final = gk < 0;
W_final = 1./normcdf(-gk/sigmak(m+1));
Pr = const * mean(I_final .* W_final);

%% transform the samples to the physical/original space
samplesX = cell(length(samplesU),1);
if (samples_return ~= 0) 
    for i = 1:length(samplesU)
       samplesX{i} = u2x(samplesU{i});
    end
end

%% Sample Return Handling

% resample 1e4 failure samples with final weights W
weight_id = randsample(find(I_final),1e4,'true',W_final(I_final));
f_s = samplesX{end}(weight_id,:);


if samples_return == 0
    samplesU = cell(1,1);  % empty return samples U
    samplesX = cell(1,1);  % and X
end

% Convergence is not achieved message
if m == max_it
    fprintf('-Exit with no convergence at max iterations \n\n');
end

return;