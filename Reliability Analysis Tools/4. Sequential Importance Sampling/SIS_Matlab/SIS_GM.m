function [Pr, l_tot, samplesU, samplesX, k_fin, S_F1] = SIS_GM(N, p, g_fun, distr, k_init, burn, tarCOV, sensitivity_analysis, samples_return)
%% Sequential importance sampling using Gaussian mixture
%{
---------------------------------------------------------------------------
Created by:
Iason Papaioannou (iason.papaioannou@tum.de)
Matthias Willer
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2018-05
---------------------------------------------------------------------------
Current version: 2022-04
* Inclusion of sensitivity analysis.
---------------------------------------------------------------------------
Comments:
* The SIS method in combination with a Gaussian Mixture model can only be
  applied for low-dimensional problems, since its accuracy decreases
  dramatically in high dimensions.
---------------------------------------------------------------------------
Input:
* N                    : number of samples per level
* p                    : N/number of chains per level
* g_fun                : limit state function
* distr                : Nataf distribution object or
                         marginal distribution object of the input variables
* k_init               : initial number of Gaussians in the mixture model
* burn                 : burn-in period
* tarCOV               : target COV of weights
* sensitivity_analysis : implementation of sensitivity analysis: 1 - perform, 0 - not perform
* samples_return       : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr       : probability of failure
* l_tot    : total number of levels
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* k_fin    : final number of Gaussians in the mixture
* S_F1     : vector of first order Sobol' indices
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
   u2x = @(u) distr.U2X(u);          % from u to x
   
else   % use distribution information for the transformation (independence)
   % Here we are assuming that all the parameters have the same distribution !!!
   % Adjust accordingly otherwise or use an ERANataf object
   dim = length(distr);                    % number of random variables (dimension)
   u2x = @(u) distr(1).icdf(normcdf(u));   % from u to x   
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
gk      = zeros(1,nsamlev);       % space for evaluations of g
accrate = zeros(max_it,1);        % space for acceptance rate
sigmak  = zeros(max_it,1);        % space for the standard deviation
Sk      = ones(max_it,1);         % space for expected weights

samplesU  = cell(1,1);            % space for the samples in the standard space

%% SIS GM
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
                                      mean(normcdf(-gk/x)) - tarCOV), 0, 10*gmu);
      sigmak(m+1) = sigma2;
      wk          = normcdf(-gk/sigmak(m+1));
   else
      sigma2      = fminbnd(@(x) abs( std( normcdf(-gk/x)./normcdf(-gk/sigmak(m)))/...
                                      mean(normcdf(-gk/x)./normcdf(-gk/sigmak(m))) - tarCOV), 0, sigmak(m));
      sigmak(m+1) = sigma2;
      wk          = normcdf(-gk/sigmak(m+1))./normcdf(-gk/sigmak(m));
   end
   
   %===== Step 4: compute estimate of expected w
   Sk(m) = mean(wk);   
   % exit algorithm if no convergence is achieved
   if Sk(m) == 0
      break;
   end
   wnork        = wk./Sk(m)/nsamlev;          % compute normalized weights
   [mu, si, ww] = EMGM(uk',wnork',k_init);    % fit Gaussian Mixture
   
   %===== Step 5: resample
   % seeds for chains
   ind = randsample(nsamlev,nchain,true,wnork);
   gk0 = gk(ind);
   uk0 = uk(ind,:);
   
   %===== Step 6: perform M-H
   count  = 0;
   alphak = zeros(nchain,1);   % initialize chain acceptance rate
   gk     = [];                % delete previous samples
   uk     = [];                % delete previous samples
   for k = 1:nchain
      % set seed for chain
      u0 = uk0(k,:);
      g0 = gk0(k);
      
      for i = 1:lenchain+burn
         count = count+1;         
         if i == burn+1
            count = count-burn;
         end
         
         % get candidate sample from conditional normal distribution
         indw  = randsample(length(ww), 1, true,ww);
         ucand = mvnrnd(mu(:,indw), si(:,:,indw));
         %ucand = muf + (Acov*randn(dim,1))';
         
         % Evaluate limit-state function
         gcand = g(ucand);
         
         % compute acceptance probability
         pdfn = 0;
         pdfd = 0;
         for ii = 1:length(ww)
            pdfn = pdfn + ww(ii)*mvnpdf(u0',mu(:,ii),si(:,:,ii));
            pdfd = pdfd + ww(ii)*mvnpdf(ucand',mu(:,ii),si(:,:,ii));
         end         
         alpha     = min(1, (normcdf(-gcand/sigmak(m+1))*prod(normpdf(ucand))*pdfn)/...
                            (normcdf(-g0/sigmak(m+1))*prod(normpdf(u0))*pdfd));
         alphak(k) = alphak(k) + alpha/(lenchain+burn);
         
         % check if sample is accepted
         uhelp = rand;
         if uhelp <= alpha
            uk(count,:) = ucand;
            gk(count)   = gcand;
            u0  = ucand;
            g0  = gcand;            
         else
            uk(count,:) = u0;
            gk(count)   = g0;
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
      fprintf('\t *MH-GM accrate = %g\n', accrate(m));
   end   
   if COV_Sl < tarCOV
      % Samples return - last
      if (samples_return == 1) || (samples_return == 0 && sensitivity_analysis == 1)
          samplesU{1} = uk;
      end
      break;
   end   
end

% Samples return - all by default message
if ~ismember(samples_return, [0 1 2])
    fprintf('\n-Invalid input for samples return, all samples are returned by default \n');
end

% required steps
k_fin = length(ww); 
l_tot = m + 1;

%% calculation of the probability of failure
% accfin = accrate(m);
const = prod(Sk);
I_final = gk < 0;
W_final = 1./normcdf(-gk/sigmak(m+1));
Pr = const * mean(I_final .* W_final);

%% transform the samples to the physical/original space
samplesX = cell(length(samplesU),1);
if (samples_return ~= 0) || (samples_return == 0 && sensitivity_analysis == 1)
    for i = 1:length(samplesU)
       samplesX{i} = u2x(samplesU{i});
    end
end

%% sensitivity analysis
if sensitivity_analysis == 1
    % resample 1e4 failure samples with final weights W
    weight_id = randsample(find(I_final),1e4,'true',W_final(I_final));
    f_s = samplesX{end}(weight_id,:);
   
    [S_F1, exitflag, errormsg] = Sim_Sobol_indices(f_s, Pr, distr);
    if exitflag == 1
        fprintf("\n-First order indices: \n");
        disp(S_F1);
    else
        fprintf('\n-Sensitivity analysis could not be performed, because: \n')
        fprintf(errormsg);
    end
else 
    S_F1 = [];
end

if samples_return == 0
    samplesU = cell(1,1);  % empty return samples U
    samplesX = cell(1,1);  % and X
end

% Convergence is not achieved message
if m == max_it
    fprintf('-Exit with no convergence at max iterations \n\n');
end

return;
%%END