function [Pr, l_tot, samplesU, samplesX, k_fin,W_final,f_s_iid] = ...
    SIS_vMFNM(N, p, g_fun, distr, k_init, burn, tarCOV,samples_return)
%% Sequential importance sampling using vMFN mixture model
%{
---------------------------------------------------------------------------
Created by:
Max Ehre
Iason Papaioannou
Daniel Straub
Ivan Olarte Rodriguez
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2023-01
---------------------------------------------------------------------------
Current version: 2023-07
* Modification in the Sensitivity Analysis Output
---------------------------------------------------------------------------
Comments:
* SIS-method with the independent MH-sampler using the von Mises-Fisher 
Nakagami mixture model for the proposal distribution
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
* samples_return       : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr       : probability of failure
* l_tot    : total number of levels
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* k_fin    : final number of Gaussians in the mixture
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

% initialize output arrays
sigma_t   = zeros(max_it,1);     % squared difference between COV and target COV
samplesU  = cell(1,1);           % space for the samples in the standard space

% Properties of SIS
nsamlev  = N;                % number of samples
nchain   = nsamlev*p;       % number Markov chains
lenchain = nsamlev/nchain;   % number of samples per Markov chain

% initialize samples
gk      = zeros(1,nsamlev);       % space for evaluations of g
accrate = zeros(max_it,1);        % space for acceptance rate
sigmak  = zeros(max_it,1);        % space for the standard deviation
Sk      = ones(max_it,1);         % space for expected weights

%% SIS GM
%===== Step 1: Perform the first Monte Carlo simulation
uk = randn(nsamlev,dim);    % initial samples
for k = 1:nsamlev
   gk(k) = g(uk(k,:));
end

% save samples
if ~ismember(samples_return, [0 1])
    samplesU{m+1} = uk;
end

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
   [mu, kappa, mshape, omega, ww] = EMvMFNM(uk', wnork', k_init); % % fit vMFN Mixture

    % remove unnecessary components
    if min(ww) <= 0.01
        ind    = find(ww > 0.01);
        mu     = mu(:,ind);
        kappa  = kappa(ind);
        mshape = mshape(ind);
        omega  = omega(ind);
        ww     = ww(ind);
    end

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

         % get candidate sample from conditional vMFN distribution
         ucand = vMFNM_sample(mu',kappa,omega,mshape,ww,1);
         
         % Evaluate limit-state function
         gcand = g(ucand);
         
         % compute acceptance probability
         W_log_cand = likelihood_ratio_log(ucand,mu',kappa,omega,mshape,ww);
         W_log_0 = likelihood_ratio_log(u0,mu',kappa,omega,mshape,ww);

         alpha     = min(1,exp(W_log_cand - W_log_0)*normcdf(-gcand/sigmak(m+1))/(normcdf(-g0/sigmak(m+1))));

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
      if (samples_return == 1) 
          samplesU{1} = uk;
      end
      break;
   end   

   % Assigning updated parameters
   mu     = mu';
   k_init = length(ww);

end

% Samples return - all by default message
if ~ismember(samples_return, [0 1 2])
    fprintf('\n-Invalid input for samples return, all samples are returned by default \n');
end

% required steps
k_fin = length(ww); 
l_tot = length(samplesU);

%% calculation of the probability of failure
% accfin = accrate(m);
const = prod(Sk);
I_final = gk < 0;
W_final = 1./normcdf(-gk/sigmak(m+1));
Pr = const * mean(I_final .* W_final);

%% transform the samples to the physical/original space
samplesX = cell(length(samplesU),1);
f_s_iid = [];
if (samples_return ~= 0) 
    for i = 1:length(samplesU)
       samplesX{i} = u2x(samplesU{i});
    end

    % resample indices according to sample weights, where only failure
    % samples receive a nonzero weight
    weight_id = randsample(find(I_final),1e4,'true',W_final(I_final));
    f_s_iid = samplesX{end}(weight_id,:);
end

return;


%===========================================================================
%===========================NESTED FUNCTIONS================================
%===========================================================================
function p = approx_normCDF(x)

% Returns an approximation for the standard normal CDF based on a
% polynomial fit of degree 9

erfun=zeros(size(x));

idpos=x>0;
idneg=x<0;

t=(1+0.5*abs(x/sqrt(2))).^-1;

tau=t.*exp(-(x/sqrt(2)).^2-1.26551223+1.0000236*t+0.37409196*t.^2+0.09678418*t.^3-0.18628806*t.^4+0.27886807*t.^5-1.13520398*t.^6+1.48851587*t.^7-0.82215223*t.^8+0.17087277*t.^9);
erfun(idpos)=1-tau(idpos);
erfun(idneg)=tau(idneg)-1;

p=0.5*(1+erfun);

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
return;

%===========================================================================
function W_log = likelihood_ratio_log(X,mu,kappa,omega,m,Pi)

k       = length(Pi);
[N,dim] = size(X);
R       = sqrt(sum(X.^2,2));
if k == 1
    % log pdf of vMF distribution
    logpdf_vMF = vMF_logpdf((bsxfun(@times,X,1./R))',mu',kappa)';
    % log pdf of Nakagami distribution
    logpdf_N = nakagami_logpdf(R,m,omega);
    % log pdf of weighted combined distribution
    h_log = logpdf_vMF+logpdf_N;
else
    logpdf_vMF = zeros(N,k);
    logpdf_N   = zeros(N,k);
    h_log      = zeros(N,k);
    
    % log pdf of distributions in the mixture
    for i = 1:k
        % log pdf of vMF distribution
        logpdf_vMF(:,i) = vMF_logpdf((bsxfun(@times,X,1./R))',mu(i,:)',kappa(i))';
        % log pdf of Nakagami distribution
        logpdf_N(:,i) = nakagami_logpdf(R,m(i),omega(i));
        % log pdf of weighted combined distribution
        h_log(:,i) = logpdf_vMF(:,i)+logpdf_N(:,i)+log(Pi(i));
    end
    
    % mixture log pdf
    h_log = logsumexp(h_log,2);
end

% unit hypersphere uniform log pdf
A   = log(dim)+log(pi^(dim/2))-gammaln(dim/2+1);
f_u = -A;

% chi log pdf
f_chi = log(2)*(1-dim/2)+log(R)*(dim-1)-0.5*R.^2-gammaln(dim/2);

% logpdf of the standard distribution (uniform combined with chi distribution)
f_log = f_u + f_chi;
W_log = f_log - h_log;
return;

%===========================================================================
function X = vMFNM_sample(mu,kappa,omega,m,Pi,N)
% Returns samples from the von Mises-Fisher-Nakagami mixture

[k,dim] = size(mu);
if k == 1
    % sampling the radius
    %     pd=makedist('Nakagami','mu',m,'omega',omega);
    %     R=pd.random(N,1);
    R      = sqrt(gamrnd(m,omega./m,N,1));
    X_norm = vsamp(mu',kappa,N);    % sampling on unit hypersphere
    
else
    % Determine number of samples from each distribution
    %    z = sum(dummyvar(randsample(k,N,true,Pi)));
    %    k = length(z);
    ind = randsample(size(mu,1),N,true,Pi);
    z = histcounts(ind,[(1:size(mu,1)) size(mu,1)+1]);
    % Generation of samples
    R      = zeros(N,1);
    R_last = 0;
    X_norm = zeros(N,dim);
    X_last = 0;
    
    for i = 1:k
        % sampling the radius
        R(R_last+1:R_last+z(i)) = sqrt(gamrnd(m(i),omega(i)./m(i),z(i),1));
        R_last                  = R_last + z(i);
        
        % sampling on unit hypersphere
        X_norm(X_last+1:X_last+z(i),:) = vsamp(mu(i,:)',kappa(i),z(i));
        X_last                         = X_last+z(i);
        clear pd;
    end
end

% assign sample vector
X = bsxfun(@times,R,X_norm);
return;


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
return;


%===========================================================================
function y = vMF_logpdf(X,mu,kappa)
% Returns the von Mises-Fisher mixture log pdf on the unit hypersphere

d = size(X,1);
n = size(X,2);
if kappa == 0
    A = log(d) + log(pi^(d/2)) - gammaln(d/2+1);
    y = -A*ones(1,n);
elseif kappa > 0
    c = (d/2-1)*log(kappa)-(d/2)*log(2*pi)-logbesseli(d/2-1,kappa);
    q = bsxfun(@times,mu,kappa)'*X;
    y = bsxfun(@plus,q,c');
else
    error('kappa must not be negative');
end
return;


%===========================================================================
function y = nakagami_logpdf(X,m,om)

y = log(2)+m*(log(m)-log(om)-X.^2./om)+log(X).*(2*m-1)-gammaln(m);
return;


%===========================================================================
function h_log = vMFNM_logpdf(X,mu,kappa,omega,m,Pi)

k       = length(Pi);
[N,dim] = size(X);
R       = sqrt(sum(X.^2,2));

if k == 1
    % log pdf of vMF distribution
    logpdf_vMF = vMF_logpdf((bsxfun(@times,X,1./R))',mu',kappa)';
    % log pdf of Nakagami distribution
    logpdf_N = nakagami_logpdf(R,m,omega);
    % log pdf of weighted combined distribution
    h_log = logpdf_vMF+logpdf_N;
else
    logpdf_vMF = zeros(N,k);
    logpdf_N   = zeros(N,k);
    h_log      = zeros(N,k);
    
    % log pdf of distributions in the mixture
    for i = 1:k
        % log pdf of vMF distribution
        logpdf_vMF(:,i) = vMF_logpdf((bsxfun(@times,X,1./R))',mu(i,:)',kappa(i))';
        % log pdf of Nakagami distribution
        logpdf_N(:,i) = nakagami_logpdf(R,m(i),omega(i));
        % log pdf of weighted combined distribution
        h_log(:,i) = logpdf_vMF(:,i)+logpdf_N(:,i)+log(Pi(i));
    end
    
    % mixture log pdf
    h_log = logsumexp(h_log,2);
end

return;



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
return;


%===========================================================================
function logb = logbesseli(nu,x)
% log of the Bessel function, extended for large nu and x approximation
% from Eq. 9.7.7 of Abramowitz and Stegun
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
%%END