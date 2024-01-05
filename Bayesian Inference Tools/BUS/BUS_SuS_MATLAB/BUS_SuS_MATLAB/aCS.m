function [u_jp1,geval,new_lambda,sigma,accrate] = aCS(N, lam, h, u_j, h_LSF)
%% Adaptive conditional sampling algorithm
%{
---------------------------------------------------------------------------
Created by:
Fong-Lin Wu
Felipe Uribe
Iason Papaioannou (iason.papaioannou@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonios.kamariotis@tum.de)
---------------------------------------------------------------------------
Current version 2020-10
* Adaptation to new ERANataf class
---------------------------------------------------------------------------
Version 2019-06
* Update LSF with log_likelihood, modify average acceptance for Octave and clean up code
Version 2018-05
* Fixing bug in the adaptation (line 115)
Version 2017-07
* Fixing allocation bugs and adaptation at the last SuS step
Version 2017-04
* Fixing bug in the calculation of sigma_0 (line 69)
---------------------------------------------------------------------------
Input:
* N          : number of samples to be generated
* lam        : scaling parameter lambda
* h          : actual intermediate level
* u_j        : seeds used to generate the new samples
* h_LSF      : limit state function in the standard space
---------------------------------------------------------------------------
Output:
* u_jp1      : next level samples
* geval      : limit state function evaluations of the new samples
* new_lambda : next scaling parameter lambda
* sigma      : standard deviation of the proposal
* accrate    : acceptance rate
---------------------------------------------------------------------------
NOTES
* The way the initial standard deviation is computed can be changed in line 67.
By default we use option 'a' (it is equal to one).
In option 'b', it is computed from the seeds.
* The final accrate might differ from the target 0.44 when no adaptation is
performed. Since at the last level almost all seeds are already in the failure
domain, only a few are selected to complete the required N samples. The final
accrate is computed for those few samples
---------------------------------------------------------------------------
Based on:
1."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
---------------------------------------------------------------------------
%}

%% 0. initialize variables
[Ns, dim] = size(u_j);         % dimension parameter space and number of seeds
Na        = ceil(100*Ns/N);    % number of chains after which the proposal is adapted

% number of samples per chain
Nchain = ones(Ns,1)*floor(N/Ns);
Nchain(1:mod(N,Ns)) = Nchain(1:mod(N,Ns))+1;

% initialization
u_jp1  = zeros(N,dim);               % generated samples
geval  = zeros(1,N);                 % store lsf evaluations
acc    = zeros(1,N);                 % store acceptance
mu_acc = zeros(1,floor(Ns/Na)+1);    % store mean acceptance until adaptation
hat_a  = zeros(1,floor(Ns/Na));      % average acceptance rate of the chains
lambda = zeros(1,floor(Ns/Na)+1);    % scaling parameter \in (0,1)

%% 1. compute the standard deviation
opc = 'a';
switch opc
    case 'a'   % 1a. sigma = ones(dim,1);
        sigma_0 = ones(dim,1);
        
    case 'b'   % 1b. sigma = sigma_hat; (sample standard deviations)
        sigma_0 = std(u_j,0,1)';
        
    otherwise
        error('Choose a or b');
end

%% 2. iteration
star_a    = 0.44;         % optimal acceptance rate
lambda(1) = lam;   % initial scaling parameter \in (0,1)

% a. compute correlation parameter
i         = 1;                                  % index for adaptation of lambda
sigma     = min(lambda(i)*sigma_0,ones(dim,1));   % Ref. 1 Eq. 23
rho       = sqrt(1-sigma.^2);                   % Ref. 1 Eq. 24
mu_acc(i) = 0;

% b. apply conditional sampling
for k = 1:Ns
    idx          = sum(Nchain(1:k-1))+1;  %((k-1)/pa+1);
    u_jp1(idx,:) = u_j(k,:);              % pick a seed
    geval(idx)   = h_LSF(u_jp1(idx,:));   % store the lsf evaluation
    %
    for t = 1:Nchain(k)-1
        % generate candidate sample
        v = normrnd(rho.*(u_jp1(idx+t-1,:)'), sigma);
        
        % accept or reject sample
        Ge = h_LSF(v);
        if Ge <= h
            u_jp1(idx+t,:) = v;    % accept the candidate in failure region
            geval(idx+t)   = Ge;   % store the lsf evaluation
            acc(idx+t)     = 1;    % note the acceptance
        else
            u_jp1(idx+t,:) = u_jp1(idx+t-1,:);   % reject the candidate and use the same state
            geval(idx+t)   = geval(idx+t-1);     % store the lsf evaluation
            acc(idx+t)     = 0;                  % note the rejection
        end
    end
    % average of the accepted samples for each seed
    macc = acc(idx+1:idx+Nchain(k)-1);
    if isempty(macc)
        mu_acc_chain = eps;
    else
        mu_acc_chain = mean(macc, 2);
    end
    mu_acc(i) = mu_acc(i) + min([1, mu_acc_chain]);
    
    if mod(k,Na) == 0
        if (Nchain(k) > 1)   % only if the length of the chain is larger than 1
            % c. evaluate average acceptance rate
            hat_a(i) = mu_acc(i)/Na;   % Ref. 1 Eq. 25
            
            % d. compute new scaling parameter
            zeta        = 1/sqrt(i);   % ensures that the variation of lambda(i) vanishes
            lambda(i+1) = exp(log(lambda(i)) + zeta*(hat_a(i)-star_a));  % Ref. 1 Eq. 26
            
            % update parameters
            sigma = min(lambda(i+1)*sigma_0,ones(dim,1));   % Ref. 1 Eq. 23
            rho   = sqrt(1-sigma.^2);                     % Ref. 1 Eq. 24
            
            % update counter
            i = i+1;
        end
    end
end

% next level lambda
new_lambda = lambda(i);

% compute mean acceptance rate of all chains
accrate = sum(acc)/(N-Ns);

return;
%%END