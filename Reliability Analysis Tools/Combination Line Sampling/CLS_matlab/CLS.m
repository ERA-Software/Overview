function [Pf_LS, cv_LS, u_new, x_new, n_eval] = CLS(u_star, N, g_fun, distr)
%% Combination Line Sampling (CLS) function
%{
---------------------------------------------------------------------------
Created by:
Iason Papaioannou
Daniel Koutas
Engineering Risk Analysis Group   
Technische Universitat Muenchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Current version 2021-06
---------------------------------------------------------------------------
Based on:
1."Combination line sampling for structural reliability analysis"
   Iason Papaioannou & Daniel Straub.
   Structural Safety 88 (2021) 102025.
---------------------------------------------------------------------------
Comments:
* Here we focus on the case where the failure domain is concentrated in a 
  distinct region of the outcome space. Problems with multiple failure 
  modes often require accounting for several important directions.
* The potential of this method is limited to problems with low to moderate
  dimensions as it gets more difficult to achieve substantial improvements
  with an increasing sampling space dimension.
---------------------------------------------------------------------------
Input: 
* u_star: a point in U-space that defines the initial direction
* N: the  desired number of samples, 
* g_fun:  the limit-state function (LSF)
* distr:  the input distribution
---------------------------------------------------------------------------
Output:
* Pf_LS:  estimate of the failure probability
* cv_LS:  estimate of the coefficient of variation of the probability estimate
* u_new:  final selected point in U-space (estimate for the FORM design point)
* x_new:  same point in X-space 
* n_eval: the total number of LSF-evaluations 
%}

%% initial check if there exists a Nataf object

% transform to the standard Gaussian space
if any(strcmp('Marginals',fieldnames(distr)))   % use Nataf transform (dependence)
    n   = length(distr.Marginals);    % number of random variables (dimension)
    u2x = @(u) distr.U2X(u);          % from u to x
    
else   % use distribution information for the transformation (independence)
    % Here we are assuming that all the parameters have the same distribution !!!
    % Adjust accordingly otherwise
    n   = length(distr);                    % number of random variables (dimension)
    u2x = @(u) distr(1).icdf(normcdf(u));   % from u to x
end


%% Initialization

% LSF in standard space
G_LSF = @(u) g_fun(u2x(u));

% number of LSF evaluations
n_eval = 0;

% fixed number of LSF evaluations for the find_root_RS function in the 
% first run and subsequent runs in the for-loop
n_LSF_1 = 10;
n_LSF_subs = 5;

% select root finding function as either 'find_root' or 'find_root_RS'
rt_fcn = 'find_root';

if ~(strcmp(rt_fcn, 'find_root') || strcmp(rt_fcn, 'find_root_RS'))
    error("Select either 'find_root' or 'find_root_RS' as your root finding function") 
end


%% CLS method

% Initial direction
alpha = u_star/norm(u_star);

% Find root in initial direction
G_LSF1D = @(d) G_LSF(d*alpha);
if strcmp(rt_fcn, 'find_root')
    d0 = 1;
    [c0, n_eval1, exitflag] = find_root(G_LSF1D, d0);
    n_eval = n_eval + n_eval1;
elseif strcmp(rt_fcn, 'find_root_RS')
    d0 = 3;
    [c0,exitflag] = find_root_RS(G_LSF1D,d0,n_LSF_1);
    n_eval = n_eval + n_LSF_1;
end

% If no root was found, penalize with a high value (corresponding to ~0
% contribution)
if exitflag == 0 
    c0 = 10;
end

% Loop variables
l = 0;
k = 0;

for i = 1:N
    
    l = l+1;
    
    % Generate random sample vector
    u = randn(1,n);
    % Orthogonalization to search direction
    uper = u - (u*alpha')*alpha;
    
    % Solve root finding problem
    G_LSF1D = @(d) G_LSF(uper+d*alpha);
    if strcmp(rt_fcn, 'find_root')
        [d1,n_eval1,exitflag]=find_root(G_LSF1D,c0);
        n_eval = n_eval + n_eval1;
    elseif strcmp(rt_fcn, 'find_root_RS')
        [d1,exitflag] = find_root_RS(G_LSF1D,c0,n_LSF_subs);
        n_eval = n_eval + n_LSF_subs;
    end
    
    % Save found distance or penalize with a high value
    if exitflag == 1
        d(i) = d1;
    else
        d(i) = 10;
    end
     
    pfi(i) = normcdf(-abs(d(i)));
    
    % Calculate the distance from the origin to the intersection of the 
    % line parallel to alpha with the failure surface
    c1 = norm(uper+d(i)*alpha);
    
    % If the distance c1 is shorter than the previously found one, do 
    % corresponding updates 
    if c1 < c0
        k = k+1;
        % Line Sampling (LS) estimator for direction alpha
        pfk(k) = mean(pfi(i-l+1:i));
        % Variance estimator 
        var_LSk(k) = 1/l*(1/l*sum(pfi(i-l+1:i).^2)-pfk(k)^2);
        
        % Heuristic weights
        wk(k) = l*normcdf(-abs(c0));
        
        if var_LSk(k)==0 || l<=2
            var_LSk(k)= pfk(k)^2;
        end
        
        % Update importance direction and values
        alpha = 1/c1*(uper+d(i)*alpha);
        c0 = c1;
        
        l = 0;
    end
    
end



if l > 0
    k = k+1;
    % LS estimator for last direction 
    pfk(k) = mean(pfi(i-l+1:i));
    % Variance estimate
    var_LSk(k) = 1/l*(1/l*sum(pfi(i-l+1:i).^2)-pfk(k)^2);
    
    % Heuristic weights
    wk(k) = l*normcdf(-abs(c0));
    
    if var_LSk(k)==0 || l<=2
        var_LSk(k)= pfk(k)^2;
    end
    
end


% Normalization
wk_norm = wk/sum(wk);

% Variance estimate is the weighted sum of the individual variance 
% estimates
var_LS  = sum(wk_norm.^2.*var_LSk);

% Failure probability estimate is the weighted sum of the individual
% failure estimates
Pf_LS   = sum(pfk.*wk_norm);

% Coefficient of variation estimate
cv_LS = sqrt(var_LS)/Pf_LS;

% Update estimate for the FORM design point in U-space and X-space
u_new = c0*alpha;
x_new = u2x(u_new);

return




