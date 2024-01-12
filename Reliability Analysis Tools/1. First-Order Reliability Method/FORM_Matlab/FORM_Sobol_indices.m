function [S_F1, S_F1_T, exitflag, errormsg] = FORM_Sobol_indices(alpha, beta, Pf)
%% Compute Sobol' and total-effect indices from FORM parameters
%{
---------------------------------------------------------------------------
Created by:
Daniel Koutas
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2021-06
---------------------------------------------------------------------------
Based on:
1."Variance-based reliability sensitivity analysis and the FORM a-factors"
   Iason Papaioannou & Daniel Straub.
   Reliability Engineering and System Safety 210 (2021) 107496.
---------------------------------------------------------------------------
Comment: 
* The results apply directly to the first order and total-effect indices 
  with respect to the original variables X for the case where X consists of 
  statistically independent components.
---------------------------------------------------------------------------
Input:
* alpha : normalized negative gradient vector at design point
* beta  : reliability index
* Pf    : probability of failure
---------------------------------------------------------------------------
Output:
* S_F1      : vector of first order indices
* S_F1_T    : vector of total-effect indices
* exitflag  : flag whether method was successful or not
* errormsg  : error message describing what went wrong
---------------------------------------------------------------------------
%}

%% Initial check for valid inputs
if isempty(alpha) || any(~isfinite(alpha))
    S_F1 = []; 
    S_F1_T = []; 
    exitflag = 0;
    errormsg = 'Invalid function input: alpha either infinite, NaN or empty.';
    return;
    
elseif isempty(beta) || any(~isfinite(beta))
    S_F1 = []; 
    S_F1_T = []; 
    exitflag = 0;
    errormsg = 'Invalid function input: beta either infinite, NaN or empty.';
    return;

elseif isempty(Pf) || any(~isfinite(Pf))
    S_F1 = []; 
    S_F1_T = []; 
    exitflag = 0;
    errormsg = 'Invalid function input: Pf either infinite, NaN or empty.';
    return;
    
elseif (Pf <0) || (Pf>1)
    S_F1 = []; 
    S_F1_T = []; 
    exitflag = 0;
    errormsg = 'Invalid function input: Pf must be between 0 and 1.';
    return;
end

% function to integrate
fun = @(r) 1/(2*pi) * 1./sqrt(1-r.^2) .* exp(-(beta.^2)./(1+r));

% Compute first order and total-effect indices for each dimension
for k=1:length(alpha)
    S_F1(k)   = 1/(Pf*(1-Pf)) * integral(fun, 0, alpha(k)^2);
    S_F1_T(k) = 1/(Pf*(1-Pf)) * integral(fun, 1-alpha(k)^2, 1);
end

% check if computed indices are valid
if ~isreal(S_F1) || ~isreal(S_F1_T)
    errormsg = 'Integration was not succesful, at least one index is complex.';
    exitflag = 0;
elseif any(S_F1 > 1) || any(S_F1_T > 1)
    errormsg = 'Integration was not succesful, at least one index is greater than 1.';
    exitflag = 0;
elseif any(S_F1 < 0) || any(S_F1_T < 0)
    errormsg = 'Integration was not succesful, at least one index is smaller than 0.';
    exitflag = 0;
elseif any(~isfinite(S_F1)) || any(~isfinite(S_F1_T))
    errormsg = 'Integration was not succesful, at least one index is NaN.';
    exitflag = 0;
else 
    errormsg = [];
    exitflag = 1;
end
    

return