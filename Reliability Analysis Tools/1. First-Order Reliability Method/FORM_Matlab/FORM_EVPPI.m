function [EVPPI,exitflag, errormsg] = FORM_EVPPI(alpha,beta,pf,distr,c_R,c_F,normalization)
%% EVPPI CALCULATION FROM FORM INDICES
%{
---------------------------------------------------------------------------
Created by:
Daniel Koutas
Max Ehre
Ivan Olarte-Rodriguez

Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2023-06
---------------------------------------------------------------------------
Changelog 
---------------------------------------------------------------------------
Based on:
1."Decision-theoretic reliability sensitivity"
   Daniel Straub, Max Ehre, & Iason Papaioannou.
   Luyi Li, Iason Papaioannou & Daniel Straub.
   Reliability Engineering & System Safety (2022), 221, 108215.
---------------------------------------------------------------------------
Comment:
* The EVVPI computation uses the FORM indices as input. 
  WARNING: Only valid for independent marginal distributions
---------------------------------------------------------------------------
Input:
* alpha         : vector with the values of FORM indices
* beta          : reliability index
* pf            : probaiblity of failure
* distr         : ERANataf or ERARosen distribution object
* c_R           : cost of replacement (scalar)
* c_F           : cost of Failure
* normalization : normalization output of EVPPI
---------------------------------------------------------------------------
Output:
* EVPPI         : Vector with computation of EVPPI per each input variable
---------------------------------------------------------------------------
%}

%% PRECOMPUTATIONS

% Extract the dimension 
d = numel(distr.Marginals);

% Array to store the EVPPI values
crude_EVPPI = zeros(1,d);

% Preset the error message and exit flag
exitflag = 1;
errormsg = [];


if ~(norm(eye(d)-distr.Rho_X,2)<1e-05)
    % Raise warning the computations are not exact
    warning("\n \n The marginals are not independent \n" + ...
        "The results must be analyzed with care! \n \n ")
end


% Compute the threshold of the boundaries of the cost of repair vs the
% cost of failure
PF_thres = c_R/c_F;

if c_R > c_F
        EVPPI = repelem(NaN,d);
        exitflag = 0;
        errormsg = "The cost of replacement is greater than " + ...
                   "the cost of failure \n \n";
        return;
    else
        % Show the cost of replacement and cost of failure
        fprintf("\n-cost of replacement: %.2f \n", c_R);
        fprintf("-cost of failure: %.2f \n", c_F);
end


%% COMPUTATION OF CRUDE EVPPI
% WARNING! -> The procedure is meant only for independent marginal distributions

u_i_thres = zeros(1,d);

u_i_thres_fun = @(u) 1./u*(sqrt(1-u.^2).*norminv(PF_thres)+beta);

% Array to store the s_i
s_i = zeros(size(u_i_thres));

for ii = 1:d
    % Compute the threshold
    u_i_thres(ii) = u_i_thres_fun(alpha(ii));
    
    % Compute the s_i
    if (pf-PF_thres)*alpha(ii) ~=0
        s_i(ii) = sign((pf-PF_thres)*alpha(ii));
    else
        s_i(ii) = -1;
    end

    % Compute the EVPPI
    % Generate auxiliary mean array for bivariate standard normal cdf
    aux_mean = zeros(1,2);
    % Generate auxiliary sigma array for bivariate standard normal cdf
    aux_sigma = [1,-s_i(ii)*alpha(ii);-s_i(ii)*alpha(ii),1];
    crude_EVPPI(ii) = abs(c_F*mvncdf([-beta,s_i(ii)*u_i_thres(ii)],aux_mean,aux_sigma)-...
        c_R*normcdf(s_i(ii)*u_i_thres(ii),0,1));
end


%% OUTPUT
 % Modify the output depending on the output mode set by user
    switch normalization
        case "crude"
            EVPPI = crude_EVPPI;
        case "normalized"
            EVPPI = zeros(size(crude_EVPPI));
            for ii = 1:d
                EVPPI(ii) = crude_EVPPI(ii)/sum(crude_EVPPI,"all");
            end
        case "relative"
            % Compute EVPI
            if pf <= PF_thres
                EVPI = pf*(c_F-c_R);
            else
                EVPI = c_R*(1-pf);
            end
            EVPPI = crude_EVPPI./EVPI;
    end

end