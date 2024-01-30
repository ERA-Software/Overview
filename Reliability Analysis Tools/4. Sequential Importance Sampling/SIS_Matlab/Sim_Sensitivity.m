function [S_F1, S_EVPPI] = Sim_Sensitivity(f_s_iid, pf, distr, comp_Sobol,...
    comp_EVPPI, c_R, c_F, normalization, varargin)
%% Compute the Sensitivity Indices and EVPPI from Samples 
%{
---------------------------------------------------------------------------
Created by:
Daniel Koutas
Ivan Olarte-Rodriguez
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2022-04
2023-07 (Ivan Olarte-Rodriguez): 
Splitted the Sensitivity Computations from main functions
---------------------------------------------------------------------------
Based on:
1."Global reliability sensitivity estimation based on failure samples"
   Luyi Li, Iason Papaioannou & Daniel Straub.
   Structural Safety 81 (2019) 101871.
2."Kernel Estimator and Bandwidth Selection for Density and its
   Derivatives"
   Arsalane Chouaib Guidoum.
   Department of Probabilities and Statistics, University of Science and 
   Technology, Houari Boumediene, Algeria (2015)
}
---------------------------------------------------------------------------
Comments: 
* The upper bound of fminbnd is set to a multiple of the maximum distance 
  between the failure samples, because Inf is not handled well.
* Significantly dominates computation time at higher number of samples
* User can trigger plot of posterior kernel density estimations as well as
  maximum likelihood cross validation dependent on the bandwidth (optimal
  bandwidth marked as star)
---------------------------------------------------------------------------
Input:
- Required
* f_s_iid       : Independent and identically distributed failure samples 
* pf            : estimated failure probability
* distr         : ERADist or ERANataf object containing the infos about 
                  the random variables.
* comp_Sobol    : boolean variable to indicate the computation of the
                  Sobol Indices.
* comp_EVPPI    : boolean variable to indicate the computation of EVPPI
                  indices

- Optional
* c_R : Cost of replacement
* c_F : Cost of Failure
* normalization: Normalization options for EVPPI calculation
---------------------------------------------------------------------------
Output:
* S_F1      : vector of first order sensitivity indices
* S_EVPPI   : vector of EVPPI measures for each variable
---------------------------------------------------------------------------
%}

exit_msg = "";
%% Generate Input Parser to evaluate the inputs to this function
p = inputParser;

% Validation Handles
validNum = @(x) isnumeric(x);
validBool = @(x) islogical(x) && isscalar(x);
validScalarProb = @(x) isnumeric(x) && isscalar(x) && (x >= 0) && (x <= 1);
validScalarNum =  @(x) isnumeric(x) && isscalar(x) && (x >= 0);
validERAObj = @(x) (strcmp(class(x),"ERANataf") || strcmp(class(x),"ERARosen") || ...
                    strcmp(class(x(1)),"ERADist"));

% Default values for the cost of failure and cost of replacements
default_c_R = 10^5;
default_c_F = 10^8;

% Set the output type
expectedEVPPIOutputType = {'crude','normalized','relative'};
defaultEVPPIOutputType  = 'normalized';

% Set the required parameters to be received by the function
if ~validNum(f_s_iid)
    exit_msg = append(exit_msg, 'f_s_iid is not numeric! ');
end
if isempty(f_s_iid)
    exit_msg = append(exit_msg, "failure samples array f_s_iid is empty! Check e.g. if samples_return > 0. ");
end
if ~validScalarProb(pf)
    exit_msg = append(exit_msg, "pf must be a scalar and in range [0,1]! ");
end
if ~validERAObj(distr)
    exit_msg = append(exit_msg, "distribution object not ERADist, ERANataf or ERARosen instance! ");
end
if ~validBool(comp_Sobol)
    exit_msg = append(exit_msg, "comp_Sobol has be boolean! ");
end
if ~validBool(comp_EVPPI)
    exit_msg = append(exit_msg, "comp_EVPPI has to be boolean! ");
end

if exit_msg ~= ""
    S_F1 = [];
    S_EVPPI = [];
    fprintf("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    fprintf("Sensitivity computation aborted due to:\n %s", exit_msg)
    fprintf("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    return
end

addRequired(p,'f_s',validNum);
addRequired(p,'pf',validScalarProb);
addRequired(p,'distr',validERAObj);
addRequired(p,'comp_Sobol',validBool);
addRequired(p,'comp_EVPPI',validBool);


% Set the optional parameters
addOptional(p,'c_R',default_c_R,validScalarNum);
addOptional(p,'c_F',default_c_F,validScalarNum);
addOptional(p,"EVPPI_Output",defaultEVPPIOutputType,...
                 @(x) any(validatestring(x,expectedEVPPIOutputType,1)));

% Set parameters

%% Validate the optional parameters
if ~exist('c_R','var') || isempty(c_R)
    c_R = default_c_R;
end

if ~exist("c_F","var") || isempty(c_F)
    c_F = default_c_F;
end

if ~exist("normalization","var") || isempty(normalization)
    normalization = defaultEVPPIOutputType ;
end

%% Set the parsing object
parse(p, f_s_iid, pf, distr, comp_Sobol, comp_EVPPI, c_R, c_F, normalization, varargin{:});

%% Compute the Sensitivity Indices 

S_F1 = [];
w_opt = [];
if p.Results.comp_Sobol
    fprintf("\n\nComputing Sobol Sensitivity Indices \n");
    [S_F1, exitflag, errormsg,w_opt] = Sim_Sobol_indices(p.Results.f_s, ...
                                                         p.Results.pf, ...
                                                         p.Results.distr);

    if exitflag == 1
        fprintf("-First order indices: \n");
        disp(S_F1);
    else
        fprintf('\n-Sensitivity analysis could not be performed, because: \n')
        fprintf(errormsg);
    end

end


%% Compute the EVPPI based on input parameters

S_EVPPI = [];

if strcmp(class(p.Results.distr(1)),"ERADist")
    marginal_Dists = p.Results.distr;
else
    marginal_Dists = p.Results.distr.Marginals;
end
% Compute only if the outputs are different than the default
if p.Results.comp_EVPPI
    fprintf("\n\nComputing EVPPI Sensitivity Indices \n");

    if ~isempty(w_opt)
        [S_EVPPI, exitflag, errormsg] = Sim_EVPPI(p.Results.f_s, p.Results.pf,...
                                                   p.Results.c_R, p.Results.c_F, ...
                                                  marginal_Dists, p.Results.EVPPI_Output, ...
                                                  "optimal_bandwidth_weights", w_opt); 
    else
        [S_EVPPI, exitflag, errormsg] = Sim_EVPPI(p.Results.f_s, p.Results.pf, ...
                                                  p.Results.c_R, p.Results.c_F, ...
                                                  marginal_Dists, p.Results.EVPPI_Output); 
    end
    
    if exitflag == 1
        % Show the computation of EVPPI given the parameters
        fprintf("-EVPPI normalized as: %s \n",p.Results.EVPPI_Output)
        fprintf("\n-EVPPI indices: \n")
        disp(S_EVPPI);
    else
        fprintf('\n-Sensitivity analysis could not be performed, because: \n')
        fprintf(errormsg);
    end
end




end