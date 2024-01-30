function [S_F1,S_F1_T,S_EVPPI] = FORM_Sensitivity(pf, distr, beta, alpha, ...
    comp_Sobol, comp_EVPPI, c_R, c_F, normalization)
%% Compute the Sobol Indices and EVPPI from Samples 
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
* Calls to FORM_Sobol are splitted from FORM main functions
---------------------------------------------------------------------------
Input:
- Required
* pf      : estimated failure probability
* distr   : ERANataf or ERARosen object containing the infos about 
            the random variables
* beta    : reliability index
* alpha   : vector with the values of FORM indices
* comp_Sobol    : boolean variable to indicate the computation of the
                  sensitivity metrics based on Sobol Indices.
* comp_EVPPI    : boolean variable to indicate the computation of EVPPI
                  indices

- Optional
* c_R           : Cost of replacement
* c_F           : Cost of Failure
* normalization : Normalization options for EVPPI calculation
---------------------------------------------------------------------------
Output:
* S_F1      : vector of first order sensitivity indices
* S_F1_T    : vector of total-effect indices
* S_EVPPI   : vector of EVPPI measures for each variable
---------------------------------------------------------------------------
%}


exit_msg = "";
%% Generate Input Parser to evaluate the inputs to this function
p = inputParser;

% Validation Handles
validNum = @(x) isnumeric(x);
validScalarProb = @(x) isnumeric(x) && isscalar(x) && (x >= 0) && (x <= 1);
validScalarNum =  @(x) isnumeric(x) && isscalar(x);
validBool = @(x) islogical(x) && isscalar(x);
validERAObj = @(x) (strcmp(class(x),"ERANataf") || strcmp(class(x),"ERARosen"));

% Set the required parameters to be received by the function
if ~validScalarProb(pf)
    exit_msg = append(exit_msg, "pf must be a scalar and in range [0,1]! ");
end
if ~validERAObj(distr)
    exit_msg = append(exit_msg, "distribution object not ERADist, ERANataf or ERARosen instance! ");
end
if ~validScalarNum(beta)
    exit_msg = append(exit_msg, "beta has to be a scalar! ");
end
if ~validNum(alpha)
    exit_msg = append(exit_msg, "alpha has to be numeric! ");
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


% Default values for the cost of failure and cost of replacements
default_c_R = 10^5;
default_c_F = 10^8;

% Set the output type
expectedEVPPIOutputType = {'crude','normalized','relative'};
defaultEVPPIOutputType  = 'normalized';

% Set the required parameters to be received by the function
addRequired(p,'pf',validScalarProb);
addRequired(p,'distr',validERAObj);
addRequired(p,'beta',validScalarNum);
addRequired(p,'alpha',validNum);
addRequired(p,'comp_Sobol',validBool);
addRequired(p,'comp_EVPPI',validBool);

% Set the optional parameters
addOptional(p,"c_R",default_c_R,validScalarNum);
addOptional(p,"c_F",default_c_F,validScalarNum);

addOptional(p,'EVPPI_Output',defaultEVPPIOutputType,...
                 @(x) any(validatestring(x,expectedEVPPIOutputType,1)));

%% Validate the optional parameters
if ~exist("c_R","var") || isempty(c_R)
    c_R = default_c_R;
end

if ~exist("c_F","var") || isempty(c_F)
    c_F = default_c_F;
end

if ~exist("normalization","var") || isempty(normalization)
    normalization = defaultEVPPIOutputType ;
end

%% Set the parsing object
parse(p,pf,distr,beta,alpha,comp_Sobol,comp_EVPPI,c_R,c_F,normalization);

%% Compute the Sobol Indices

S_F1 = []; S_F1_T = [];
if p.Results.comp_Sobol
    fprintf("\n\nComputing Sobol Sensitivity Indices  \n");
    [S_F1, S_F1_T, exitflag, errormsg] = FORM_Sobol_indices(p.Results.alpha, p.Results.beta, p.Results.pf);
    
    % Print error messages and flags
    if any(strcmp('Marginals',fieldnames(distr)))
            if ~isequal(distr.Rho_X, eye(length(distr.Marginals)))
                fprintf("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING: !!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                fprintf("Results of sensitivity analysis do not apply for dependent inputs.")
                fprintf("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
            end
    end
        
    if exitflag == 1
        fprintf("-First order indices: \n");
        disp(S_F1);
        fprintf("-Total-effect indices: \n");
        disp(S_F1_T);
    else
        fprintf('-Sensitivity analysis could not be performed, because: \n')
        fprintf(errormsg);
    end

end


%% Compute the EVPPI based on input parameters

% Start the output as an empty set
S_EVPPI = [];

% Compute only if the outputs are different than the default
if p.Results.comp_EVPPI
    fprintf("\n\nComputing EVPPI Sensitivity Indices \n");
    [S_EVPPI,exitflag, errormsg] = FORM_EVPPI(p.Results.alpha, p.Results.beta, ...
        p.Results.pf, p.Results.distr, p.Results.c_R, p.Results.c_F, p.Results.EVPPI_Output);

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