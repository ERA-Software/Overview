function [EVPPI, exitflag, errormsg] = Sim_EVPPI(f_s, pf, c_R, c_F, X_marg, normalization, varargin)
    % EVPPI CALCULATION FROM Samples
    %{
    ---------------------------------------------------------------------------
    Created by:
    Ivan Olarte-Rodriguez
    Engineering Risk Analysis Group   
    Technische Universitat Munchen
    www.bgu.tum.de/era
    -----------------------------------------------------------------------
    First version: 2023-06
    -----------------------------------------------------------------------
    Based on:
    1."Decision-theoretic reliability sensitivity"
       Daniel Straub, Max Ehre, & Iason Papaioannou.
       Luyi Li, Iason Papaioannou & Daniel Straub.
       Reliability Engineering & System Safety (2022), 221, 108215.
    -----------------------------------------------------------------------
    Comments:
    * for MLCV calculation: see Sim_Sobol_indices code
    -----------------------------------------------------------------------

    Inputs:
    - f_s:              failure sample containing n points (d x n matrix)
	- pf:               unconditional system failure probability
    - c_R:              cost of repair
    - c_F:              cost of failure
	- X_marg:           cell array of d marginal input PDFs (ERADist object)
    - normalization:    string with the output mode of the function. The
                        normalization can take three modes: 'crude',
                        'normalized' and 'relative'. This input is treated
                        as optional within this function as the default
                        mode is "normalized"
	
	Outputs:
	- EVPPI:			Results of the calculation of EVPPI
	- exitflag:			Integer with indication of possible error
	- errormsg:			String with error message (in case thereof)
    %}
    
    %% Variable Checks and setup
    p = inputParser;
    addRequired(p,"f_s",@(x) isnumeric(x));
    addRequired(p,"pf",@(x) isnumeric(x) && isscalar(x));
    addRequired(p,"X_marg",@(x) strcmp(class(x),"ERADist"));
    addRequired(p,"c_R",@(x) isnumeric(x) && isscalar(x));
    addRequired(p,"c_F",@(x) isnumeric(x) && isscalar(x));

    % Set of expected output types
    expectedEVPPIOutputType = {'crude','normalized','relative'};

    % Set the expected Bandwidth Estimation Methods
    expectedBandwidthEstimationOptions = {'MLCV','fitdist'};

    addRequired(p,"normalization", @(x) ...
        any(validatestring(x,expectedEVPPIOutputType,1)));

    addParameter(p,"integration_points",5000,@(x) x>0 && isscalar(x) && isnumeric(x));
    addParameter(p,"bandwidth_estimation_method",'MLCV', @(x) ...
        any(validatestring(x, expectedBandwidthEstimationOptions,1)));

    addOptional(p,"optimal_bandwidth_weights",[],@(x) isnumeric(x));

    parse(p,f_s,pf,X_marg,c_R,c_F,normalization,varargin{:});

    % Set the optimal weights
    w_opt = p.Results.optimal_bandwidth_weights;
    
    %% PRECOMPUTATIONS - PART 1: Setup for MLCV

    % remove multiple and Inf failure samples for optimization
    f_s =  p.Results.f_s;
    f_s(any(isinf(f_s),2),:) = [];

    % Perform computations in case the 'MLCV' option is chosen
    if strcmp(p.Results.bandwidth_estimation_method,"MLCV")
        fprintf('\n-Computation of EVPPI through MLCV:');
        
    else
        fprintf('\n-Computation of EVPPI through fitdist-MATLAB function:');
    end

    % Get the dimension and number of samples from given distribution
    [N,d] = size(f_s);

    %% PRECOMPUTATIONS - PART 2: Setup for 
    % Generate Empty arrays to store objects (set memory in advance)
    crude_EVPPI = zeros(1,d);
    KDE = cell(1,d);

    % Preset the error message and exit flag
    exitflag = 1;
    errormsg = [];

    
    % Compute the threshold of the boundaries of the cost of repair vs the
    % cost of failure
    PF_thres = p.Results.c_R/p.Results.c_F;

    %% Find optimal bandwidth for kernel density estimation (kde)

    % Perform computations in case the 'MLCV' option is chosen
    if any(validatestring(p.Results.bandwidth_estimation_method,{'MLCV','fitdist'})) && ...
            isempty(w_opt)

        [~, idcs_x, idcs_u] = unique(f_s, 'rows', 'stable');
        counts = accumarray(idcs_u, 1);    % number of occurrence of each sample
        
        % Solver parameters
        Maxiter = 100;       % default of fminbnd is 500
        Tolx    = 1e-4;     % default of fminbnd is 1e-4
        lb      = 0;         % lower boundary for search interval on w
        % heuristic upper boundary for search interval on w; Inf not well handled
        ub      = 5*abs(max(f_s)-min(f_s));
        
        % Silverman's rule of thumb for starting points for band width estimation
        %w0_vec = 1.06*std(samplesX)*N^(-1/5);
        % heuristic starting points (global minimum is often closer to 0) 
        w0_vec = 1/10*abs(max(f_s)-min(f_s));
        fprintf('\n-Calculating optimal bandwidths:\n');
        
        % Find the optimal bandwidth for each dimension with maximum likelihood 
        % cross validation (MLCV)
        for u=1:d
            options = optimset('Display', 'off', 'MaxIter', Maxiter, 'TolX', Tolx);
            w_opt_handle = @(w) w_opt_finder(w, N, f_s(:,u), idcs_x, counts);
            [w_opt(u), ~, ex_flag] = fminbnd(w_opt_handle, lb, ub(u), options);
            % if optimal w not found, try fminbnd
            if ex_flag ~= 1
                disp('fminbnd was not successful, now trying fmincon\n')
                [w_opt(u), ~, ex_flag] = fmincon(w_opt_handle, w0_vec(u), [], [], [], [], lb, ub(u), [], options);
            end
            
            % exitflag handling
            switch ex_flag
                case 1
                    exitflag = 1;
                    errormsg = []; 
                case 0
                    exitflag = 0;
                    EVPPI = repelem(NaN,d);
                    errormsg = strcat('Number of iterations exceeded options.MaxIterations', ... 
                                      'or number of function evaluations exceeded', ...
                                      'options.MaxFunEvaluations.');
                    return; 
                case -1
                    exitflag = 0;
                    EVPPI = repelem(NaN,d);
                    errormsg = 'Stopped by an output function or plot function.';
                    return;
                case -2
                    exitflag = 0;
                    EVPPI = repelem(NaN,d);
                    errormsg = 'No feasible point was found.';
                    return;
                otherwise
                    exitflag = 0;
                    EVPPI = repelem(NaN,d);
                    errormsg = ['fminbnd exitflag: ' num2str(ex_flag)];
                    return;
            end
        end        
    end
    fprintf('\n-Optimal bandwidths: \n');
    disp(w_opt);


    %% Print values
    
    if p.Results.c_R > p.Results.c_F
        EVPPI = repelem(NaN,d);
        exitflag = 0;
        errormsg = "The cost of replacement is greater than " + ...
                   "the cost of failure \n \n";
        return;
    else
        % Show the cost of replacement and cost of failure
        fprintf("-cost of replacement: %.2f \n", p.Results.c_R);
        fprintf("-cost of failure: %.2f \n", p.Results.c_F);
    end
    
    %% Output
    % EVPPI: expected value of partial perfect information of each of the d
    % input variables of a system regarding a binary repair (repair/do nothing)
    % decision defined by the inputs
        
    nx = p.Results.integration_points;

    % discretize input r.v.:
    for i=1:d
        xmin = p.Results.X_marg(i).mean - 15* p.Results.X_marg(i).std;
        xmax = p.Results.X_marg(i).mean + 15* p.Results.X_marg(i).std;
        dxi  = (xmax - xmin)/nx;
        xi   = xmin+dxi/2:dxi:xmax-dxi/2;
    
        % fit kernel density estimator

        if strcmp(p.Results.bandwidth_estimation_method,"fitdist")
            fitting_dist = fitdist(f_s(:,i),'kernel');
            KDE{i} = @(x) pdf(fitting_dist,x);
            clear fitting_dist
        else
            KDE{i} = @(x) (min(kde(f_s(:,i), x, w_opt(i))));
        end
        
        %% compute conditional probability of failure, CVPPI and EVPPI:
        
        % conditional probability of failure
        % Set the storage size with empty array first
        PF_xi = zeros(1,numel(xi));
        for jj = 1:numel(xi)
             PF_xi(jj) =  KDE{i}(xi(jj))./X_marg(i).pdf(xi(jj))*pf;
        end

        % compute CVPPI
        if pf>PF_thres
            CVPPI_xi = heaviside(PF_thres-PF_xi).*(c_R-c_F*PF_xi);
        else
            CVPPI_xi = heaviside(PF_xi-PF_thres).*(c_F*PF_xi-c_R);
        end

        % Replace Inf or NaN with zeros
        CVPPI_xi(isnan(CVPPI_xi)) = 0;
        CVPPI_xi(isinf(CVPPI_xi)) = 0;
       
        % compute EVPPI
        crude_EVPPI(i) = sum(CVPPI_xi.*X_marg(i).pdf(xi),2)*dxi;
    end

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
        otherwise
            error("The output type was not properly set!");
    end

end

%% ===========================NESTED FUNCTIONS============================= 
function mlcv_w = w_opt_finder(w, N, x, id, c)
    %{
    -----------------------------------------------------------------------
    function which evaluates the mlcv for given bandwidth
    -----------------------------------------------------------------------
    Input:
    * w     : bandwidth
    * N     : total number of failure samples
    * x     : failure samples (samplesX)
    * idc   : indices to reconstruct x (samplesX) from its unique array
    * c     : vector of the multiplicity of the unique values in x (samplesX)
    -----------------------------------------------------------------------
    Output:
    * mlcw_w: maximum-likelihood cross-validation for specfic w
    -----------------------------------------------------------------------
    %}
    
    % Calculate a width sample via maximum likelihood cross-validation
    mlcv_w = 0;
    
    for k=1:length(c)
        % get all indices from samples which are different than x(k)
        idx = id(1:end~=k);
           
        mlcv_w = mlcv_w + c(k)*(log(sum(normpdf((x(id(k))-x(idx))/w).*c(1:end ~= k))) - log((N-c(k))*w));
    end
    
    % Scale mlcv_w
    %mlcv_w = mlcv_w/N;
    
    % The maximum mlcv is the minimum of -mlcv (Matlab has only minimizer
    % functions)
    mlcv_w = -mlcv_w;
end
%


% =========================================================================
function y = kde(samplesX, x_eval, bw)
%{
    -----------------------------------------------------------------------
    function to return a kde at given evaluation points and given bandwidth
    -----------------------------------------------------------------------
    Input:
    * samplesX: array of failure samplesX
    * x_eval  : array of evaluation points
    * bw      : given bandwidth
    -----------------------------------------------------------------------
    Output:
    * y: kernel density estimation evaluated at x_eval with bw
    -----------------------------------------------------------------------
    %}
    [f,~,~] = ksdensity(samplesX, x_eval(:), 'Bandwidth', bw);
    y = reshape(f(:), size(x_eval));
end
% END

