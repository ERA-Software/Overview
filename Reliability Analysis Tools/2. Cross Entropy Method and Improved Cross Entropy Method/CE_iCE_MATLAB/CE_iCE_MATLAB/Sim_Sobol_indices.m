function [S_F1, exitflag, errormsg] = Sim_Sobol_indices(samplesX, Pf, distr)
%% Compute first order Sobol' indices from failure samples
%{
---------------------------------------------------------------------------
Created by:
Daniel Koutas
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2022-04
2022-10 (Max Ehre): 
Fixed bad performance for ex1_Luyi_paper.m by extending 
integration domain in line 145 (pm 5 dev std -> pm 15 std dev)
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
* samplesX: failure samples 
* Pf      : estimated failure probability
* distr   : ERADist or ERANataf object containing the infos about the random
            variables
---------------------------------------------------------------------------
Output:
* S_F1      : vector of first order sensitivity indices
* exitflag  : flag whether method was successful or not
* errormsg  : error message describing what went wrong
---------------------------------------------------------------------------
%}

%% Initialization
% Check for ERANataf or ERADist
if any(strcmp('Marginals',fieldnames(distr)))  
    dist = distr.Marginals;
    % check for dependent variables
    if ~isequal(distr.Rho_X, eye(length(distr.Marginals)))
        fprintf("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        fprintf("Results of sensitivity analysis should be interpreted with care for dependent inputs.")
        fprintf("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
    end
else
    dist = distr;
end

% remove multiple and Inf failure samples for optimization
samplesX(any(isinf(samplesX),2),:) = [];

[~, idcs_x, idcs_u] = unique(samplesX, 'rows', 'stable');
counts = accumarray(idcs_u, 1);    % number of occurrence of each sample

[N, d] = size(samplesX);    % Number of RVs and samples

% Solver parameters
Maxiter = 100;       % default of fminbnd is 500
Tolx    = 1e-4;     % default of fminbnd is 1e-4
lb      = 0;         % lower boundary for search interval on w
% heuristic upper boundary for search interval on w; Inf not well handled
ub      = 5*abs(max(samplesX)-min(samplesX));

% Silverman's rule of thumb for starting points for band width estimation
%w0_vec = 1.06*std(samplesX)*N^(-1/5);
% heuristic starting points (global minimum is often closer to 0) 
w0_vec = 1/10*abs(max(samplesX)-min(samplesX));

% variable to plot posterior pdfs and mlcv dependent on w
plot_v = 0;

% select integration method 'inverse' or 'integral'
int_method = 'integral';

if ~(strcmp(int_method, 'integral') || strcmp(int_method, 'inverse'))
    error("Select either 'integral' or 'inverse' as your numerical integration method");
end

%% Find optimal bandwidth for kernel density estimation (kde)
fprintf('\n-Calculating optimal bandwidths:\n');

% Find the optimal bandwidth for each dimension with maximum likelihood 
% cross validation (MLCV)
for u=1:d
    options = optimset('Display', 'off', 'MaxIter', Maxiter, 'TolX', Tolx);
    w_opt_handle = @(w) w_opt_finder(w, N, samplesX(:,u), idcs_x, counts);
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
            S_F1 = [];
            errormsg = strcat('Number of iterations exceeded options.MaxIterations', ... 
                              'or number of function evaluations exceeded', ...
                              'options.MaxFunEvaluations.');
            return; 
        case -1
            exitflag = 0;
            S_F1 = [];
            errormsg = 'Stopped by an output function or plot function.';
            return;
        case -2
            exitflag = 0;
            S_F1 = [];
            errormsg = 'No feasible point was found.';
            return;
        otherwise
            exitflag = 0;
            S_F1 = [];
            errormsg = ['fminbnd exitflag: ' num2str(ex_flag)];
            return;
    end
end
fprintf('\nOptimal bandwidths: \n');
disp(w_opt);
%% Compute sensitivity indices
fprintf('\nCalculating sensitivity indices:\n');

if strcmp(int_method, 'integral')
    for u=1:d
        pdf_int = @(x) (min(kde(samplesX(:,u), x, w_opt(u))* Pf ./ dist(u).pdf(x), 1) - Pf).^2 .* dist(u).pdf(x);
        % integration range from: [mean-5std, mean+5std]
        B(u) = integral(pdf_int, dist(u).mean-15*dist(u).std, dist(u).mean+15*dist(u).std);
    end
    % Sensitivity indices computed via eq. 4 with the samples obtained from 
    % maximum-likelihood cross-validation method
    S_F1 = B / (Pf*(1-Pf));
    
elseif strcmp(int_method, 'inverse')
    % numerical integration samples
    N_i = 2e4;
    % define sobolset for quasi-random integration
    U0 = sobolset(1, 'Skip', 1e3, 'Leap', 1e2);
    U0 = scramble(U0,'MatousekAffineOwen');
    % final sampling points
    UX = net(U0, N_i);
    for u=1:d
        % get samples distributed according to the individual pdfs
        s = dist(u).icdf(UX);
        % get posterior pdf values approximated by kde at sample points
        [b, ~, ~] = ksdensity(samplesX(:,u), s, 'Bandwidth', w_opt(u));
        % calculate updated Bayes' samples with maximum value of 1
        B(:,u) = min(b * Pf ./ dist(u).pdf(s), 1);
    end
    S_F1 = var(B) / (Pf*(1-Pf));
        
end

% check if computed indices are valid
if ~isreal(S_F1)
    errormsg = 'Computation was not successful, at least one index is complex.';
    exitflag = 0;
elseif any(S_F1 > 1)
    errormsg = 'Computation was not successful, at least one index is greater than 1.';
    exitflag = 0;
elseif any(S_F1 < 0)
    errormsg = 'Computation was not successful, at least one index is smaller than 0.';
    exitflag = 0;
elseif any(~isfinite(S_F1))
    errormsg = 'Computation was not successful, at least one index is NaN.';
    exitflag = 0;
else 
    errormsg = [];
    exitflag = 1;
end

if plot_v
    fprintf("\n***Plotting MLCV(w) and kde(x)\n");
    
    figure;
    w_x = linspace(0.001, 2*max(w_opt), 40);
    for u=1:d
        for t=1:length(w_x)
            %y(u,t) = w_opt_finder(w_x(t), N, aid_vec(:,u), idx, counts);
            y(u,t) = w_opt_finder(w_x(t), N, samplesX(:,u), idcs_x, counts);
        end
        plot(w_x, y(u,:));
        hold on;
        %plot(w_opt(u), w_opt_finder(w_opt(u), N, aid_vec(:,u), idx, counts), '*r');
        plot(w_opt(u), w_opt_finder(w_opt(u), N, samplesX(:,u), idcs_x, counts), '*r');
    end
    xlabel('w');
    ylabel('MLCV(w)');
    title('MLCV depending on bandwidth');
    
    figure;
    for u=1:d
        [yi,xi] = ksdensity(samplesX(:,u),'Bandwidth', w_opt(u));
        %figure;
        plot(xi,yi);
        hold on;
        %histogram(samplesX(:,u), 'Normalization', 'pdf')
        xlabel('x');
        ylabel('kde(x)');
        title('KDE of failure samples')
    end
    xlabel('x');
    ylabel('kde(x)');
    title('KDE of failure samples')
end

end
%
%==========================================================================
%===========================NESTED FUNCTIONS===============================
%==========================================================================
%
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