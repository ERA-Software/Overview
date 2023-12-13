classdef ERARosen
% This class is not working for MATLAB releases older than MATLAB R2019a.
%
% Generation of joint distribution objects. 
% Construction of the joint distribution object with
%
% Obj = ERARosen(dist,depend,opt)
%
% 'dist' must be a cell array with vector shape which contains all the
% marginal distributions (ERADist objects) and conditional distributions
% (ERACond objects) that define the joint distribution.
%
% 'depend' describes the dependency between the different marginal and
% conditional distributions. The dependency is defined by collecting vector 
% shaped matrices which contain the indices of the parents of the
% respective distributions in a cell array. The matrices within the cell 
% array must be ordered according to the place of the corresponding
% distribution in the input 'dist'. If a distribution is
% defined as a marginal, and therefore has no parents, an empty matrix([])
% must be given for that distribution in 'depend'. For conditional 
% distributions the order of the indices within one of the matrices
% corresponds to the order of the variables of the respective function
% handle of the respective ERACond object.

%{
---------------------------------------------------------------------------
Developed by:
Antonios Kamariotis (antonis.kamariotis@tum.de)
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Luca Sardi
Nicola Bronzetti
Alexander von Ramm
Matthias Willer
Peter Kaplan

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2022-01:
* Adaptation to the modification in the ERACond class for the definition of
the parameter/moment functions using cell arrays of function handles 
depending on a single input variable.
Version 2021-05:
* Corrections to make everything compatible with older MATLAB versions
First Release, 2020-10  
%--------------------------------------------------------------------------
This software generates joint distribution objects. The joint distribution
is defined by the connection of different marginal and conditional
distributions through the framework of a Bayesian network. 
While the marginal distributions are defined by ERADist classes, the
conditional distribution are defined by ERACond classes(see respective
classes).
The ERARosen class allows to carry out the transformation from physical
space X to standard normal space U (X2U) and vice versa (U2X) according
to the Rosenblatt transformation.
The other methods allow the computation of the joint PDF, the generation of
multivariate random samples and the plot of the Bayesian network which
defines the dependency between the different marginal and conditional
distributions.
---------------------------------------------------------------------------
References:

1. Hohenbichler, M. and R. Rackwitz (1981) - Non-Normal Dependent Variables 
   in Structural Reliability. 
   Journal of Eng. Mech., ASCE, 107, 1227-1238.

2. Rosenblatt, M. (1952) - Remarks on a multivariate transformation.
   Ann. Math. Stat., 23, 470-472.

3. Documentation of the ERA Distribution Classes
---------------------------------------------------------------------------
%}  

    %% MATLAB class: definition of the 'properties' block
    properties
        Dist        % cell array containing all the different marginal and conditional distributions
        Parents     % cell array containing the indices of the parent nodes
        Layers      % cell array which contains information about the computation order
        Adjacency   % adjacency matrix of the directed acyclic graph which defines the dependency between the different distributions 
    end
    
    %% MATLAB class: definition of the 'methods' block
    
    methods
        function Obj = ERARosen(dist,depend)
            
            if ~iscell(dist) % inspection of dist input
                error('Input dist must be given as a cell array.')
            else
                if min(size(dist)) ~= 1
                    error('Input dist must have a [n,1] or [1,n] shape.')
                end   
                n_dist = length(dist);
                for i = 1:n_dist
                    if ~all(size(dist{i})==1)   
                        error('The different distributions in the dist cell array must be given as 1x1 objects.')
                    end
                    if ~(isa(dist{i},'ERADist') || isa(dist{i},'ERACond'))
                        error('The objects in the dist cell array must be either ERADist or ERACond class objects.')
                    end
                end
            end      
            
            % check if depend is given as a cell array and other different
            % checks are carried out
            if ~iscell(depend)
                error('Input dist must be given as a cell array.')
            else
                if min(size(dist)) ~= 1
                    error('Input dist must have a [n,1] or [1,n] shape.')
                elseif max(size(dist)) ~= n_dist
                    error('The number of distributions according to depend does not match the number of distributions given in dist.')
                end  
            end
            
            % check if dist and depend have the same length
            if length(depend) ~= length(dist)
                error('The number of distributions according to the inputs dist and depend doesn''t match.')
            end
                        
            % construction of adjacency matrix of the Bayesian network
            adj_mat = zeros(n_dist);
            for i = 1:n_dist
                adj_mat(i,depend{i}) = 1;
            end
            % check if obtained network represents a directed acyclical
            % graph
            adj_prod = eye(n_dist);
            for i = 1:n_dist+1
                adj_prod = adj_prod*adj_mat;
                if sum(diag(adj_prod))~=0
                    error('The graph defining the dependence between the different distributions must be directed and acyclical.');
                end
            end
            
            Obj.Adjacency = adj_mat';
            
            % sort distribution into a hierarchy with different layers
            layers = {}; % cell array containing arrays of the different layers
            rem_dist = 1:n_dist; % indices of the remaining distributions
            while sum(rem_dist) > 0
                n_dep_rem = sum(adj_mat,2);
                curr_d = n_dep_rem == 0; % distributions on current layer
                curr_dist = rem_dist(curr_d);
                layers{end+1} = curr_dist; %#ok<AGROW> % save distributions on current layer
                adj_mat(:,curr_dist) = 0;
                adj_mat(curr_d,:) = [];
                rem_dist(curr_d) = [];
            end
           
            % finalize properties
            Obj.Dist = dist;
            if isrow(depend)
                Obj.Parents = depend;
            else
                Obj.Parents = depend';
            end
            
            if length(layers) < 2
                error('The defined joint distribution consists only of independent distributions. This type of joint distribution is not supported by ERARosen.')
            end
            Obj.Layers = layers;
           
        end
        
        %------------------------------------------------------------------------------------------------------------------
        function U = X2U(Obj,X,opt)
            % Carries out the transformation from physical space X to
            % standard normal space U.
            %
            % U = X2U(Obj,X,opt)
            %
            % X must be a [n,d]-matrix (n = number of data points, d = 
            % dimensions).
            % If no error message should be given in case of the detection
            % of an improper distribution, give opt as character array
            % 'NaN'.
            %
            
            n_dim = length(Obj.Dist); % number of dimensions of the joint distribution
            
            % check if all marginal and conditional distributions are
            % continuous
            for i = 1:n_dim
                switch lower(Obj.Dist{i}.Name)
                    case {'binomial','geometric','negativebinomial','poisson'}
                        error('At least one of the marginal or conditional distributions is a discrete distribution, the transformation X2U is therefore not possible.')
                end
            end
            
            % check of the dimensions of input X
            if size(X,2) == 1 % in case that only one point X is given, he can be defined either as row or column vector
                X = X';
            end
            if size(X,2) ~= n_dim
                error('X must be an array of size [n,d], where d is the number of dimensions of the joint distribution.') 
            end
            
            n_X = size(X,1);
            uncond = Obj.Layers{1}; % indices of unconditional distributions (ERADist classes)
            cond = 1:n_dim; 
            cond(uncond) = []; % indices of conditional distributions
            U = zeros(size(X));
            
            % transformation of the unconditional distributions
            for i = uncond
                U(:,i) = norminv(Obj.Dist{i}.cdf(X(:,i)));
            end
            
            % transformation of the conditional distributions
            for i = cond
                U(:,i) = norminv(Obj.Dist{i}.condCDF(X(:,i),X(:,Obj.Parents{i})));
            end
          
            % find rows with NaNs
            lin_ind = find(isnan(U));
            row_ind = mod(lin_ind-1,n_X)+1;
            
            if nargin == 3 % opt was given
                if strcmpi(opt,'NaN')
                    U(row_ind,:) = NaN; % give no error messages, but set invalid rows to NaN
                    return
                end
            else % opt was not given
                if ~isempty(lin_ind) % NaN's were found -> error message
                    error('Invalid joint distribution was created.')
                end                
            end
                        
        end
        
        %------------------------------------------------------------------------------------------------------------------
        function X = U2X(Obj,U,opt)
            % Carries out the transformation from standard normal space U 
            % to physical space X .
            %
            % X = U2X(Obj,U,opt)
            %
            % U must be a [n,d]-matrix (n = number of data points, d =
            % dimensions).
            % If no error message should be given in case of the detection
            % of an improper distribution, give opt as character array
            % 'NaN'.
            %
            
            n_dim = length(Obj.Dist); % number of dimensions of the joint distribution
            
            % check of the dimensions of input X
            if size(U,2) == 1 % in case that only one point X is given, he can be defined either as row or column vector
                U = U';
            end
            if size(U,2) ~= n_dim
                error('U must be an array of size [n,d], where d is the number of dimensions of the joint distribution.') 
            end
            
            n_U = size(U,1);
            uncond = Obj.Layers{1}; % indices of unconditional distributions
            n_layers = length(Obj.Layers);
            CDF_values = normcdf(U); 
            X = zeros(size(U));
            
            % transformation of the unconditional distributions
            for i = uncond
                X(:,i) = Obj.Dist{i}.icdf(CDF_values(:,i));
            end
            
            % transformation of the conditional distributions layer after
            % layer
            for i = 2:n_layers
                cond = Obj.Layers{i};
                for j = cond
                    X(:,j) = Obj.Dist{j}.condiCDF(CDF_values(:,j),X(:,Obj.Parents{j}));
                end
            end
            
            % find rows with NaNs
            lin_ind = find(isnan(X));
            row_ind = mod(lin_ind-1,n_U)+1;
            
            if nargin == 3 % opt was given
                if strcmpi(opt,'NaN')
                    X(row_ind,:) = NaN; % give no error messages, but set invalid rows to NaN
                    return
                end
            else % opt was not given
                if ~isempty(lin_ind) % NaN's were found -> error message
                    error('Invalid joint distribution was created.')
                end                
            end
            
        end
        
        %-----------------------------------------------------------------------------------------------------------------
        function jointpdf = pdf(Obj,X,opt)
            % Computes the joint PDF.
            %
            % jointpdf = pdf(Obj,X,opt)
            %
            % X must be a [n,d]-matrix (n = number of data points, d =
            % dimensions).
            % If no error message should be given in case of the detection
            % of an improper distribution, give opt as character array
            % 'NaN'.
            %
            
            n_dim = length(Obj.Dist); % number of dimensions of the joint distribution
            
            % check of the dimensions of input X
            if size(X,2) == 1 % in case that only one point X is given, he can be defined either as row or column vector                
                X = X';
            end
            if size(X,2) ~= n_dim
                error('X must be an array of size [n,d], where d is the number of dimensions of the joint distribution.') 
            end
            
            uncond = Obj.Layers{1}; % indices of unconditional distributions
            cond = 1:n_dim; % indices of conditional distributions
            cond(uncond) = [];
            pdf_values = zeros(size(X)); % pdf values of marginal and conditional distributions
            
             % computation of the pdf values of the unconditional
             % distributions (ERADist classes)
            for i = uncond
                pdf_values(:,i) = Obj.Dist{i}.pdf(X(:,i));
            end
            
            % computation of the pdf values of the conditional
            % distributions (ERACond classes)
            for i = cond
                pdf_values(:,i) = Obj.Dist{i}.condPDF(X(:,i),X(:,Obj.Parents{i}));
            end
            
            jointpdf = prod(pdf_values,2); % computing the product of all the unconditional and conditional PDF values
            
            % give error message if invalid joint distribution was created (PDF value is NaN)
            if nargin ~= 3
                if sum(isnan(jointpdf))
                    error('Invalid joint distribution was created.')
                end
                
            elseif ~strcmpi(opt,'NaN')
                if sum(isnan(jointpdf))
                    error('Invalid joint distribution was created.')
                end
            end
            
        end
            
        %------------------------------------------------------------------------------------------------------------------
        function jointrandom = random(Obj,n)
            % Creates n samples of the joint distribution.
            % Every row in the output matrix corresponds to one sample.
            %
            % jointrandom = random(Obj,n)
            %
            
            n_dim = length(Obj.Dist); % number of dimensions of the joint distribution
            jointrandom = zeros(n,n_dim);
            uncond = Obj.Layers{1}; % indices of unconditional distributions
            n_layers = length(Obj.Layers);

            % samples from the unconditional distributions
            for i = uncond
                jointrandom(:,i) = Obj.Dist{i}.random(n,1);
            end
            
            % samples from the conditional distributions layer after
            % layer
            for i = 2:n_layers
                cond = Obj.Layers{i};
                for j = cond
                    jointrandom(:,j) = Obj.Dist{j}.condRandom(jointrandom(:,Obj.Parents{j}));
                end
            end
            
            % find NaNs and give error message
            if sum(sum(isnan(jointrandom)))
                error('Invalid joint distribution was created.')
            end
            
        end
        
        %------------------------------------------------------------------------------------------------------------------
        function fig = plotGraph(Obj,opt)
            % Plots Bayesian network which defines the dependency between 
            % the different distributions.
            %
            % fig = plotGraph(Obj,opt)
            %
            % If opt is given as 'numbering' the nodes are named according
            % to their order of input in dist(e.g., the first distribution
            % is named #1, etc.). If no ID was given to a certain 
            % distribution, the distribution is also named according to its
            % position in dist, otherwise the property ID is taken as the
            % name of the distribution.
            %
            
        n_dist = length(Obj.Dist);
        node_names = cell(n_dist,1);
        
        % determine the naming of the nodes
        if nargin == 2 % opt was given
            if strcmpi(opt,'numbering') % opt was given as 'numbering'
                for i = 1:n_dist
                    node_names{i} = [' #',num2str(i)];
                end
            else
                error('opt must be given as ''numbering''.')
            end
        elseif nargin == 1
            for i = 1:n_dist
                if isempty(Obj.Dist{i}.ID) % distribution has no ID
                   node_names{i} = [' #',num2str(i)];
                else % distribution has ID
                   node_names{i} = [' ',Obj.Dist{i}.ID];
                end
            end
        end
        
        graph = digraph(Obj.Adjacency,node_names); % create graph
        fig = figure('Color',[1,1,1]);
        p = plot(graph); % plot graph
        
        % change of the plot design
        layout(p,'layered')
        p.NodeColor = [0.3,0.3,0.3];
        p.MarkerSize = 5;
        p.EdgeColor = [0,0,0];
        p.LineWidth = 0.5;
        p.EdgeAlpha = 0.3;
        p.ArrowSize = 5;
        % p.NodeFontSize = 15; % does not work for some older MATLAB versions
        axis off
        
        end
        
    end
end