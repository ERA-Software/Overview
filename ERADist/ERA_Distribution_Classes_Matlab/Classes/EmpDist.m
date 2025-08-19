classdef EmpDist
% Returns a distribution object similar to scipy.stats based on a dataset.
%{
---------------------------------------------------------------------------
Developed by: Michael Engel
---------------------------------------------------------------------------
Initial Version: 2025-07
---------------------------------------------------------------------------
%}
    
    properties
        cleanData
        N_
        normalizedWeights
        pdfMethod
        pdfPoints
        pdfMethodParams
        mean_
        var_
        std_
        cdfFunc
        icdfFunc
        pdfFunc
    end
    
    methods
        %% Constructor
        function obj = EmpDist(data, varargin)
            % Parse inputs
            p = inputParser;
            addRequired(p, 'data', @isnumeric);
            addParameter(p, 'weights', [], @isnumeric);
            addParameter(p, 'pdfMethod', 'kde', @ischar);
            addParameter(p, 'pdfPoints', [], @isnumeric);
            addParameter(p, 'pdfMethodParams', struct(), @isstruct);
            parse(p, data, varargin{:});
            
            data = data(:);
            data = data(~isnan(data));
            obj.cleanData = data;
            obj.N_ = length(data);
            
            if isempty(p.Results.weights)
                obj.normalizedWeights = ones(size(data)) / obj.N_;
            else
                w = p.Results.weights(:);
                w = w(~isnan(p.Results.data));
                obj.normalizedWeights = w / sum(w);
            end
            
            obj.pdfMethod = lower(p.Results.pdfMethod);
            if isempty(p.Results.pdfPoints)
                obj.pdfPoints = max(1, floor(sqrt(obj.N_)));
            else
                obj.pdfPoints = p.Results.pdfPoints;
            end
            obj.pdfMethodParams = p.Results.pdfMethodParams;
            
            % statistics
            obj.mean_ = sum(obj.cleanData .* obj.normalizedWeights);
            obj.var_ = sum(obj.normalizedWeights .* (obj.cleanData - obj.mean_).^2);
            obj.std_ = sqrt(obj.var_);
            
            % cdf and icdf
            obj.cdfFunc = EmpDist.create_weighted_cdf_interp(obj.cleanData, obj.normalizedWeights, 'previous');
            obj.icdfFunc = EmpDist.create_weighted_icdf_interp(obj.cleanData, obj.normalizedWeights, 'next');
            
            % pdf
            if strcmp(obj.pdfMethod, 'kde')
                disp('EmpDist: Using Gaussian KDE for PDF!');
                [sortedData, sortedWeights] = EmpDist.sortDataWeights(obj.cleanData, obj.normalizedWeights);
                % MATLAB KDE (ksdensity supports weights)
                obj.pdfFunc = @(x) ksdensity(sortedData, x, 'Weights', sortedWeights, 'Function', 'pdf');
            else
                disp('EmpDist: Using numerical derivative for PDF!');
                obj.pdfFunc = EmpDist.create_normalized_pdf_from_cdf(obj.cdfFunc, ...
                    min(obj.cleanData), max(obj.cleanData), ...
                    obj.pdfPoints, obj.pdfMethod);
            end
        end
        
        %% Accessors
        function out = N(obj), out = obj.N_; end
        function MEAN = mean(obj), MEAN = obj.mean_; end
        function VAR = var(obj), VAR = obj.var_; end
        function STD = std(obj), STD = obj.std_; end
        function PDF = pdf(obj, x), PDF = obj.pdfFunc(x); end
        function CDF = cdf(obj, x), CDF = obj.cdfFunc(x); end
        function ICDF = icdf(obj, y), ICDF = obj.icdfFunc(y); end
        function RANDOM = random(obj, n)
            rands = rand(n, 1);
            RANDOM = obj.icdf(rands);
        end
    end
    
    methods (Static)
        %% Helpers
        function [sortedData, sortedWeights] = sortDataWeights(data, weights)
            [sortedData, idx] = sort(data);
            sortedWeights = weights(idx);
        end
        
        %% CDF interpolation
        function cdfFunc = create_weighted_cdf_interp(data, weights, method)
            [sortedData, sortedWeights] = EmpDist.sortDataWeights(data, weights);
            cumWeights = cumsum(sortedWeights);
            totalWeight = cumWeights(end);
            weightedCdfVals = cumWeights / totalWeight;
            cdfFunc = @(x) interp1(sortedData, weightedCdfVals, x, method, 'extrap');
        end
        
        function icdfFunc = create_weighted_icdf_interp(data, weights, method)
            [sortedData, sortedWeights] = EmpDist.sortDataWeights(data, weights);
            cumWeights = cumsum(sortedWeights);
            totalWeight = cumWeights(end);
            weightedCdfVals = cumWeights / totalWeight;
            icdfFunc = @(y) interp1(weightedCdfVals, sortedData, y, method, 'extrap');
        end
        
        %% PDF from numerical CDF derivative
        function pdfFunc = create_normalized_pdf_from_cdf(cdfFunc, xMin, xMax, numPoints, method)
            xGrid = linspace(xMin, xMax, numPoints);
            cdfVals = cdfFunc(xGrid);
            rawPdfVals = gradient(cdfVals, xGrid);
            area = trapz(xGrid, rawPdfVals);
            if area ~= 0
                pdfVals = rawPdfVals / area;
            else
                pdfVals = rawPdfVals;
            end
            pdfFunc = @(x) interp1(xGrid, pdfVals, x, method, 0);
        end
    end
end
