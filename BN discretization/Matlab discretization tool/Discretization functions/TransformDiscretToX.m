% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

function [ Processing ] = TransformDiscretToX(Input,Processing)
disp('Transform discretization scheme to X-space...')
for iDim = 1:numel(Input.basicRVs_X)
    
    if Input.AsNodeInBN(iDim) == 1
        Processing.DiscrSchemeX{iDim}(1) = Input.basicRVs_X{iDim}.icdf(0);
        for iBound = 1:numel(Processing.DiscrScheme{iDim})
            
            TrafoPoint = Processing.DesP_U; TrafoPoint(iDim) = Processing.DiscrScheme{iDim}(iBound);
            V_TransformationPoint = (Processing.L_UU*TrafoPoint')';
            Processing.DiscrSchemeX{iDim}(iBound+1) = Input.basicRVs_X{iDim}.icdf(normcdf(V_TransformationPoint(iDim)));
        end
        Processing.DiscrSchemeX{iDim}(end+1) = Input.basicRVs_X{iDim}.icdf(1);
        
    else
        Processing.DiscrSchemeX{iDim} = [Input.basicRVs_X{iDim}.icdf(0),Input.basicRVs_X{iDim}.icdf(1)];
    end
end


end

