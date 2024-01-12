% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

function [ Processing ] = IntervalBoundsInU( Input,Processing )

for iDim = 1:numel(Input.basicRVs_X)
    
    if Input.AsNodeInBN(iDim) == 1
        Processing.DiscrScheme{iDim} = Processing.DesP_U(iDim) - Processing.Width(iDim)/2 : Processing.Width(iDim)/(Input.IntervalsPerRV-2) : Processing.DesP_U(iDim) + Processing.Width(iDim)/2;
       
    else
        Processing.DiscrScheme{iDim} = [-inf,inf];
    end
end

end

