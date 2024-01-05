% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

function [ Processing ] = BasicRVsNodeCPTs( Input, Processing )

disp('Determine the CPT of the basic random variable nodes...')
for iDim = 1:numel(Input.basicRVs_X)
    
    if Input.AsNodeInBN(iDim) == 1
        
        if sum(Input.BN_DAG(1:end,iDim)) == 0
            
            CPTbasRVs{iDim} = Input.basicRVs_X{iDim}.cdf(Processing.DiscrSchemeX{iDim}(2:end))-...
                Input.basicRVs_X{iDim}.cdf(Processing.DiscrSchemeX{iDim}(1:end-1));
        else
            
            RVsIndicestemp = [iDim,fliplr(find(Input.BN_DAG(1:end,iDim) == 1)')];
            
            
            CPTbasRVs{iDim} = PrOfBasicRVwithParents( RVsIndicestemp,Input,Processing );
            
            
        end
        
    end
end


Processing.CPTbasRVs = CPTbasRVs;
end

