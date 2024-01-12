% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

function [ Processing ] = TargetNodeCPT( Input, Processing )


disp('Determine the CPT of the component performance node...')
Combis = zeros(Input.IntervalsPerRV.^sum(Input.AsNodeInBN),numel(Input.basicRVs_X));
nCount = 0;
for iDim = 1:numel(Input.basicRVs_X)
    
    if Input.AsNodeInBN(iDim) == 1
       nCount = nCount + 1;
       
        temp = [];
        
        for iInts = 1:Input.IntervalsPerRV
            
            temp = [temp;repmat(iInts,Input.IntervalsPerRV^(sum(Input.AsNodeInBN)-nCount),1)];
            
        end
        temp = repmat(temp,Input.IntervalsPerRV^(nCount-1),1);
        Combis(:,iDim) = temp;
        
    else
        Combis(:,iDim) = 1;
        
    end
end
clear temp


Samples = zeros(Input.nSamplesTargetRV_CPT,numel(Input.basicRVs_X));

CPT = zeros(1,2*size(Combis,1));

nCount = 0;
for iCombis =  1:size(Combis,1)
    nCount = nCount + 2;
    SampPDF_samplDens = ones(Input.nSamplesTargetRV_CPT,1);
    
    for iDim = 1:numel(Input.basicRVs_X)
        
        SampleDist = Input.basicRVs_X{iDim};
        SampleDist = truncate(SampleDist,Processing.DiscrSchemeX{iDim}(Combis(iCombis,iDim)),Processing.DiscrSchemeX{iDim}(Combis(iCombis,iDim)+1));
        
        Samples(:,iDim) = SampleDist.random(Input.nSamplesTargetRV_CPT,1);
        
        SampPDF_samplDens = SampPDF_samplDens.* SampleDist.pdf(Samples(:,iDim));
    end
    
    
    CPT(nCount-1) = (single(1/Input.nSamplesTargetRV_CPT)*sum(heaviside(-LSF_X(Samples)).* natafPDF( Samples,Input.basicRVs_X,Processing.Corr_UU )./SampPDF_samplDens))./...
        (single(1/Input.nSamplesTargetRV_CPT)*sum(natafPDF( Samples,Input.basicRVs_X,Processing.Corr_UU )./SampPDF_samplDens));
    
    CPT(nCount)   = 1 - CPT(nCount-1);
    
end

Processing.CPTtarget = CPT;

end

