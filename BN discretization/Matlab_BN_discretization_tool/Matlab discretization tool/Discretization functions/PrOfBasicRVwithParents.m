% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

function [ CPTbasRVsiDim] = PrOfBasicRVwithParents(RVsIndicestemp,Input,Processing )

Combis = zeros(Input.IntervalsPerRV.^sum(Input.AsNodeInBN(RVsIndicestemp)),numel(Input.basicRVs_X));
    nCount = 0;
for iDim = RVsIndicestemp
       
    if Input.AsNodeInBN(iDim) == 1
        nCount = nCount + 1;
        
        temp = [];
 
        
        for iInts = 1:Input.IntervalsPerRV          
            temp = [temp;repmat(iInts,Input.IntervalsPerRV^(nCount-1),1)];
            
        end
        temp = repmat(temp,Input.IntervalsPerRV^(sum(Input.AsNodeInBN(RVsIndicestemp))-nCount),1);
        Combis(:,iDim) = temp;
        
    else
        Combis(:,iDim) = 1;
        
    end
end
clear temp

nCount = 0;
for iDim = sort(RVsIndicestemp)
    nCount = nCount + 1;
    Distributions{nCount} = Input.basicRVs_X{iDim}; 
end

nCount = 0;
for iDim = sort(RVsIndicestemp(2:end))
    nCount = nCount + 1;
    Distributions2{nCount} = Input.basicRVs_X{iDim}; 
end


Corr_UUtemp =   Processing.Corr_UU(:,sort(RVsIndicestemp));
Corr_UUtemp =   Corr_UUtemp(sort(RVsIndicestemp),:);
Corr_UUtemp2 =   Processing.Corr_UU(:,sort(RVsIndicestemp(2:end)));
Corr_UUtemp2 =   Corr_UUtemp2(sort(RVsIndicestemp(2:end)),:);

Samples = zeros(Input.nSamplesBasicRV_CPT,numel(Input.basicRVs_X));
CPTbasRVsiDim = zeros(1,size(Combis,1));

nCount = 0;
for iCombis =  1:size(Combis,1)
    nCount = nCount + 1;
    f_X_IS_Dsty = ones(Input.nSamplesBasicRV_CPT,numel(Input.basicRVs_X));
    
    for iDim = RVsIndicestemp
        
        SampleDist = Input.basicRVs_X{iDim};
        SampleDist = truncate(SampleDist,Processing.DiscrSchemeX{iDim}(Combis(iCombis,iDim)),Processing.DiscrSchemeX{iDim}(Combis(iCombis,iDim)+1));
        
        Samples(:,iDim) = SampleDist.random(Input.nSamplesBasicRV_CPT,1);
        
        f_X_IS_Dsty(:,iDim) = SampleDist.pdf(Samples(:,iDim));
    end
    
Samples2 = Samples;
Samples = Samples(:,sort(RVsIndicestemp)); 
Samples2 = Samples2(:,sort(RVsIndicestemp(2:end)));
f_X_IS_Dsty2 = f_X_IS_Dsty; f_X_IS_Dsty2(:,RVsIndicestemp(1)) = [];

CPTbasRVsiDim(nCount) = 1/Input.nSamplesBasicRV_CPT * sum(natafPDF( Samples,Distributions,Corr_UUtemp ).*prod(f_X_IS_Dsty2,2)./(natafPDF( Samples2,Distributions2,Corr_UUtemp2).*prod(f_X_IS_Dsty,2)));
    
end

end

