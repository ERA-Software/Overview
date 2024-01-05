% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

function [ ] =  WriteBNxml(Input,Processing,OutputFileName)

nCount = 0;
for iDim = 1:numel(Input.basicRVs_X)
    
    if Input.AsNodeInBN(iDim) == 1
        nCount = nCount + 1;
        
        bayesNet.node_sizes(nCount) = numel(Processing.DiscrSchemeX{iDim})-1;
        
        nSignificantDigits = 3; ExitFlag = 0;
        while and(min(ExitFlag) == 0, nSignificantDigits < 20)
            
            for iState = 1:bayesNet.node_sizes(nCount)
                
                if round(Processing.DiscrSchemeX{iDim}(iState),nSignificantDigits,'significant') == round(Processing.DiscrSchemeX{iDim}(iState+1),nSignificantDigits,'significant')
                    nSignificantDigits = nSignificantDigits + 2;
                else
                    ExitFlag(iState) = 1;
                end
            end
            
        end
        
        for iState = 1:bayesNet.node_sizes(nCount)
            
            state_identifiers{nCount}{iState} = [num2str(round(Processing.DiscrSchemeX{iDim}(iState),nSignificantDigits,'significant')),'_TO_',...
                num2str(round(Processing.DiscrSchemeX{iDim}(iState+1),nSignificantDigits,'significant'))];
            
            state_identifiers{nCount}{iState} = strrep(state_identifiers{nCount}{iState},'-','minus');
            state_identifiers{nCount}{iState} =  strrep(state_identifiers{nCount}{iState},'.','_');
            
        end
        
        bayesNet.CPD{nCount}.CPT = Processing.CPTbasRVs{iDim};
    end
end
bayesNet.node_sizes(end+1) = 2;
state_identifiers{nCount+1} = [{'Failure'},{'Survival'}];
bayesNet.CPD{nCount+1}.CPT = Processing.CPTtarget;

icount = 0
for iDim = 1:numel(Input.basicRVs_X)
    if Input.AsNodeInBN(iDim) == 0
        Input.BN_DAG(iDim,:) = []; Input.BN_DAG(:,iDim) = [];
    else
        icount = icount + 1;
        NodeNames{icount} = Input.NodeLabel{iDim};
    end
    

end
NodeNames{icount+1} = Input.NodeLabel{iDim + 1};

for iDAG = 1:size(Input.BN_DAG,1)
    bayesNet.parents{iDAG} = find(Input.BN_DAG(:,iDAG) == 1);
end

bnt2genie( OutputFileName,bayesNet,[],state_identifiers,NodeNames )

end

