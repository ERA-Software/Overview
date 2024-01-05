% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

function [ ] = bnt2genie( FileName,bnet,node_identifiers,state_identifiers,node_names )
% [ ] = bnt2genie(FileName,bnet,node_identifiers,state_identifiers,node_names)
% Coded by Kilian Zwirglmaier (ERA group, TUM), 11/12/2014
% kilian.zwirglmaier@tum.de
% Transferes a Bayesian network defined in the MATLAB bnt
% toolbox to a xml-file such that it can opened in Genie
% INPUT:
% FileName ... xml-file name without extension
% bnet... bnet structure defined according to the MATLAB toolbox
% node_identifiers ... a cell aray of identifiers (consider identifier rules)
%   if a empty cell array is given the nodes will be identified by default
%   as Node1 to NodeN
% state_identifiers ... a cell aray, with a cell for each node... each of
%   the cells contains another cell array with as many cells (containing
%   strings), as there are states i.e. 
%   state_identifiers = {[{'string1'},{'string2'}],[{'string3'},{'string4'},{}],[{}],[{},{},{},{}]}
%   if a empty cell array is given the states will be called by default
%   State1 to StateN
% node_names ... a cell array of strings. Note the requirements for node
%   names are not as strikt as for identifiers in genie 


docNode = com.mathworks.xml.XMLUtils.createDocument('smile');
smile = docNode.getDocumentElement;
smile.setAttribute('version','1.0');
smile.setAttribute('id','Network1');
smile.setAttribute('numsamples','1000');
smile.setAttribute('discsamples','10000');

nodesElement = docNode.createElement('nodes');
smile.appendChild(nodesElement);

%% Quantitatively and Qualitatively defining the dependence structure 

    % If the nodes are not given identifiers the defaults are Node1 to
    % NodeN
for iNode = 1:numel(bnet.node_sizes)
    
    if isempty(node_identifiers)
        NodeID{iNode} = ['Node',num2str(iNode)];
    else
        NodeID{iNode} = node_identifiers{iNode};
    end
end

for iNode = 1:numel(bnet.node_sizes)
    

    
    cptElement = docNode.createElement('cpt');
    cptElement.setAttribute('id',NodeID{iNode});
    nodesElement.appendChild(cptElement);
    
    for iState = 1:bnet.node_sizes(iNode)
        
    % If the states are not given identifiers the defaults are State1 to
    % StateN
    if isempty(state_identifiers)
        State_ID = ['State',num2str(iState)];
    else
        State_ID = state_identifiers{iNode}{iState};
    end
        
        stateElement = docNode.createElement('state');
        stateElement.setAttribute('id',State_ID);
        cptElement.appendChild(stateElement);
    end
    
% Writes the parents of the node to the cpt element    
    if isempty(bnet.parents{iNode}) == 0;
        parentsElement = docNode.createElement('parents');

        for iPar = 1:numel(bnet.parents{iNode})
            parentsElement.appendChild(docNode.createTextNode(NodeID{bnet.parents{iNode}(iPar)}));
            parentsElement.appendChild(docNode.createTextNode(' '));
        end
        cptElement.appendChild(parentsElement);
    end


CPT_bnet = bnet.CPD{iNode}.CPT;

if isempty(bnet.parents{iNode})
    Probabilities_genie = reshape(CPT_bnet,1,numel(CPT_bnet));
else    
    Probabilities_genie = reshape(permute(CPT_bnet,numel(bnet.parents{iNode})+1:-1:1),1,numel(CPT_bnet));
end

probabilitiesElement = docNode.createElement('probabilities');
probabilitiesElement.appendChild(docNode.createTextNode(num2str(Probabilities_genie)));
cptElement.appendChild(probabilitiesElement);

end

%% Defining the BN appearance

extensionsElement = docNode.createElement('extensions');
smile.appendChild(extensionsElement);

genieElement = docNode.createElement('genie');
genieElement.setAttribute('version','1.0');
genieElement.setAttribute('app','GeNIe 2.0.4843.0');
genieElement.setAttribute('name','Network1');
genieElement.setAttribute('faultnameformat','nodestate');
extensionsElement.appendChild(genieElement);


for iNode = 1:numel(bnet.node_sizes)

    % If the nodes are not given names by the user when he/she calls the 
    % function the default names are Node 1 to Node N
    if isempty(node_names)
        node_names{iNode} = ['Node ',num2str(iNode)];
    end
    
    nodeElement = docNode.createElement('node');
    nodeElement.setAttribute('id',NodeID{iNode});
    genieElement.appendChild(nodeElement);   
   
    
    nameElement = docNode.createElement('name');
    nameElement.appendChild(docNode.createTextNode(node_names{iNode}));
    nodeElement.appendChild(nameElement);
    
    interiorElement = docNode.createElement('interior');
    interiorElement.setAttribute('color','ffffff');
    nodeElement.appendChild(interiorElement);
    
    outlineElement = docNode.createElement('outline');
    outlineElement.setAttribute('color','000000');
    nodeElement.appendChild(outlineElement);
    
    fontElement = docNode.createElement('font');
    fontElement.setAttribute('color','000080');   
    fontElement.setAttribute('name','Arial');
    fontElement.setAttribute('size','10');
    nodeElement.appendChild(fontElement);
    
    if iNode == numel(bnet.node_sizes)
    positionElement = docNode.createElement('position');
    positionElement.appendChild(docNode.createTextNode([num2str(numel(bnet.node_sizes)*70),' ']));
    positionElement.appendChild(docNode.createTextNode([num2str(225),' ']));
    positionElement.appendChild(docNode.createTextNode([num2str(numel(bnet.node_sizes)*70 + 100),' '])); 
    positionElement.appendChild(docNode.createTextNode([num2str(300),' ']));
    nodeElement.appendChild(positionElement);       
    else
    positionElement = docNode.createElement('position');
    positionElement.appendChild(docNode.createTextNode([num2str(iNode*140),' ']));
    positionElement.appendChild(docNode.createTextNode([num2str(100),' ']));
    positionElement.appendChild(docNode.createTextNode([num2str(iNode*140 + 100),' '])); 
    positionElement.appendChild(docNode.createTextNode([num2str(150),' ']));
    nodeElement.appendChild(positionElement);
    end
end


xmlwrite([FileName,'.xdsl'],docNode);
type([FileName,'.xdsl']);

end