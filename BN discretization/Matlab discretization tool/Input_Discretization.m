% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

% This software efficiently discretizes a reliability problem defined by the user and
% generates a Bayesian network representing the reliability problem. The
% Bayesian network is outputted in the form of a .xml file that can be
% opened e.g. in Genie, which is available for free: 

%% Description of the reliability problem definition of the necessary discretization parameters

%% Step 1
% Definition of a cell array of matlab distribution objects
% containing the mariginal distributions of the basic random 
% variables of the reliability problem 
% (If in the corresponding BN basicRVS_X{i} is a parent of basicRVS_X{j}
% then i < j)
% If basic random variable i is to be modeled explicitly as a node in the BN 
% AsNodeInBN(i) = 1 otherwise AsNodeInBN(i) = 0
% (This decision can be based on the FORM importance measures)
basicRVs_X{1} = makedist('normal','mu',1,'sigma',0.5); 
AsNodeInBN(1) = 1;
basicRVs_X{2} = makedist('normal','mu',1,'sigma',0.3);
AsNodeInBN(2) = 1;
basicRVs_X{3} = makedist('normal','mu',1,'sigma',0.2);
AsNodeInBN(3) = 1;


% Definition of the labels of the BN-nodes... The last NodeLabel
% corresponds to the component state node

NodeLabel{1} = 'Load1';
NodeLabel{2} = 'Load2';
NodeLabel{3} = 'Load3';
NodeLabel{4} = 'LSFnode';
% ... 

%% Step 2 
% Definition of the correlation between the basic random variables in
% X-space

Corr_XX = eye(numel(basicRVs_X));
Corr_XX(1,3) = 0.7; Corr_XX(3,1) = 0.7;
Corr_XX(2,3) = 0.5; Corr_XX(3,2) = 0.5;
%% Step 3
% Definition of the limit state function (LSF) in of the reliability problem in X-space 
% in a seperate function file  the LSF has the form function [y] = LSF_X(x)
% where x is a matrix of dimensions (number of samples, number of basicRVs_X) 
% y <= 0 for failure of the system/component and
% y > 0 for survival of the component/system 

LSF = @(x) LSF_X(x);

%% Step 4
% Definition of discretization parameters 
% number of intervals used to discretize the basic random variables
IntervalsPerRV = 5;  
% number of samples used to calculate the conditional failure probability for each cell
nSamplesTargetRV_CPT = 1e5;
% number of samples used to calculate each conditional probability in the CPTs of the basic random variables
% (if the basic random variable does not have any parents the marginal probability table is calculated analytically)
nSamplesBasicRV_CPT = 1e6;

%% Definition of BN structure

BN_DAG = zeros(numel(basicRVs_X)+1);
BN_DAG(1:numel(basicRVs_X),numel(basicRVs_X) + 1) = 1; % All basic random variables are parents of the node representing component performance

% Add additional links between the basic random variables
% i.e. for a link from basicRVS_X{i} to basicRVS_X{j} write 
% BN_DAG(i,j) = 1;
% Note: Genie requires parent nodes to have a smaller node index than their
% children, thus it is not possible to define e.g. a link from node 3 to node 1...
% BN_DAG(3,1) = 1 

BN_DAG(1,3) = 1;
BN_DAG(2,3) = 1;

%%%%%%%%%%%%%%%%%%% END of user input %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% no need to alter rest of file %%%%%%%%%%%%%


Input.basicRVs_X = basicRVs_X;
Input.AsNodeInBN = AsNodeInBN;
Input.NodeLabel = NodeLabel;
Input.Corr_XX = Corr_XX;
Input.LSF = LSF;
Input.IntervalsPerRV = IntervalsPerRV;
Input.nSamplesTargetRV_CPT = nSamplesTargetRV_CPT;
Input.nSamplesBasicRV_CPT = nSamplesBasicRV_CPT;
Input.BN_DAG = BN_DAG;

clear basicRVS_X NodeLabel Corr_XX LSF IntervalsPerRV nImportanceSamplesCPT BN_DAG 
addpath([cd,'/Discretization functions'])