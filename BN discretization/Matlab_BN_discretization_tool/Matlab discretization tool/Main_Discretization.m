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


% Choose option 
% OnlyFORManalysis performs a FORM analysis of the problem and outputs
% the FORM importance measures alpha in the Command Window; 
% These imortance measures may be used (together with other criterions) to
% decide wether a RV is to be modeled explicitely in the BN
% 
Option = 'CompleteDiscretization'; %'OnlyFORManalysis';%

%%%% read input, defined by the user in the file Input_Discretization.m %%%%
run('Input_Discretization.m')

% I. Find Design Point
  [ Processing ] = DesignPoint_U( Input );

if strcmp(Option,'CompleteDiscretization')
    % II. Determine Discretization Scheme in U-space
      [ Processing ] = DiscretizeInU(Input,Processing);

    % III. Transform Discretization Scheme to X-space
     [ Processing ] = TransformDiscretToX(Input,Processing);


    % IV. Determine CPTs of Component performance node through importance sampling
    [ Processing ] = TargetNodeCPT( Input, Processing );


    % V. Determine CPTs of basic random variables
    [ Processing ] = BasicRVsNodeCPTs( Input, Processing );

    % VI. Write BN to file (.xml) 
      WriteBNxml(Input,Processing,'DiscrStruRelProbForBN.xml'); % may be altered to rename the output file
end