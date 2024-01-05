% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

function [ Processing ] = DetermineCPTs( Input,Processing )

[ Processing ] = TargetNodeCPT( Input, Processing );


[ Processing ] = BasicRVsNodeCPTs( Input, Processing );

end
