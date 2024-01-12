% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

function [ Processing ] = DiscretizeInU(Input,Processing)
disp('Discretizing the basic random variables in U-space...')
[ Processing ] = CalculateWidthInU( Input,Processing );

[ Processing ] = IntervalBoundsInU( Input,Processing );

end

