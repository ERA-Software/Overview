% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

function [ y ] = LSF_X( x )
% Limit state function in X-space
% x is a row vector of realisations of the basic random variables
% basicRVS_X
% y <= 0 for failure of the system/component and
% y > 0 for survival of the component/system 

a = 6;
y = a - sum(x,2);

end

