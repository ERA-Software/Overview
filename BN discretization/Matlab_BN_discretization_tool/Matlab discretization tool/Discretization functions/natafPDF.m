% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

function [ f_X] = natafPDF( x,Distributions,Corr_YY )
% function [ f_X ] = natafPDF( x,Distributions,Corr_YY )
% n-dimensional Nataf pdf



f_X_marg_prod_log = zeros(size(x,1),1);
f_Y_marg_prod_log = zeros(size(x,1),1);
y = zeros(size(x));

for iDist = 1:numel(Distributions) 
    
    y(:,iDist) = norminv(Distributions{iDist}.cdf(x(:,iDist)));    
    f_X_marg_prod_log = f_X_marg_prod_log + log(Distributions{iDist}.pdf(x(:,iDist)));
    f_Y_marg_prod_log = f_Y_marg_prod_log + log(normpdf(y(:,iDist)));
    
end

f_X = exp(f_X_marg_prod_log - f_Y_marg_prod_log) .* mvnpdf(y,zeros(1,numel(Distributions)),Corr_YY);

end

