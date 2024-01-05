% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

function [ Processing ] = CalculateWidthInU( Input,Processing )

[ widthWeights ] = CalculateWidthWeights( Input.IntervalsPerRV,Processing.FORMbeta );

logPr{1,1} = @(alpha) -0.28 * exp(2.9 *abs(alpha));
logPr{1,2} = @(alpha) -1.6e-2 * exp(5.8 *abs(alpha));
logPr{1,3} = @(alpha) -9.8e-4 * exp(8.7 *abs(alpha));
logPr{2,1} = @(alpha) -0.15 * exp(4.3 *abs(alpha));
logPr{2,2} = @(alpha) -2.4e-2 * exp(6.1 *abs(alpha));
logPr{2,3} = @(alpha) -2.1e-2 * exp(6.2 *abs(alpha));
logPr{3,1} = @(alpha) -0.36 * exp(3.7 *abs(alpha));
logPr{3,2} = @(alpha) -0.11 * exp(5.0 *abs(alpha));
logPr{3,3} = @(alpha) -3.7e-2 * exp(6.0 *abs(alpha));

w_mat = zeros(3,3);
for iDim = 1:numel(Input.basicRVs_X)
    
    if Input.AsNodeInBN(iDim) == 1
        
        for iBeta = 1:3
            for iNint = 1:3
                
                fun = @(w) (logPr{iBeta,iNint}(Processing.FORMalphas(iDim)) - log(normcdf(Processing.DesP_U(iDim) + w/2) - normcdf(Processing.DesP_U(iDim) - w/2)))^2;
                w_mat(iBeta,iNint) = fminbnd(fun,0,7);
            end
        end
        
        
        Processing.Width(iDim) = sum(sum(w_mat.*widthWeights));
    end
end

end

function [ widthWeights ] = CalculateWidthWeights( nIntPerDim,beta_FORM )
% Linear Interpolation between different values of nIntPerDim and
% Form Beta Values

widthWeights = zeros(3,3);

if nIntPerDim <= 5;
    widthWeights(1,1) = 1;
elseif and(nIntPerDim > 5,nIntPerDim <= 10)
    widthWeights(1,1) = (10-nIntPerDim)/(10-5);
    widthWeights(1,2) = 1-(10-nIntPerDim)/(10-5);
elseif and(nIntPerDim > 10,nIntPerDim <= 20)
    widthWeights(1,2) = (20-nIntPerDim)/(20-10);
    widthWeights(1,3) = 1-(20-nIntPerDim)/(20-10);
elseif nIntPerDim > 20
    widthWeights(1,3) = 1;
end

if beta_FORM <= -norminv(1e-3);
    widthWeights(1,:) = widthWeights(1,:);
elseif and(beta_FORM > -norminv(1e-3),beta_FORM <= -norminv(1e-5))
    widthWeights(2,:) = (1-(-norminv(1e-5)-beta_FORM)/(-norminv(1e-5)+norminv(1e-3)))*widthWeights(1,:);
    widthWeights(1,:) = ((-norminv(1e-5)-beta_FORM)/(-norminv(1e-5)+norminv(1e-3)))*widthWeights(1,:);
elseif and(beta_FORM > -norminv(1e-5),beta_FORM <= -norminv(1e-7))
    widthWeights(2,:) = ((-norminv(1e-7)-beta_FORM)/(-norminv(1e-7)+norminv(1e-5)))*widthWeights(1,:);
    widthWeights(3,:) = (1-(-norminv(1e-7)-beta_FORM)/(-norminv(1e-7)+norminv(1e-5)))*widthWeights(1,:);
    widthWeights(1,:) = zeros(1,3);
elseif beta_FORM > -norminv(1e-7)
    widthWeights(3,:) = widthWeights(1,:);
    widthWeights(1,:) = zeros(1,3);
end

end
