% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

function [ Processing ] = DesignPoint_U( Input )
% Find the Design point of a reliability problem in U-space
% the function makes use of the matlab-built in optimization function fmincon

disp('Finding the design point in U-space ...');

[ Processing ] = CorrelationInUspace_nataf( Input );

Processing.DesP_U = fmincon(@(u)norm(u),zeros(1,numel(Input.basicRVs_X)),[],[],[],[],[],[],@(u)OptConstraint(u,Processing,Input));
Processing.FORMbeta = norm(Processing.DesP_U);
Processing.FORMalphas = Processing.DesP_U/Processing.FORMbeta;

  disp('FORM importance measures:')
  disp(num2str(round(100*Processing.FORMalphas)./100))

end

function [ c,ceq ] = OptConstraint(u,Processing,Input)
% Constraint for the optimization (fmincon) used to find the design point

v_temp = (Processing.L_UU*u')';
x_temp = zeros(size(v_temp));

for iRVs = 1:numel(Input.basicRVs_X)
    x_temp(iRVs) =  Input.basicRVs_X{iRVs}.icdf(normcdf(v_temp(:,iRVs)));
end

ceq = Input.LSF(x_temp);
c = [];

end


