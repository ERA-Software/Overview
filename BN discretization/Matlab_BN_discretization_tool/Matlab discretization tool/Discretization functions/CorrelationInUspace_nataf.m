% Structural reliability problems discretized for Bayesian networks
% Kilian Zwirglmaier
% kilian.zwirglmaier@tum.de
% Engineering Risk Analysis Group, Technische Universität München
% www.era.bgu.tum.de
% Version Oktober 2015

function [ Processing ] = CorrelationInUspace_nataf( Input )
% Calculates the correlation matrix of standard normal random variables U
% (Corr_UU) and its lower triangular matrix obtained through cholesky
% decomposition (L_UU), if the corresponding random variables X in X-space have
% distributions as defined in Input.basicRVs_X and correlation as defined
% in Input.Corr_XX

disp('Calculating correlation of basic random variables in U-space...');

nVar = size(Input.Corr_XX);
Corr_UU = zeros(nVar);


for iVar1 = 1:nVar % runs through all pairs of RVs iVar1,iVar2
    
    
    Parameters1 = Input.basicRVs_X{iVar1}.Params;
    MEAN_iVar1 = Input.basicRVs_X{iVar1}.mean;
    STD_iVar1 = Input.basicRVs_X{iVar1}.std;
    
    switch length(Parameters1) % Switches cases depending on if the distribution has 1, 2 or 3 parameters
        
        
        case 1
            
            z1 = @ (x) (icdf(Input.basicRVs_X{iVar1}.DistName,normcdf(x,0,1),Parameters1(1))...
                -MEAN_iVar1)/STD_iVar1;
            
        case 2
            
            z1 = @ (x) (icdf(Input.basicRVs_X{iVar1}.DistName,normcdf(x,0,1),Parameters1(1)...
                ,Parameters1(2))-MEAN_iVar1)/STD_iVar1;
            
        case 3
            
            z1 = @ (x) (icdf(Input.basicRVs_X{iVar1}.DistName,normcdf(x,0,1),Parameters1(1)...
                ,Parameters1(2),Parameters1(3))-MEAN_iVar1)/STD_iVar1;
            
    end
    
    
    for iVar2 = iVar1+1:nVar
        
        if Input.Corr_XX(iVar1,iVar2) ==  0
            Corr_UU(iVar1,iVar2) = 0;
        else
            rij = Input.Corr_XX(iVar1,iVar2); % Correlation between the Variables in their original space
            
            
            Parameters2 = Input.basicRVs_X{iVar2}.Params;
            MEAN_iVar2 = Input.basicRVs_X{iVar2}.mean;
            STD_iVar2 = Input.basicRVs_X{iVar2}.std;
            
            switch length(Parameters2) % Switches cases depending on if the distribution has 1, 2 or 3 parameters
                
                
                case 1
                    
                    z2 = @ (y) (icdf(Input.basicRVs_X{iVar2}.DistName,normcdf(y,0,1),Parameters2(1))...
                        -MEAN_iVar2)/STD_iVar2;
                    
                case 2
                    
                    z2 = @ (y) (icdf(Input.basicRVs_X{iVar2}.DistName,normcdf(y,0,1),Parameters2(1)...
                        ,Parameters2(2))-MEAN_iVar2)/STD_iVar2;
                    
                case 3
                    
                    z2 = @ (y) (icdf(Input.basicRVs_X{iVar2}.DistName,normcdf(y,0,1),Parameters2(1)...
                        ,Parameters2(2),Parameters2(3))-MEAN_iVar2)/STD_iVar2;
                    
            end
            
            
            
            n = 1;
            
            rho = - 0.9;
            
            NULL(1) = -1;
            
            while NULL(end) < 0 ;
                
                
                RHO(n) = rho;
                
                
                fun =@ (x,y) z1(x).*z2(y).*(1/(2.*pi.*sqrt(1-rho.^2))).*exp((-1/(2*(1-rho^2))).*(x.^2 - 2.*rho.*x.*y+ y.^2));
                
                
                NULL(n) = integral2(fun,-8,8,-8,8)-rij;
                
                % chosing wider integration boundaries will result in numerical
                % problems as allready normcdf(9) is simply 1 in matlab
                
                if NULL(1) > 0
                    disp('ERROR not crossing zero')
                    break
                end
                
                n = n + 1;
                rho = rho + 0.1;
                
            end
            Bound_lower = RHO(end-1); Bound_upper = RHO(end);
            
            clear RHO NULL n
            
            rho = Bound_lower; n = 1; NULL(1) = - 1;
            
            while NULL(end) < 0
                
                RHO(n) = rho;
                
                fun =@ (x,y) z1(x).*z2(y).*(1/(2.*pi.*sqrt(1-rho.^2))).*exp((-1/(2*(1-rho^2))).*(x.^2 - 2.*rho.*x.*y+ y.^2));
                
                
                NULL(n) = integral2(fun,-8,8,-8,8)-rij;
                
                rho = rho + 0.01;
                n = n + 1;
                
            end
            
            Bound_lower = RHO(end-1); Bound_upper = RHO(end);
            
            clear RHO NULL n
            
            rho = Bound_lower; n = 1; NULL(1) = - 1;
            
            
            while NULL(end) < 0
                
                RHO(n) = rho;
                
                fun =@ (x,y) z1(x).*z2(y).*(1/(2.*pi.*sqrt(1-rho.^2))).*exp((-1/(2*(1-rho^2))).*(x.^2 - 2.*rho.*x.*y+ y.^2));
                
                
                NULL(n) = integral2(fun,-8,8,-8,8)-rij;
                
                rho = rho + 0.001;
                n = n + 1;
                
            end
            
            [~,I] = min(NULL.^2);
            
            Corr_UU(iVar1,iVar2) = RHO(I);
            
            clear NULL RHO n
            
        end
        
    end
end

Corr_UU = Corr_UU + Corr_UU' + eye (nVar); % final cov matrix is symmetric and rho(i,i) = 1;
Corr_UU(Corr_UU<1e-10) = 0;



Processing.Corr_UU = Corr_UU;
Processing.L_UU = chol(Corr_UU,'lower');


end

