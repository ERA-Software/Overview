function alpha = multi_index(M,p)
%% multi-index sequence for the computation of M-dimensional polynomials
%{
--------------------------------------------------------------------------
Multi-index sequence for the computation of M-dimensional polynomials
--------------------------------------------------------------------------
Created by:
Fong-Lin Wu (fonglin.wu@tum.de)
Felipe Uribe (felipe.uribe@tum.de)
Technische Universitat Munchen
www.era.bgu.tum.de
--------------------------------------------------------------------------
Version: 
* 2019-08 Transition from Matlab
--------------------------------------------------------------------------
Input:
* M      : Number of K-L terms (number of random variables)
* p_order: Order of PC
--------------------------------------------------------------------------
Output:
* alpha  : Multi-index sequence
--------------------------------------------------------------------------
Based on:
1."Numerical methods for stochastic computations. A spectral method approach"
   D. Xiu. 2010. Princeton University Press.
2."Stochastic finite element methods and reliability"
   B. Sudret and A. Der Kiureghian. State of the art report.
--------------------------------------------------------------------------
%}

%% procedure
alpha    = cell(p+1,1);    % multi-index
alpha{1} = zeros(1,M);     % multi-index for p = 0

switch M
   case 1  % dimension = 1
      for q = 1:p
         alpha{q+1} = q;
      end
   otherwise % dimension>1
      for q = 1:p
         s          = nchoosek(1:M+q-1,M-1);
         s1         = zeros(size(s,1),1);
         s2         = (M+q)+s1;
         alpha{q+1} = flipud(diff([s1 s s2],1,2))-1;   % -1 due to MATLAB indexing
         if sum(alpha{q+1},2) ~= q*ones(nchoosek(M+q-1,M-1),1)
            error('The sum of each row has to be equal to q-th order');
         end
      end
end
alpha = cell2mat(alpha);   % as in Ref.(1)

return;