function gamma = corr_factor(I_Fj,p_j,Ns,Nc)
%% Computation of the correlation factor Ref. 1 Eq. 20
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2017-04
---------------------------------------------------------------------------
Input:
* I_Fj : matrix with the values of the indicator function in Ref. 1 Eq. 22
* p_j  : conditional probability of the level
* Ns   : number of samples simulated from each Markov chain (N/Nc)
* Nc   : number of Markov chains (seeds)
---------------------------------------------------------------------------
Output:
* gamma : correlation factor in Ref. 1 Eq. 20
---------------------------------------------------------------------------
Based on:
1."Bayesian post-processor and other enhancements of subset simulation for
   estimating failure probabities in high dimensions"
   Konstantin M. Zuev, James L. Beck, Siu-Kui Au & Lambros S. Katafygiotis.
   Computers and Structures 92-93 (2012) 283-296.
---------------------------------------------------------------------------
%}
if (nargin ~= 4)
   error('Incorrect number of parameters in function corr_factor');
end

%% Initialize variables
gamma = zeros(1,Ns-1);   % correlation factor
R     = zeros(1,Ns-1);   % autocovariance sequence
N     = Nc*Ns;           % total number of samples

%% correlation at lag 0
sums = 0;
for k = 1:Nc
   for ip = 1:Ns
      sums = sums + (I_Fj(ip,k)*I_Fj(ip,k));   % sums inside (Ref. 1 Eq. 22)
   end
end
R_0 = (1/N)*sums - p_j^2;   % autocovariance at lag 0 (Ref. 1 Eq. 22)

%% correlation factor calculation
for i = 1:Ns-1
   sums = 0;
   for k = 1:Nc
      for ip = 1:Ns-i
         sums = sums + (I_Fj(ip,k)*I_Fj(ip+i,k));  % sums inside (Ref. 1 Eq. 22)
      end
   end
   R(i)     = (1/(N-i*Nc))*sums - p_j^2;   % autocovariance at lag i (Ref. 1 Eq. 22)
   gamma(i) = (1-(i/Ns))*(R(i)/R_0);       % correlation factor (Ref. 1 Eq. 20)
end
gamma = 2*sum(gamma);   % small values --> higher efficiency  (Ref. 1 Eq. 20)

return;
%%END