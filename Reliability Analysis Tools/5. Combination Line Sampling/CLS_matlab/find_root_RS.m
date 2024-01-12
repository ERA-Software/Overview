function [u_root,exitflag] = find_root_RS(g_1D,u_0,N)
%% Find the root(s) of a function by fitting a polynomial
%{
---------------------------------------------------------------------------
Created by:
Iason Papaioannou (iason.papaioannou@tum.de)
Daniel Koutas
Engineering Risk Analysis Group   
Technische Universitat Muenchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Current version 2021-06
---------------------------------------------------------------------------
Input: 
* g_1D: desired function for which to find the root(s)
* u_0: starting point
* N: number of evaluations of the g_1D function
---------------------------------------------------------------------------
Output:
* u_root: found root (either the found value or empty [])
* exitflag: flag whether a root was found (1) or not (0)
%}


if mod(N,2) == 1
    k = (N+1)/2;
else
    k = N/2;
end

% selection of points for LSF evaluations
for i=1:N
    u(i) = u_0+(i-k)*0.6;
    y(i) = g_1D(u(i));
end


idx1 = isfinite(y);
u = u(idx1);
y = y(idx1);

if length(u) < 3
    u_root=[];
    exitflag=0;
else
    p = polyfit(u,y,2);
    
    % check if any polynomial coefficients are inf or nan -> invalid fit
    if sum(~isnan(p) + isfinite(p)) < 2*length(p)
        u_root = [];
        exitflag = 0;
    else 
        r = roots(p);
        
        % check if there is any root which is not nan or inf 
        % -> one is sufficient here
        if sum(isfinite(r)) > 0
            [~,idx] = min(abs(r));
            u_root = r(idx);
            exitflag = 1;
        else 
            u_root = [];
            exitflag = 0;
        end
    end
    
    % filter out complex roots
    if ~isreal(u_root)
        u_root = [];
        exitflag = 0;
    end
    
end

end

