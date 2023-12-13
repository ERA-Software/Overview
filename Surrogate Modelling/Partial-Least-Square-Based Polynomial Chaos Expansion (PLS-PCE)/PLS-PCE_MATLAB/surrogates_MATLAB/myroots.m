function answer = myroots(lam, M)
% lam is the given correlation length, M is the number of roots we want.

% set up function. 
c = 1/lam; % So c>0.
g = @(x) (tan(x) -(2*c*x)/(x^2-c^2));

% initialise.
m = 1; 
answer= zeros(1,M);

% for all the intervals.
for i = 0:M
    
    % lower and upper bounds.
    wmin = (i-0.499)*pi;
    wmax = (i+0.499)*pi;
    
    %interval with two solutions, this is the one containing c.
    if ((wmin <= c) && (wmax >= c))
        % If it is not the first interval, look for a solution towards the
        %left boundary.
        if (wmin > 0)
            answer(m) = fzero(g, (c+wmin)/2);
            m=m+1;           
        end
        % Always look for a solution towards right boundary.
        answer(m) = fzero(g, (c+wmax)/2);
        m = m+1;        
        
    % other intervals
    elseif (wmin > 0)
        answer(m) = fzero(g, [wmin, wmax]);
        m = m+1;  
    end
   
end
answer = answer(1:M);
end

%We exclude the root zero since this results in the zero eigenfunction.
