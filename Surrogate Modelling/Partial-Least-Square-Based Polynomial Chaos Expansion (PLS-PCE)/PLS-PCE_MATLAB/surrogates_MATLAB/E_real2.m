function Ex = E_real2( efx, zeta, mu, sigma )

% sample lognormal random field
Ex = mu*ones(size(efx,1),1);
Ex = Ex + efx*sigma*zeta;
Ex = exp(Ex);

end

