function [A,eigvals] = klsetup(lam, mkl, xq)

% calculate the eigenvalues.
c=1/lam;
omega=myroots(lam,mkl);
eigvals = 2*c./(omega.^2 + c^2);
eigfcts = ef(omega, lam, xq);
n = length(xq);
A = repmat(sqrt(eigvals),n,1).*eigfcts;

end

function out = ef(omega, lam, x)

% This function evaluates the eigenfunctions at the point x. Input omega 
% are the eigenvalues for the given correlation length lam.

norm = 1./(sqrt(sin(2*omega).*(0.25*(lam^2*omega - 1./omega)) - ...
    0.5*lam*cos(2*omega) + (0.5*(1+lam+lam^2*omega.^2))));

% norm is a 1 x mkl vector
% x is a n x 1 vector
mkl = length(omega);
n = length(x);
XO = repmat(x,1,mkl).*repmat(omega,n,1);

out = repmat(norm,n,1).*(sin(XO)+lam*repmat(omega,n,1).*cos(XO));

end