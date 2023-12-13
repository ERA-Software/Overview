function y_tip = bar_model(u)

% input parameters
nele = 100;
lam = 0.04;
mkl = 100;

% cross sectional area
A = 1.0;

% length of bar [m]
L = 1.0;

% uniformly distributed axial load
q = 1.0;

% Young's modulus [kPa]
mu_E = 100.0;
sigma_E = 10.0;

% parameters of lognormal distribution
m_E = log(mu_E^2/sqrt(sigma_E^2+mu_E^2));
s_E = sqrt(log(sigma_E^2/mu_E^2+1));

% quadrature nodes
xGL=[-sqrt(3/5) 0 sqrt(3/5)];

% evaluate KL eigenfunctions at quadrature points
xq = eval_quad(nele,L,xGL);         
ef = klsetup(lam, mkl, xq );

% evaluate tip displacement
y_tip = febar2(E_real2(ef,u,m_E,s_E),q,A,L,nele);

end