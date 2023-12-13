function u_tip = febar2( Ex, q, A, L, nele )
% returns the tip displacement of a bar subjected to variable axial loading
%  Ex: Young's modulus evaluated at quadrature nodes
%  q: uniformly distributed load
%  A: cross sectional area
%  L: length of beam
%  nele: number of elements

% number of nodes
nnode = nele+1;

% element length
l = L/nele;

% stiffness matrix
K = sparse(nnode,nnode);

% consistent nodal forces
F = zeros(nnode,1);

% shape functions and their derivatives
wGL = [5 8 5]/9;

% assemble stiffness matrix and force vector
F(1:end) = l*q; 
F(1) = F(1)/2; 
F(end) = F(end)/2;

Kei = A/l/2*Ex;
Ke = sum(reshape(Kei.*repmat(wGL',nele,1),3,nele),1);  

K = K + sparse(1:nele, 1:nele, Ke, nnode, nnode);
K = K + sparse(1:nele,2:nele+1,-Ke,nnode,nnode);
K = K + sparse(2:nele+1,1:nele,-Ke,nnode,nnode);
K = K + sparse(2:nele+1,2:nele+1,Ke,nnode,nnode);

Kred = K(2:nnode,2:nnode);
Fred = F(2:nnode);

% solution for nodal displacements
u = Kred \ Fred;

% tip displacement
u_tip = u(length(u));

end

