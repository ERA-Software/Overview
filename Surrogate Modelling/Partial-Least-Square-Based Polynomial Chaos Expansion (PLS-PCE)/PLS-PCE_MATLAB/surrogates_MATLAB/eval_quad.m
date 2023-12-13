function xq = eval_quad(nele,L,xGL)

ind = 1:nele;       
l = L/nele;       
XQ = l/2*repmat(xGL',1,length(ind))+l/2*repmat((2*ind-1),length(xGL),1);
xq = XQ(:);

end