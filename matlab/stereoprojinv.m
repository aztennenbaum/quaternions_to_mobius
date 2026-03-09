function s = stereoprojinv(z,zsign)
% STEREOPROJINV  Inverse stereographic map to unit vectors.
% Usage: s = stereoprojinv(z,zsign)
if nargin<2, zsign = 1; end
s = [zsign*(z+conj(z))/2;(z-conj(z))/(2i);zsign*ones(size(z))];
s = 2*s./sum(s.^2,1) - [0;0;1];
s = real(s);
end
