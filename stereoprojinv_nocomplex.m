function s = stereoprojinv_nocomplex(z,f,zsign)
% STEREOPROJINV_NOCOMPLEX  Inverse of stereoproj_nocomplex.
% Usage: s = stereoprojinv_nocomplex(z,f,zsign)
if nargin<3, zsign = 1; end
[zx,zy] = csplit(z);
c = 2./(zx.^2 + zy.^2 + 1);
c = f.*c./(c-1);
s = cjoin(zx.*c,zsign.*zy.*c);
end
