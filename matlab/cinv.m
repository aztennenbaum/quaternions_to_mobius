function y = cinv(x)
% CINV  Reciprocal of packed complex pairs.
% Usage: y = cinv(x)
[xr,xi] = csplit(x);
d = xr.^2 + xi.^2;
y = cjoin(xr./d,-xi./d);
end
