function y = cjoin(xr,xi)
% CJOIN  Interleave real and imaginary columns.
% Usage: y = cjoin(xr,xi)
a = xr.';
b = xi.';
y = reshape([a(:) b(:)].',2*size(a,1),[]).';
end
