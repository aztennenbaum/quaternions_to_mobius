function [xr,xi] = csplit(x)
% CSPLIT  Split interleaved real-imag columns.
% [x1r x1i x2r x2i ...] -> xr=[x1r x2r ...], xi=[x1i x2i ...]
% Usage: [xr,xi] = csplit(x)
xr = x(:,1:2:end);
xi = x(:,2:2:end);
end
