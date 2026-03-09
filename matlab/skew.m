function S = skew(v)
% SKEW  3x3 cross-product matrix.
% Usage: S = skew(v)
S = [0 -v(3) v(2);v(3) 0 -v(1);-v(2) v(1) 0];
end
