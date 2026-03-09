function y = normalize_vec(x)
% NORMALIZE_VEC  Divide by Euclidean norm.
% Usage: y = normalize_vec(x)
n = norm(x);
if n==0, error('normalize_vec:zero','zero input'); end
y = x./n;
end
