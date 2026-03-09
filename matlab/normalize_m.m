function M = normalize_m(M)
% NORMALIZE_M  Scale a 2x2 complex matrix so det(M)=1.
% Usage: M = normalize_m(M)
d = det(M);
if d==0, error('normalize_m:singular','singular matrix'); end
M = M./sqrt(d);
end
