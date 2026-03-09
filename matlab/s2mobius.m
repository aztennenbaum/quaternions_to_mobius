function M = s2mobius(s)
% S2MOBIUS  Rotation sending the north-pole stereographic origin to s.
% Usage: M = s2mobius(s)
M = normalize_m([1 s;-conj(s) 1]);
end
