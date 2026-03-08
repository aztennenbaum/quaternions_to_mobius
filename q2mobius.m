function M = q2mobius(q)
% Q2MOBIUS  Lift a unit quaternion to SU(2) then normalize det(M)=1.
% Usage: M = q2mobius(q)
q = q(:)./norm(q);
a = q(4) - 1i*q(3);
b = -q(2) + 1i*q(1);
M = [a b;-conj(b) conj(a)];
M = normalize_m(M);
end
