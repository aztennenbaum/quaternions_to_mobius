function q = mobius2q_norm(M)
% MOBIUS2Q_NORM  Read a quaternion from a unitary Mobius matrix.
% Usage: q = mobius2q_norm(M)
M = normalize_m(M);
q = [imag(M(1,2));real(M(2,1));imag(M(2,2));real(M(1,1))];
q = q./(sign_nz(q(4))*norm(q));
end
