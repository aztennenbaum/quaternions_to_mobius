function Phi = quatmat(q)
% QUATMAT  Left-multiplication matrix for quaternion q.
% This is the 4x4 matrix Phi(q) such that quatmult(q,r) = Phi*r.
% Usage: Phi = quatmat(q)
q = q(:);
Phi = [q(4)*eye(3) - skew(q(1:3)), q(1:3);...
      -q(1:3).',                     q(4)];
end
