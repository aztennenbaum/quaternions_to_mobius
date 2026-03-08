function qi = qinv(q)
% QINV  Quaternion inverse for [qx;qy;qz;qw].
% For a unit quaternion this is the conjugate [-qv; qw].
% Usage: qi = qinv(q)
q = q(:);
qi = [-q(1:3);q(4)]./sum(q.^2);
end
