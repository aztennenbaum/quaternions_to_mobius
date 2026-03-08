function p = quatmult(q,r)
% QUATMULT  Hamilton product for [qx;qy;qz;qw].
% Usage: p = quatmult(q,r)
q = q(:); r = r(:);
p = [q(4)*r(1:3) + r(4)*q(1:3) - cross(q(1:3),r(1:3));...
     q(4)*r(4) - q(1:3).'*r(1:3)];
end
