function A = q2a(q)
% Q2A  Quaternion to attitude matrix.
% q = [qx;qy;qz;qw]. Input need not be normalized.
% Usage: A = q2a(q)
q = q(:);
qx = q(1); qy = q(2); qz = q(3); qw = q(4);
A = [qx*qx-qy*qy-qz*qz+qw*qw, 2*(qx*qy+qz*qw),       2*(qx*qz-qy*qw);...
     2*(qx*qy-qz*qw),       -qx*qx+qy*qy-qz*qz+qw*qw, 2*(qy*qz+qx*qw);...
     2*(qx*qz+qy*qw),        2*(qy*qz-qx*qw),      -qx*qx-qy*qy+qz*qz+qw*qw];
A = A./sum(q.^2);
end
