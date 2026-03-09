function s = q2mrp(q)
% Q2MRP  Quaternion to modified Rodrigues parameters.
% Usage: s = q2mrp(q)
q = q(:);
s = q(1:3)./(q(4) + sign_nz(q(4))*norm(q));
end
