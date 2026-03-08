function q = sqrt_q(qin)
% SQRT_Q  Principal quaternion square root for a unit quaternion.
% Usage: q = sqrt_q(qin)
qin = qin(:)./norm(qin);
s = sqrt(2*(qin(4)+sign_nz(qin(4))));
q = [qin(1:3)./s;s/2];
end
