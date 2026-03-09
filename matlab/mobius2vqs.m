function vqs = mobius2vqs(M)
% MOBIUS2VQS  Factor a Mobius matrix into speed, quaternion, and axis.
% The svd2b() helper in this repo returns factors satisfying M = U*S*V.''
% so the rotation part is U*V.' rather than U*V'.
% Usage: vqs = mobius2vqs(M)
[U,S,V] = svd2b(M);
R = U*transpose(V);
q = mobius2q_norm(R);
r = abs(S(2,2)./S(1,1));
r2 = r.^2;
v = (r2 - 1)./(r2 + 1);
s = U(1,2)./U(2,2);
vqs = [real(v) q.' real(s) imag(s)];
end
