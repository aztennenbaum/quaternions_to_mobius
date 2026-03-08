function M = vqs2mobius(vqs)
% VQS2MOBIUS  Quaternion + speed + stereographic axis to one Mobius matrix.
% vqs = [v qx qy qz qw] or [v qx qy qz qw sx sy].
% Usage: M = vqs2mobius(vqs)
v = vqs(1);
q = vqs(2:5).';
s = 0;
if numel(vqs)>=7, s = vqs(6) + 1i*vqs(7); end
M = normalize_m(s2mobius(s)*lorentz_z(v)*s2mobius(-s)*q2mobius(q));
end
