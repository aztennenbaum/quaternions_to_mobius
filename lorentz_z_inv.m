function v = lorentz_z_inv(M)
% LORENTZ_Z_INV  Recover speed from a pure z-boost matrix.
% Usage: v = lorentz_z_inv(M)
M = normalize_m(M);
r = abs(M(2,2)./M(1,1));
r2 = r.^2;
v = (r2 - 1)./(r2 + 1);
end
