function x = lorentz_z_vec(v,x)
% LORENTZ_Z_VEC  Apply a z-boost to a 4-vector [x;y;z;t].
% Usage: x = lorentz_z_vec(v,x)
g = 1/sqrt(1-v^2);
z0 = x(3);
t0 = x(4);
x(3) = g*(z0 + v*t0);
x(4) = g*(t0 + v*z0);
end
