function M = lorentz_z(v)
% LORENTZ_Z  Pure boost along +z in the Mobius picture.
% v is speed in units of c with |v|<1.
% Usage: M = lorentz_z(v)
if abs(v)>=1, error('lorentz_z:range','require abs(v)<1'); end
M = normalize_m([sqrt(1-v) 0;0 sqrt(1+v)]);
end
