function [z,dzdu,dzdv,dzdf] = uvf_jacobian(u,v,f)
% UVF_JACOBIAN  Analytic Jacobian of stereographic pixels->z.
% z = (p_x+i p_y)/(1+p_z), p = [u v f]/||[u v f]||.
% Usage: [z,dzdu,dzdv,dzdf] = uvf_jacobian(u,v,f)
p = [u;v;f];
L = norm(p);
ph = p./L;
J = (eye(3).*L - (p*p')./L)./(L^2);
num = ph(1) + 1i*ph(2);
den = 1 + ph(3);
z = num./den;
dnum = [J(1,1)+1i*J(2,1),J(1,2)+1i*J(2,2),J(1,3)+1i*J(2,3)];
dden = J(3,:);
dz = (dnum.*den - num.*dden)./(den^2);
dzdu = dz(1);
dzdv = dz(2);
dzdf = dz(3);
end
