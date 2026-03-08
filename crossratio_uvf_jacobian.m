function [z,J] = crossratio_uvf_jacobian(u,v,f)
% CROSSRATIO_UVF_JACOBIAN  Jacobian wrt [u1 v1 u2 v2 u3 v3 u4 v4 f].
% u,v are length-4 vectors. Returns z and a 2x9 real Jacobian.
% Usage: [z,J] = crossratio_uvf_jacobian(u,v,f)
zz = zeros(4,1);
B = zeros(8,9);
for k = 1:4
    [zz(k),dzdu,dzdv,dzdf] = uvf_jacobian(u(k),v(k),f);
    rows = (2*k-1):(2*k);
    B(rows,2*k-1) = [real(dzdu);imag(dzdu)];
    B(rows,2*k) = [real(dzdv);imag(dzdv)];
    B(rows,9) = [real(dzdf);imag(dzdf)];
end
z = crossratio(zz(1),zz(2),zz(3),zz(4));
g = crossratio_jacobian(zz(1),zz(2),zz(3),zz(4));
G = [real(g(1)) -imag(g(1)) real(g(2)) -imag(g(2)) real(g(3)) -imag(g(3)) real(g(4)) -imag(g(4));...
     imag(g(1))  real(g(1)) imag(g(2))  real(g(2)) imag(g(3))  real(g(3)) imag(g(4))  real(g(4))];
J = G*B;
end
