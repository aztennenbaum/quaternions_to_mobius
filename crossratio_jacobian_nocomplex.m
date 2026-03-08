function J = crossratio_jacobian_nocomplex(x)
% CROSSRATIO_JACOBIAN_NOCOMPLEX  Real 2x8 Jacobian of [Re z, Im z].
% x = [ax ay bx by cx cy dx dy].
% Usage: J = crossratio_jacobian_nocomplex(x)
a = x(:,1:2);
b = x(:,3:4);
c = x(:,5:6);
d = x(:,7:8);
z = crossratio_nocomplex(x);
[gR,gI] = csplit([cmult(z,cinv(a-c)-cinv(a-d)),...
                  cmult(z,cinv(b-d)-cinv(b-c)),...
                  cmult(z,cinv(b-c)-cinv(a-c)),...
                  cmult(z,cinv(a-d)-cinv(b-d))]);
J = cjoin(cjoin(gR,-gI).',cjoin(gI,gR).').';
end
