function y = crossratio_jacobian_det_nocomplex(x)
% CROSSRATIO_JACOBIAN_DET_NOCOMPLEX  det(J*J') without forming J explicitly.
% Useful for sensitivity ranking in the canonicalization search.
% Usage: y = crossratio_jacobian_det_nocomplex(x)
a = x(:,1:2);
b = x(:,3:4);
c = x(:,5:6);
d = x(:,7:8);
z2 = sum(crossratio_nocomplex(x).^2,2);
s2 = sum([cinv(a-c)-cinv(a-d),...
          cinv(b-d)-cinv(b-c),...
          cinv(b-c)-cinv(a-c),...
          cinv(a-d)-cinv(b-d)].^2,2);
y = (z2.*s2).^2;
end
