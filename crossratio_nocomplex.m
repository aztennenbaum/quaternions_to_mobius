function z = crossratio_nocomplex(x)
% CROSSRATIO_NOCOMPLEX  Cross-ratio on packed real-imag columns.
% x = [ax ay bx by cx cy dx dy].
% Usage: z = crossratio_nocomplex(x)
a = x(:,1:2);
b = x(:,3:4);
c = x(:,5:6);
d = x(:,7:8);
z = cmult(cmult(a-c,b-d),cinv(cmult(b-c,a-d)));
end
