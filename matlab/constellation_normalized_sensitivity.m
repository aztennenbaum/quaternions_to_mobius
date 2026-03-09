function y = constellation_normalized_sensitivity(x)
% CONSTELLATION_NORMALIZED_SENSITIVITY  Equal-noise cross-ratio sensitivity.
% x = [ax ay bx by cx cy dx dy].
% Usage: y = constellation_normalized_sensitivity(x)
a = x(:,1:2);
b = x(:,3:4);
c = x(:,5:6);
d = x(:,7:8);
y = sum([cinv(a-c)-cinv(a-d),...
         cinv(b-d)-cinv(b-c),...
         cinv(b-c)-cinv(a-c),...
         cinv(a-d)-cinv(b-d)].^2,2);
end
