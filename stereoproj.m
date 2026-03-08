function z = stereoproj(s,zsign)
% STEREOPROJ  Unit-vector to stereographic coordinate.
% s is 3xN. zsign=1 uses the north-pole chart, zsign=-1 the south chart.
% Usage: z = stereoproj(s,zsign)
if nargin<2, zsign = 1; end
v = zsign.*sqrt(sum(s.^2,1));
z = (s(1,:) + 1i*zsign.*s(2,:))./(s(3,:) + v);
end
