function z = crossratio(a,b,c,d)
% CROSSRATIO  Ordered projective invariant of four complex points.
% Usage: z = crossratio(a,b,c,d)
z = ((a-c).*(b-d))./((b-c).*(a-d));
end
