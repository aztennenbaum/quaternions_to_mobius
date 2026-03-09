function z = stereoproj_nocomplex(s,f,zsign)
% STEREOPROJ_NOCOMPLEX  Real-pair stereographic map with focal length f.
% s packs x and y columns as [x1 y1 x2 y2 ...].
% Usage: z = stereoproj_nocomplex(s,f,zsign)
if nargin<3, zsign = 1; end
[sx,sy] = csplit(s);
v = zsign.*sqrt(sx.^2 + sy.^2 + f.^2);
z = cjoin(sx./(f+v),zsign.*sy./(f+v));
end
