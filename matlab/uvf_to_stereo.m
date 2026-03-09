function z = uvf_to_stereo(u,v,f)
% UVF_TO_STEREO  Pixel coordinates and focal length to stereographic z.
% u,v may be scalars or equal-sized arrays. f may be scalar or array.
% Usage: z = uvf_to_stereo(u,v,f)
L = sqrt(u.^2 + v.^2 + f.^2);
px = u./L;
py = v./L;
pz = f./L;
z = (px + 1i*py)./(1 + pz);
end
