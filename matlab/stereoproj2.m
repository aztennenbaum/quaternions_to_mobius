function z = stereoproj2(s,f)
% STEREOPROJ2  Scalar stereographic map from image-plane radius and focal.
% Usage: z = stereoproj2(s,f)
z = s./(sqrt(s.*conj(s) + f^2) + f);
end
