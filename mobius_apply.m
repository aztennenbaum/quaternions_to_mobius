function z = mobius_apply(M,z)
% MOBIUS_APPLY  Evaluate z -> (a*z+b)/(c*z+d).
% Usage: z = mobius_apply(M,z)
z = (M(1,1).*z + M(1,2))./(M(2,1).*z + M(2,2));
end
