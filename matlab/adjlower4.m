function A = adjlower4(L)
% ADJLOWER4  Adjugate of a 4x4 lower-triangular matrix.
% The prototype used this to avoid an explicit inverse inside MOBEST.
% Usage: A = adjlower4(L)
a1 = L(1,1);
b1 = L(2,1); b2 = L(2,2);
c1 = L(3,1); c2 = L(3,2); c3 = L(3,3);
d1 = L(4,1); d2 = L(4,2); d3 = L(4,3); d4 = L(4,4);
A = [b2*c3*d4,0,0,0;...
    -b1*c3*d4,a1*c3*d4,0,0;...
     b1*c2*d4-b2*c1*d4,-a1*c2*d4,a1*b2*d4,0;...
     b1*c3*d2-b1*c2*d3+b2*c1*d3-b2*c3*d1,...
     a1*c2*d3-a1*c3*d2,-a1*b2*d3,a1*b2*c3];
end
