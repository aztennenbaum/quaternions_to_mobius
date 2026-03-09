function y = cmult(x1,x2)
% CMULT  Multiply packed complex pairs column-wise.
% Usage: y = cmult(x1,x2)
[x1r,x1i] = csplit(x1);
[x2r,x2i] = csplit(x2);
y = cjoin(x1r.*x2r - x1i.*x2i,x1r.*x2i + x1i.*x2r);
end
