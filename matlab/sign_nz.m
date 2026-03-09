function y = sign_nz(x)
% SIGN_NZ  Sign with sign(0)=1.
% Usage: y = sign_nz(x)
y = sign(x);
y(y==0) = 1;
end
