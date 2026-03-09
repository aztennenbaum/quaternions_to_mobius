function [U,S,V] = svd2b(X)
% SVD2B  Closed-form 2x2 complex polar/SVD helper for det(X)=1.
% Returns U,V unitary and S Hermitian positive with det(S)=1.
% The factors satisfy X = U*S*V.'' in this implementation.
% Usage: [U,S,V] = svd2b(X)
X = normalize_m(X);
P = X*X';
sgn = sign_nz(real(P(1,1)-P(2,2)));
tr = trace(P);
root = sqrt(tr^2 - 4);
Sp2 = [tr-sgn*root 0;0 tr+sgn*root]./2;
U = sgn*(P-Sp2)*[1 0;0 -1];
if norm(U,'fro')<100*eps
    U = eye(2);
else
    U = normalize_m(U);
end
Sp = diag(sqrt(real(diag(Sp2))));
S = inv(Sp);
V = (Sp*U'*X).';
V = normalize_m(V);
end
