function [M,w,lambda] = mobest(z,wobs,alpha,niter,shift)
% MOBEST  Fast inverse-iteration solver for the M-method eigenvector.
% Uses (K+shift*I)^(-1) repeatedly, normalizes each step, then reshapes.
% Usage: [M,w,lambda] = mobest(z,wobs,alpha,niter,shift)
if nargin<4 || isempty(niter), niter = 6; end
if nargin<5 || isempty(shift), shift = 1e-10; end
[~,K] = davenport_m_method(z,wobs,alpha);
A = K + shift*eye(4);
w = [1;0;0;1];
w = w./norm(w);
for k = 1:niter
    w = A\w;
    w = w./norm(w);
end
lambda = real((w'*K*w)/(w'*w));
M = normalize_m([w(1) w(2);w(3) w(4)]);
end
