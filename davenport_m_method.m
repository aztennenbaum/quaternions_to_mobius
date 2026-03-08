function [M,K,evals] = davenport_m_method(z,w,alpha)
% DAVENPORT_M_METHOD  Weighted least-squares Mobius fit for n>=3 pairs.
% The cost is m'*K*m where m=[a b c d]. The minimizer is the smallest
% eigenvector of the Hermitian 4x4 normal matrix K.
% Usage: [M,K,evals] = davenport_m_method(z,w,alpha)
z = z(:);
w = w(:);
n = numel(z);
if nargin<3 || isempty(alpha), alpha = ones(n,1); end
alpha = alpha(:);
if numel(w)~=n || numel(alpha)~=n, error('davenport_m_method:size','size mismatch'); end
K = zeros(4,4);
for k = 1:n
    zk = z(k);
    wk = w(k);
    H = [abs(zk)^2, conj(zk), -conj(wk)*abs(zk)^2, -conj(wk)*conj(zk);...
         zk, 1, -conj(wk)*zk, -conj(wk);...
         -wk*abs(zk)^2, -wk*conj(zk), abs(wk)^2*abs(zk)^2, abs(wk)^2*conj(zk);...
         -wk*zk, -wk, abs(wk)^2*zk, abs(wk)^2];
    K = K + alpha(k)*H;
end
K = (K + K')./2;
[V,D] = eig(K);
[evals,idx] = sort(real(diag(D)),'ascend');
m = V(:,idx(1));
M = normalize_m([m(1) m(2);m(3) m(4)]);
end
