function [M,K,evals] = davenport_m_method(z,w,alpha)
% DAVENPORT_M_METHOD  Weighted least-squares Mobius fit for n>=3 pairs.
% The cost is m'*K*m where m=[a b c d]. The minimizer is the smallest
% eigenvector of the Hermitian 4x4 normal matrix K.
% Each pair contributes the weighted row A_k = [z_k 1 -z_k*w_k -w_k],
% so K = sum_k alpha_k * (A_k'' * A_k).
% Usage: [M,K,evals] = davenport_m_method(z,w,alpha)
z = z(:);
w = w(:);
n = numel(z);
if nargin<3 || isempty(alpha), alpha = ones(n,1); end
alpha = alpha(:);
if numel(w)~=n || numel(alpha)~=n, error('davenport_m_method:size','size mismatch'); end
K = zeros(4,4);
for k = 1:n
    row = [z(k) 1 -z(k)*w(k) -w(k)];
    K = K + alpha(k)*(row'*row);
end
K = (K + K')./2;
[V,D] = eig(K);
[evals,idx] = sort(real(diag(D)),'ascend');
m = V(:,idx(1));
M = normalize_m([m(1) m(2);m(3) m(4)]);
end
