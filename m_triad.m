function M = m_triad(z,w)
% M_TRIAD  Exact 3-point Mobius fit from a 1D nullspace.
% Build A = [z 1 -z.*w -w] from (a*z+b)-w*(c*z+d)=0. For three generic
% pairs, null(A) is 1D. Reshape its basis vector into [a b; c d] and
% normalize det(M)=1.
% Usage: M = m_triad(z,w)
z = z(:);
w = w(:);
if numel(z)~=3 || numel(w)~=3, error('m_triad:size','need 3 pairs'); end
A = [z ones(3,1) -z.*w -w];
N = null(A);
if isempty(N), error('m_triad:rank','nullspace is empty'); end
if size(N,2)~=1, error('m_triad:rank','expected 1D nullspace'); end
m = N(:,1);
M = normalize_m([m(1) m(2);m(3) m(4)]);
end
