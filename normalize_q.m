function q = normalize_q(Q)
% NORMALIZE_Q  Pick the largest candidate row and normalize it.
% Prototype helper kept for old q_seq_rot code.
% Usage: q = normalize_q(Q)
if size(Q,2)>4, Q = Q(:,1:4); end
[~,i] = max(sum(abs(Q).^2,2));
q = Q(i,:).';
q = q./(sign_nz(q(4))*norm(q));
end
