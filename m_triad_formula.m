function M = m_triad_formula(z,w)
% M_TRIAD_FORMULA  Closed-form 3-point Mobius fit from the prototype notes.
% For three pairs (z_k,w_k), the exact map satisfies A*m=0 with
% A = [z 1 -z.*w -w]. The notes give an explicit nullspace generator built
% from 3-vector cross products. This function evaluates that formula,
% reshapes it into [a b; c d], and normalizes det(M)=1.
% Usage: M = m_triad_formula(z,w)
z = z(:).';
w = w(:).';
if numel(z)~=3 || numel(w)~=3, error('m_triad_formula:size','need 3 pairs'); end
u = cross([1 1 1],z.*w);
v = cross(z,w);
m = [u*w.';...
     v*(z.*w).';...
     v*[1;1;1];...
     u*z.'];
M = normalize_m([m(1) m(2);m(3) m(4)]);
end
