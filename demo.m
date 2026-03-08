% DEMO  Exercise the Mobius star tracker reference functions.
% This script runs small deterministic checks for the quaternion, Mobius,
% stereographic, cross-ratio, and estimator routines. It also checks that
% the closed-form M-TRIAD formula spans the same nullspace as m_triad().
% Usage: run('demo.m')

clear;
clc;
rng(1);
fprintf('Mobius star tracker demo\n');

%% Helpers and basic quaternion routines
x = sign_nz(0);
assert(x==1);
v = normalize_vec([3;4;0]);
assert(abs(norm(v)-1)<1e-12);
assert(norm(normalize([3;4;0]) - v)<1e-12);
S = skew([1;2;3]);
assert(all(size(S)==[3 3]));
q = normalize_vec([0.2;-0.3;0.1;0.9]);
A = q2a(q);
q_back = a2q(A);
assert(abs(abs(q_back.'*q)-1)<1e-10);
qp = quatmult(q,qinv2(q));
assert(norm(qp-[0;0;0;1])<1e-10);
qp2 = quatmult2(q,qinv2(q));
assert(norm(qp2-qp)<1e-12);
Phi = quatmat(q);
assert(all(size(Phi)==[4 4]));
mrp = q2mrp(q);
assert(numel(mrp)==3);
q_conj = qinv(q);
assert(abs(q_conj(4)-q(4))<1e-12);
assert(norm(q_conj(1:3)+q(1:3))<1e-12);
q_pick = q_max([1 0 0 0 1],[0 0 0 1 2]);
assert(q_pick(5)==2);
qn = normalize_q([q.' 0; 2*q.' 0]);
assert(abs(norm(qn)-1)<1e-12);
qs = sqrt_q([0;0;0;1]);
assert(abs(norm(qs)-1)<1e-12);

%% Mobius bridge and factorization
Mq = q2mobius(q);
q_from_M = mobius2q_norm(Mq);
assert(abs(abs(q_from_M.'*q)-1)<1e-10);
Ms = s2mobius(0.2+0.3i);
Md = division2mobius(2-1i);
Ml = lorentz_z(0.1);
assert(abs(det(normalize_m(Ms))-1)<1e-10);
assert(all(size(Md)==[2 2]));
assert(abs(lorentz_z_inv(Ml)-0.1)<1e-10);
vqs = [0.15 q.' 0.2 -0.1];
M = vqs2mobius(vqs);
vqs_back = mobius2vqs(M);
assert(abs(vqs_back(1)-vqs(1))<1e-8);
assert(abs(abs(vqs_back(2:5)*q)-1)<1e-8);
[U,Sb,Vb] = svd2b(M);
assert(norm(U*Sb*transpose(Vb)-normalize_m(M),'fro')<1e-8);
z0 = 0.1 + 0.2i;
z1 = mobius_apply(M,z0);
z2 = mobius_apply(mobius_inv(M),z1);
assert(abs(z2-z0)<1e-10);
vec4 = [0;0;0.3;1];
vec4b = lorentz_z_vec(0.1,vec4);
assert(numel(vec4b)==4);

%% Stereographic and camera model
s = normalize_vec([0.3;0.2;0.9]);
zs = stereoproj(s,1);
s_back = stereoprojinv(zs,1);
assert(norm(s_back-s)<1e-10);
zs2 = stereoproj2(0.3+0.2i,1.7);
assert(~isnan(zs2));
uv = [10; -7; 4; 2];
vv = [5; 3; -9; 1];
zuv = uvf_to_stereo(uv,vv,800);
[zuv1,dzdu,dzdv,dzdf] = uvf_jacobian(uv(1),vv(1),800);
assert(abs(zuv1-zuv(1))<1e-12);
assert(all(isfinite([real(dzdu) imag(dzdu) real(dzdv) imag(dzdv) real(dzdf) imag(dzdf)])));
sp_in_nc = [1 2 3 4];
sp_nc = stereoproj_nocomplex(sp_in_nc,5,1);
sp_inv_nc = stereoprojinv_nocomplex(sp_nc,5,1);
assert(all(size(sp_nc)==size(sp_in_nc)));
assert(all(size(sp_inv_nc)==size(sp_in_nc)));
assert(norm(sp_inv_nc-sp_in_nc)<1e-12);

%% Cross-ratio and canonicalization
z = [0.1+0.2i; 0.7+0.1i; -0.2+0.3i; 0.4-0.5i];
cr = crossratio(z(1),z(2),z(3),z(4));
orb = crossratio_orbit(cr);
assert(numel(orb)==6);
[cr_can,branches,ops] = canonicalize_crossratio(cr);
assert(~isempty(branches));
assert(iscell(ops));
xnc = [real(z(1)) imag(z(1)) real(z(2)) imag(z(2)) real(z(3)) imag(z(3)) real(z(4)) imag(z(4))];
cr_nc = crossratio_nocomplex(xnc);
assert(norm(cr_nc-[real(cr) imag(cr)])<1e-10);
Jc = crossratio_jacobian(z(1),z(2),z(3),z(4));
Jnc = crossratio_jacobian_nocomplex(xnc);
assert(all(size(Jc)==[1 4]));
assert(all(size(Jnc)==[2 8]));
Jdet = crossratio_jacobian_det_nocomplex(xnc);
assert(Jdet>=0);
sens = constellation_normalized_sensitivity(xnc);
assert(sens>=0);
perm3 = constellation_permute(xnc,3);
assert(numel(perm3)==8);
pidx = constellation_permutation_idx(xnc);
assert(pidx>=1 && pidx<=6);
score1 = crossratio_jacobian_score_nocomplex(xnc);
[score2,cr2] = crossratio_jacobian_score_nocomplex2(xnc);
assert(all(size(score1)==[1 6]));
assert(all(size(score2)==[1 6]));
assert(all(size(cr2)==[1 6]));
[cr_uvf,Juvf] = crossratio_uvf_jacobian(uv,vv,800);
assert(all(size(Juvf)==[2 9]));
assert(isfinite(real(cr_uvf)) && isfinite(imag(cr_uvf)));

%% Estimators
z_ref = [0.1+0.2i; -0.4+0.3i; 0.6-0.2i; -0.2-0.5i];
M_true = vqs2mobius([0.08 q.' 0.12 -0.15]);
w_obs = mobius_apply(M_true,z_ref);
M3 = m_triad(z_ref(1:3),w_obs(1:3));
M3f = m_triad_formula(z_ref(1:3),w_obs(1:3));
for k = 1:3
    assert(abs(mobius_apply(M3,z_ref(k))-w_obs(k))<1e-8);
    assert(abs(mobius_apply(M3f,z_ref(k))-w_obs(k))<1e-8);
end
[Mls,K,evals] = davenport_m_method(z_ref,w_obs,ones(4,1));
assert(all(size(K)==[4 4]));
assert(numel(evals)==4);
[Mfast,wvec,lam] = mobest(z_ref,w_obs,ones(4,1),6,1e-10);
assert(all(size(wvec)==[4 1]));
assert(isfinite(lam));
for k = 1:4
    assert(abs(mobius_apply(Mls,z_ref(k))-w_obs(k))<1e-6);
    assert(abs(mobius_apply(Mfast,z_ref(k))-w_obs(k))<1e-5);
end

%% M-TRIAD analytic equivalence proof
A3 = [z_ref(1:3) ones(3,1) -z_ref(1:3).*w_obs(1:3) -w_obs(1:3)];
m_formula = [M3f(1,1);M3f(1,2);M3f(2,1);M3f(2,2)];
res = A3*m_formula;
assert(norm(res)<1e-8);
N3 = null(A3);
m_null = N3(:,1);
scale = (m_null'*m_formula)/(m_null'*m_null);
assert(norm(m_formula-scale*m_null)<1e-8);
if license('test','Symbolic_Toolbox')
    syms z1 z2 z3 w1 w2 w3
    zs = [z1 z2 z3];
    ws = [w1 w2 w3];
    u = cross([sym(1) sym(1) sym(1)],zs.*ws);
    v = cross(zs,ws);
    mf = [u*ws.'; v*((zs.*ws).'); v*[1;1;1].'; u*zs.'];
    As = [zs.' sym(ones(3,1)) -(zs.*ws).' -ws.'];
    rs = simplify(As*mf);
    assert(all(isAlways(rs==0)));
    det_fact = simplify(det([mf(1) mf(2); mf(3) mf(4)]) / ...
        ((z1-z2)*(z1-z3)*(z2-z3)*(w1-w2)*(w1-w3)*(w2-w3)));
    assert(isAlways(det_fact==1) || isAlways(det_fact==-1));
    fprintf('Symbolic proof check passed\n');
else
    fprintf('Symbolic Math Toolbox not found; numeric proof check passed\n');
end

%% Low-level numeric helper for adjlower4
H = [4 1 0 0;1 3 0 0;0 0 2 0;0 0 0 5];
L = chol(H,'lower');
Adj = adjlower4(L);
assert(all(size(Adj)==[4 4]));

fprintf('All demo checks passed\n');
