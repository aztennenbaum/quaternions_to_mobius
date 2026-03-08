function [z,branches,ops] = canonicalize_crossratio(z,tol)
% CANONICALIZE_CROSSRATIO  Map to the canonical lens domain.
% Ordered rules:
%   1) if real(z)<1/2, z = 1-z
%   2) if abs(z)>1,    z = 1/z
%   3) if imag(z)<0,   z = 1-z
% Near |z|=1 or |1-z|=1, also return neighboring branches.
% Usage: [z,branches,ops] = canonicalize_crossratio(z,tol)
if nargin<2, tol = 1e-8; end
ops = {};
if real(z)<0.5, z = 1-z; ops{end+1} = '1-z'; end
if abs(z)>1, z = 1./z; ops{end+1} = 'inv'; end
if imag(z)<0, z = 1-z; ops{end+1} = '1-z'; end
branches = z;
if abs(abs(z)-1)<tol, branches(end+1,1) = 1./z; end
if abs(abs(1-z)-1)<tol, branches(end+1,1) = 1-z; end
branches = unique(round([real(branches) imag(branches)]./tol).*tol,'rows');
branches = branches(:,1) + 1i*branches(:,2);
end
