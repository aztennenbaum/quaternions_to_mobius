function idx = constellation_permutation_idx(x)
% CONSTELLATION_PERMUTATION_IDX  Choose the canonical permutation index.
% Uses the same scoring logic as the prototype: smallest |2z-1|^2 in the
% packed lens coordinates after evaluating the 6 standard variants.
% Usage: idx = constellation_permutation_idx(x)
cr = zeros(size(x,1),6);
for k = 1:6
    y = crossratio_nocomplex(constellation_permute(x,k*ones(size(x,1),1)));
    cr(:,k) = (2*y(:,1)-1).^2 + y(:,2).^2;
end
[~,idx] = min(cr,[],2);
end
