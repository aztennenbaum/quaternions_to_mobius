function y = crossratio_jacobian_score_nocomplex(x)
% CROSSRATIO_JACOBIAN_SCORE_NOCOMPLEX  Sensitivity score for all 6 variants.
% Usage: y = crossratio_jacobian_score_nocomplex(x)
y = zeros(size(x,1),6);
for k = 1:6
    y(:,k) = constellation_normalized_sensitivity(constellation_permute(x,k*ones(size(x,1),1)));
end
end
