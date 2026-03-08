function [sens,cr] = crossratio_jacobian_score_nocomplex2(x)
% CROSSRATIO_JACOBIAN_SCORE_NOCOMPLEX2  Sensitivity and size for 6 variants.
% Usage: [sens,cr] = crossratio_jacobian_score_nocomplex2(x)
sens = zeros(size(x,1),6);
cr = zeros(size(x,1),6);
for k = 1:6
    y = constellation_permute(x,k*ones(size(x,1),1));
    sens(:,k) = constellation_normalized_sensitivity(y);
    cr(:,k) = sum(crossratio_nocomplex(y).^2,2);
end
end
