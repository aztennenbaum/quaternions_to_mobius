function Mi = mobius_inv(M)
% MOBIUS_INV  Inverse up to irrelevant projective scale.
% Usage: Mi = mobius_inv(M)
Mi = [M(2,2) -M(1,2);-M(2,1) M(1,1)];
end
