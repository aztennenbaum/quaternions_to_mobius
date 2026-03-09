function q = q_max(q1,q2)
% Q_MAX  Return the row with the larger fifth-column score.
% Usage: q = q_max(q1,q2)
if q1(5)>q2(5), q = q1; else, q = q2; end
end
