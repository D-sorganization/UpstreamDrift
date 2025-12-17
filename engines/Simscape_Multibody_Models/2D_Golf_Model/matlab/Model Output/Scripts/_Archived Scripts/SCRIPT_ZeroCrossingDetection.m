%Generate a table with all zeroes the same size as BASEQ
t_zero = zeros(size(BASEQ.EquivalentMidpointCoupleGlobal));
for i=1:numel(BASEQ.EquivalentMidpointCoupleGlobal)
    j = BASEQ.EquivalentMidpointCoupleGlobal(i); 
    % Need both the index in the idx/t_zero vector, and the f/t vector
    t_zero(i) = interp1(BASEQ.EquivalentMidpointCoupleGlobal(j:j+1),BASEQ.Time(j:j+1),0.0,'linear');
end
