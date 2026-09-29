function coefficients = generateRandomCoefficients(num_coefficients)
    % External function for generating random coefficients - can be used in parallel processing
    % This function doesn't rely on handles

    % Generate random coefficients with reasonable ranges for golf swing parameters
    % These ranges are based on typical golf swing polynomial coefficients

    % Different ranges for different coefficient types (A, B, C, D, E, F, G)
    % A (t^6): Large range for major motion
    % B (t^5): Large range for major motion
    % C (t^4): Medium range for control
    % D (t^3): Medium range for control
    % E (t^2): Small range for fine control
    % F (t^1): Small range for fine control
    % G (constant): Small range for offset

    coefficients = zeros(1, num_coefficients);

    for i = 1:num_coefficients
        coeff_type = mod(i-1, 7) + 1; % A=1, B=2, C=3, D=4, E=5, F=6, G=7

        switch coeff_type
            case {1, 2} % A, B - Large range
                coefficients(i) = (rand() - 0.5) * 2000; % -1000 to 1000
            case {3, 4} % C, D - Medium range
                coefficients(i) = (rand() - 0.5) * 1000; % -500 to 500
            case {5, 6} % E, F - Small range
                coefficients(i) = (rand() - 0.5) * 200;  % -100 to 100
            case 7 % G - Very small range
                coefficients(i) = (rand() - 0.5) * 50;   % -25 to 25
        end
    end
end
