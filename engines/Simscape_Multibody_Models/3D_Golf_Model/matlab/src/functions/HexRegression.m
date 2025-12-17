function [Coefficients] = HexRegression(x,y)
% HEXREGRESSION Performs a 6th order polynomial regression.
%   Coefficients = HexRegression(x,y) fits a 6th order polynomial
%   y = A*x^6 + B*x^5 + ... + G to the input data (x,y) and returns
%   the coefficients [A, B, C, D, E, F, G].
%
%   Input:
%       x - Vector of x data points.
%       y - Vector of y data points.
%
%   Output:
%       Coefficients - Vector of coefficients [A, B, C, D, E, F, G].

    arguments
        x {mustBeNumeric, mustBeVector}
        y {mustBeNumeric, mustBeVector}
    end

    % Define the type of function to fit using the fittype() function:
    Fit = fittype(@(A,B,C,D,E,F,G,x) ...
        A*x.^6 + B*x.^5 + C*x.^4 + D*x.^3 + E*x.^2 + F*x + G);

    % Define Starting Values for Coefficients
    x0 = [1 1 1 1 1 1 1];

    % Fit the function using the fit() function:
    [fitted_curve, ~] = fit(x, y, Fit, 'StartPoint', x0);

    % Save the coefficient values from the fitting:
    Coefficients = coeffvalues(fitted_curve);
end
