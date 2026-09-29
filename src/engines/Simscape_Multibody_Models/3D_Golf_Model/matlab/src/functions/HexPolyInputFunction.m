function [InputFunctionOutput] = HexPolyInputFunction(A,B,C,D,E,F,G,x)
% HEXPOLYINPUTFUNCTION Compute the value of a 6th order polynomial.
%
%   InputFunctionOutput = HexPolyInputFunction(A, B, C, D, E, F, G, x)
%   calculates the polynomial:
%   y = A*x^6 + B*x^5 + C*x^4 + D*x^3 + E*x^2 + F*x + G
%
%   Inputs:
%       A, B, C, D, E, F, G - Polynomial coefficients (scalar or array)
%       x - Input value(s)
%
%   Outputs:
%       InputFunctionOutput - Result of the polynomial evaluation
%
%   Example:
%       y = HexPolyInputFunction(1, 0, 0, 0, 0, 0, 0, 2); % y = 2^6 = 64
%
%   See also: POLYVAL

    arguments
        A {mustBeNumeric}
        B {mustBeNumeric}
        C {mustBeNumeric}
        D {mustBeNumeric}
        E {mustBeNumeric}
        F {mustBeNumeric}
        G {mustBeNumeric}
        x {mustBeNumeric}
    end

    InputFunctionOutput = A.*x.^6 + B.*x.^5 + C.*x.^4 + D.*x.^3 + E.*x.^2 + F.*x + G;
end
