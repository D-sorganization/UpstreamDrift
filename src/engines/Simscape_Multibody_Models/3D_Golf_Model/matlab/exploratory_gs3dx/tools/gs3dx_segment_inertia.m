function [m, c, I] = gs3dx_segment_inertia(mass, len, com_fraction, gyration)
%GS3DX_SEGMENT_INERTIA  Segment mass, centre of mass and principal moments from radii of gyration (#10979).
%
%   [M, C, I] = GS3DX_SEGMENT_INERTIA(MASS, LEN, COM_FRACTION, GYRATION)
%   returns, for a body segment of MASS kg and length LEN m:
%
%     M   MASS (kg)
%     C   centre of mass, COM_FRACTION * LEN (m from the proximal end)
%     I   principal moments about the centre of mass (1x3, kg*m^2),
%         MASS * (GYRATION * LEN).^2, in the order of GYRATION:
%         [sagittal transverse longitudinal]
%
%   COM_FRACTION and GYRATION are the de Leva (1996) fractions of segment
%   length in GS3DX_ANTHROPOMETRY (.com, .gyration).  Pure function.

    arguments
        mass (1,1) double {mustBePositive, mustBeFinite}
        len (1,1) double {mustBePositive, mustBeFinite}
        com_fraction (1,1) double {mustBeInRange(com_fraction, 0, 1)}
        gyration (1,3) double {mustBePositive, mustBeLessThan(gyration, 1)}
    end
    m = mass;
    c = com_fraction * len;
    I = mass * (gyration * len) .^ 2;
end
