# Zero-Twist Geometry and Carry Derivatives

For λ=L−ℓ, before nominal yaw,

`dn/dθ = s×n = (cosβ sinλ, sinβ cosL, −cosβ cosλ)`.

Yaw rotates both vectors together and leaves the derivatives of reported face
and loft invariant. Thus

`df/dθ = −sinβ cosL/cosλ`, `dλ/dθ = −cosβ`,

and `dλ/df = cotβ cosλ/cosL`.

These dimensionless ratios are the same for radians/radian and degrees/degree.
At zero twist, central-face loft SD contributed by a small shaft-twist SD is
`cosβ σθ`, independent of forward lean in this rigid construction. The associated
face SD is `sinβ cosL/cos(L−ℓ) σθ`, which decreases across these preset lean values.
The loft change per unit face-angle change consequently increases. This is a
first-order result at the center-face normal; finite noise, curved-face strike
offsets, or stochastic lean introduce additional effects.

All 27 club/lean/nominal-face combinations (face −3/0/+3°), checked with centered
differences ±0.0001° and ±0.00001°, agreed within 4.73e−10 deg/deg (declared tolerance
1e−8). The yaw-invariant analytical values are:

| Club | Lean | Face / Twist | Loft / Twist | Loft / Face |
| --- | ---: | ---: | ---: | ---: |
| driver | 0° | -0.852640 | -0.522499 | 0.612801 |
| driver | 5° | -0.842894 | -0.522499 | 0.619886 |
| driver | 10° | -0.839687 | -0.522499 | 0.622254 |
| seven_iron | 0° | -0.891007 | -0.453990 | 0.509525 |
| seven_iron | 5° | -0.844570 | -0.453990 | 0.537540 |
| seven_iron | 10° | -0.808584 | -0.453990 | 0.561464 |
| pitching_wedge | 0° | -0.898794 | -0.438371 | 0.487733 |
| pitching_wedge | 5° | -0.827278 | -0.438371 | 0.529896 |
| pitching_wedge | 10° | -0.771745 | -0.438371 | 0.568026 |

## Local Carry Response

Saved pilot nominal speed/face/path are held fixed while shaft twist is perturbed
by ±0.01° and ±0.005°. All 45 supported nominal deliveries were evaluated at flight
dt 0.02 and 0.005 s; three unavailable club/lean/curve combinations remain explicit.
The table shows the straight-pattern response using the smaller twist and flight
steps. X is downrange; Y is positive right. These are approximate local numerical
slopes, not empirically calibrated player responses.

| Club | Lean | Downrange m / Twist Degree | Rightward m / Twist Degree |
| --- | ---: | ---: | ---: |
| driver | 0° | -4.62 | -6.27 |
| driver | 5° | -10.52 | -2.98 |
| seven_iron | 0° | 1.17 | -2.58 |
| seven_iron | 5° | 1.00 | -2.98 |
| seven_iron | 10° | 0.77 | -3.43 |
| pitching_wedge | 0° | 1.15 | -1.11 |
| pitching_wedge | 5° | 1.12 | -1.34 |
| pitching_wedge | 10° | 1.04 | -1.58 |

Positive twist closes and delofts the face. In these straight presets it moves
carry left, but driver carry shortens whereas iron/PW carry lengthens. Geometry
alone therefore gives no universal long-left/short-right guarantee. The derivative
combines launch speed, elevation, spin and flight response, conditional on the
chosen baseline and head assumptions.

Carry slope estimates have residual discretization sensitivity: maximum difference
between the two twist steps was 0.04651 m/degree at dt 0.02 s and 0.01804 m/degree
at dt 0.005 s. Landing interpolation can affect small finite differences. These
slopes support qualitative local signs; they have no separately asserted rigorous
derivative-error bound. The analytical geometry result is independently checked
and does not depend on flight interpolation.

Receipt SHA256: `449d7cab351f810896216cbfe03e64f813867d7f433cc01b450fe7aea86857a1`.
