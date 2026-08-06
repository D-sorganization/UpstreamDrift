# Putting Kinematics and Kinetics — Public-Data Review

> Last reviewed: 2026-08-04
> Epic: #8345 (P4, issue #8348). Seeds model defaults and test bands for
> `src/shared/python/putting_dynamics/` (P2/P3) and the 3-D putt
> visualization (P1).

This review collects everything about putting stroke kinematics (how the
putter moves) and kinetics (what the impact delivers) that can be stated
from public, verifiable material, and derives the rest from first
principles. It follows the sourcing discipline of the AffineDrift
closure-rate literature dossier
(`content-development/technology-research/closure-rate-literature-dossier.md`
in D-sorganization/AffineDrift): every number carries a provenance class,
nothing is cited that was not verified against a fetched source, and
commonly repeated figures whose primary source could not be verified are
flagged as pointers, not citations.

## Provenance Classes

| Class | Meaning                                                                                                                |
| ----- | ---------------------------------------------------------------------------------------------------------------------- |
| SPEC  | Governing-body specification, verified against a fetched public document                                               |
| DERIV | First-principles derivation from SPEC inputs; shown in full here or in a cross-linked verified derivation              |
| DATA  | Openly published measurement data, verified against a fetched source (URL given; what it states quoted)                |
| CONV  | Fleet modelling convention / assumption; no public primary source verified; the model must mark it as an assumption    |
| PTR   | Unverified pointer: commonly referenced work whose numbers are NOT reproduced here because the source was not verified |

## 1. Verified Public Specifications

| Quantity                              | Value                                                                                                                                                                | Source (fetched and verified)                                                                                                                                                                                                                                                                                                             | Class |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| Ball mass (maximum)                   | 1.620 oz = 45.93 g                                                                                                                                                   | R&A Equipment Rules, Part 4: "The weight of the ball must not be greater than 1.620 ounces avoirdupois (45.93 g)" — <https://www.randa.org/en/roe/the-rules-of-equipment/part-4-conformance-of-balls>                                                                                                                                     | SPEC  |
| Ball diameter (minimum)               | 1.680 in = 42.67 mm                                                                                                                                                  | Same source: "The diameter of the ball must not be less than 1.680 inches (42.67 mm)"                                                                                                                                                                                                                                                     | SPEC  |
| Hole diameter                         | 4.25 in = 108 mm (lining outer diameter must not exceed the same)                                                                                                    | R&A _Rules of Golf_ (effective Jan 2023), Definitions, "Hole" (p. 222 of the official PDF): "The hole must be 4 1/4 inches (108 mm) in diameter and at least 4 inches (101.6 mm) deep" — <https://assets-us-01.kc-usercontent.com/c42c7bf4-dca7-00ea-4f2e-373223f80f76/48712d47-76dc-4fd3-add1-53972c021580/2023%20Rules%20of%20Golf.pdf> | SPEC  |
| Hole depth (minimum)                  | 4 in = 101.6 mm; lining sunk >= 1 in (25.4 mm) below the surface                                                                                                     | Same source, same definition                                                                                                                                                                                                                                                                                                              | SPEC  |
| Stimpmeter bar length                 | 36 in extruded aluminium bar with a V-shaped groove along its length                                                                                                 | USGA Stimpmeter Instructions (reprinted in _Hole Notes_, Aug 2000, pp. 24-35) — <https://archive.lib.msu.edu/tic/holen/article/2000aug24.pdf>                                                                                                                                                                                             | SPEC  |
| Stimpmeter ball-release notch         | 30 in from the tapered (ground) end                                                                                                                                  | Same source                                                                                                                                                                                                                                                                                                                               | SPEC  |
| Stimpmeter release angle              | approximately 20 degrees ("a ball will always be released and start to roll when the Stimpmeter is raised to an angle of approximately 20 degrees")                  | Same source                                                                                                                                                                                                                                                                                                                               | SPEC  |
| Stimpmeter groove geometry            | V-groove included angle 145 degrees, "supporting a golf ball at two points 1/2 [in] apart"; the rolling ball has "a slight overspin, which is thoroughly consistent" | Same source                                                                                                                                                                                                                                                                                                                               | SPEC  |
| Stimp measurement protocol            | 3 balls each way along the same line, stop pattern <= 8 in, series averaged; up/back difference > 18 in questionable                                                 | Same source                                                                                                                                                                                                                                                                                                                               | SPEC  |
| USGA green-speed chart (regular play) | slow < 7'6", medium 7'6"-8'6", fast > 8'6"                                                                                                                           | Same source                                                                                                                                                                                                                                                                                                                               | SPEC  |
| USGA green-speed chart (tournament)   | slow < 8'6", medium 8'6"-9'6", fast > 9'6"                                                                                                                           | Same source                                                                                                                                                                                                                                                                                                                               | SPEC  |
| Gravity g                             | 9.80665 m/s^2 (standard value used fleet-wide)                                                                                                                       | Defined constant                                                                                                                                                                                                                                                                                                                          | SPEC  |

Model constants derived from the regulatory limits: nominal ball radius
R = 21.335 mm (half the minimum conforming diameter) and nominal mass
m = 45.93 g (the maximum conforming mass). These are reproducible fleet
conventions, not measurements of every conforming ball: the rules specify
no maximum diameter and no minimum mass. The uniform-sphere inertia
I = (2/5) m R^2 is an additional idealisation, consistent with the
measured inertia ratio ~0.40 recorded in the AffineDrift
rotation-induced-spin supplement.

## 2. Stroke Kinematics

### 2.1 The Pendulum-Stroke Idealisation and What It Predicts

Treat the arm-shoulder-putter system as a pendulum of effective length L
(standard 35 in putter: L ~ 0.889 m) released from a backstroke arc
amplitude A. Small-angle SHM gives the bottom-of-arc (impact) speed

```
v_head = A * omega,   omega_free = sqrt(g / L) = 3.32 rad/s  (L = 0.889 m)
```

so the free-pendulum stroke gain is `v_head / A = 3.32 s^-1`. [DERIV]

This is the proxy currently implemented in Tools
`swing_sim.putting.impact.clubhead_speed_from_backstroke`. The verified
tour data below says the _real_ gain is about twice that — see §2.4.

### 2.2 Tempo: The 2:1 Ratio Is Measured and Derivable

Two independent verified sources:

**Marquardt (2007), "The SAM PuttLab: Concept and PGA Tour Data",**
_Annual Review of Golf Coaching_ — open PDF fetched and read:
<https://sam-academy.com/wp-content/uploads/2020/04/ARGC07-SAM-PuttLab-Concept-and-Tour-Data.pdf>.
99 male PGA Tour players, 9 tournaments 2003-2005, 7 recorded putts each on
a nominally straight, level 4 m practice-green putt (slightly shortened on
slower greens); ultrasound system, 210 Hz, resolution 0.1 mm / 0.1 degree.
Its Table 2 (group average, group SD,
intra-player consistency SD): [DATA]

| Parameter                 | Group average | Group SD | Consistency |
| ------------------------- | ------------- | -------- | ----------- |
| Backswing time (BST)      | 670 ms        | 90 ms    | 30 ms       |
| Time to impact (TI)       | 317 ms        | 35 ms    | 11 ms       |
| Forward swing time (FST)  | 820 ms        | 100 ms   | 45 ms       |
| Backswing rhythm (BST/TI) | 2.1           | 0.29     | 0.11        |
| Impact timing (TI/FST)    | 0.39          | 0.04     | 0.02        |
| Impact speed              | 1510 mm/s     | 119 mm/s | 45 mm/s     |
| Backswing length (BSL)    | 241 mm        | 38 mm    | 10 mm       |
| Path symmetry (BSL/FSL)   | 0.36          | 0.05     | 0.02        |

So the folklore "2:1 putting tempo" is a measured tour average of 2.1
(backswing time : time to impact), with tight intra-player consistency
(0.11) against loose inter-player spread (0.29): individual tempo is a
signature, while 2.1 is the mean for this tour sample. Impact occurs at 39% of
the forward swing — before peak speed, i.e. the putter is still (gently)
accelerating through the ball.

**Grober, "Resonance in Putting" (arXiv:0903.1762)** — open PDF fetched
and read: <https://arxiv.org/abs/0903.1762>. Models the stroke as a simple
harmonic oscillator (mass = system inertia, spring = gravity +
biomechanics) driven by an impulse at takeaway and an opposite impulse at
transition. The derivation gives `omega_0 * tau_b = pi/2` for the
backswing and `omega_0 * tau_d = pi/4` for the downswing **when the two
impulses have equal magnitude**, hence

```
tau_b / tau_d = 2   exactly, for equal impulses
```

and the paper shows this equal-impulse stroke minimises the relative error
of impact speed under uncorrelated random errors in the applied impulses.
The total stroke duration is independent of putt length (only the impulse
magnitudes scale), which matches the SAM observation that times are
player-constant while backswing length varies with distance. [DATA/DERIV]

Cross-check between the measured SAM intervals and Grober's model: from
SAM's tour averages,
`omega_0 = pi / (2 * 0.670 s) = 2.34 rad/s` (backswing condition) and
`omega_0 = pi / (4 * 0.317 s) = 2.48 rad/s` (downswing condition) — the
same oscillator frequency within 6% from two intervals in the same measured
stroke dataset. This is a consistency check, not an independent validation
dataset. [DERIV]

**Internal source discrepancy:** SAM Table 2 reports average impact speed
as 1510 mm/s, while the narrative on p. 111 reports 1570 mm/s. This review
uses the tabulated 1510 mm/s value for reproducibility and carries 1570 mm/s
as a same-source discrepancy; it does not average the two or silently choose
one as ground truth.

### 2.3 Face Rotation During the Putting Stroke

From the same verified SAM tour dataset (Marquardt 2007): [DATA]

| Parameter                                | Group average   | Group SD | Consistency |
| ---------------------------------------- | --------------- | -------- | ----------- |
| Face angle at address                    | 0.35 deg right  | 1.56 deg | 0.67 deg    |
| Face angle at impact                     | 0.30 deg right  | 0.59 deg | 0.70 deg    |
| Putter path direction at impact          | 0.80 deg left   | 2.24 deg | 0.83 deg    |
| Face on path at impact                   | 1.10 deg open   | 2.76 deg | 0.70 deg    |
| Face rotation in impact zone (+/- 10 cm) | 3.2 deg closing | 1.0 deg  | 0.34 deg    |
| Total forward-swing rotation             | ~10 deg closing | —        | —           |
| Dynamic shaft angle at impact            | 0.0 deg         | 1.2 deg  | 0.49 deg    |
| Vertical angle of attack (rise)          | 2.8 deg up      | 1.8 deg  | 0.60 deg    |

The paper's text adds: no tour player shows smooth zero rotation; the few
low-rotation players achieve it by re-opening the face through impact, not
by a more vertical swing plane; face rotation is the geometric consequence
of swinging on a tilted plane (the face stays square to the _plane_), the
same frame phenomenon as in the full swing.

**Relation to the org's closure-rate work.** The closure-rate dossier's
central finding — that "closure rate" numbers are meaningless without a
stated reference frame — applies unchanged at putting scale, roughly two
orders of magnitude down. A rough scale conversion from the verified SAM
numbers: 3.2 deg of closing over +/- 10 cm of path at ~1.5 m/s impact
speed is a face-to-target-line closing rate of order
`3.2 deg / (0.2 m / 1.5 m/s) ~ 25 deg/s` near impact, versus the
~1,800-3,600 deg/s global-frame closure rates of full driver swings
recorded in the dossier. Any putting closure-rate display in the P1
visualization must state its frame (face vs target line, as SAM measures)
exactly as the dossier demands for the full swing. [DERIV from DATA]

### 2.4 Stroke Length to Impact Speed: The Verified Gain

The SAM tour averages give a directly measured stroke gain:

```
k = impact speed / backswing length = 1.510 / 0.241 = 6.27 s^-1   [DATA]
```

Compare the two idealisations: [DERIV]

- free pendulum, `sqrt(g/L)` = 3.32 s^-1 — **under-predicts the tour gain
  by a factor ~1.9**;
- Grober's resonantly driven stroke is described in arXiv:1103.2827
  (abstract verified) as "a pendulum driven at twice its natural resonance
  frequency", i.e. gain `2 * sqrt(g/L)` = 6.64 s^-1 — within 6% of the
  measured 6.27 s^-1.

The tour stroke is a _driven_ pendulum, roughly twice as fast as free
gravity swinging. Model default: `k = 6.3 s^-1`, test band [3.3, 6.6]
(free-pendulum floor to exact-2x ceiling).

> Reconciliation note (cross-repo): Tools
> `swing_sim.putting.impact.clubhead_speed_from_backstroke` currently uses
> the free-pendulum gain `sqrt(g/L)` and therefore under-predicts tour
> head speed for a given backstroke by ~2x. Not wrong as a documented
> proxy, but `putting_dynamics` should default to the measured gain.

## 3. Impact Kinetics

### 3.1 The Collision Model (Chain Used Throughout)

Two-body impulse between a striking effective mass M_eff and the free ball
(mass m), face presenting effective loft delta: [DERIV]

```
v_ball = v_head * (1 + e) / (1 + m / M_eff) * cos(delta)   (normal chain)
```

with the tangential 2/7 rolling-cap partition for the small loft component
(the same sphere impulse partition as the full-swing impact model).
Vector consistency matters: for positive loft the backspin-producing
tangential impulse points down and forward along the face, so its vertical
component subtracts from the normal component. The interrupted Tools H3
implementation recomposes that impulse upward while assigning backspin;
`putting_dynamics` corrects the direction and regression-tests linear and
angular impulse consistency. At putter lofts (2-4 deg) the launch angle is a few
degrees, launch spin is a small backspin, and the ground phase starts
essentially sliding — the initial condition the roll model needs.

Speed ratio (ball/head) for the defaults below: `q = (1 + e) \* M / (M + m)

- cos^2(delta) = 1.57`with e = 0.78, M = 0.350 kg, m = 0.04593 kg,
delta = 3 deg. If the hands couple extra mass through the grip
(M_eff -> infinity) the ratio ceiling is`(1 + e) cos^2(delta) = 1.77`.

Consistency check against verified tour data: SAM's 4 m putt at measured
head speed 1.51 m/s implies ball speed 2.37 m/s (free head, q = 1.57) to
2.67 m/s (rigid-coupling ceiling). The §3.3 table says a 4 m putt needs
v0 = 2.54-2.76 m/s at stimp 12-10. The measured tour stroke sits inside
the derived band only toward the high-q end. Within this simplified model
that suggests grip/arm coupling raises effective striking mass above the
bare head, but unmodeled launch, turf, and environmental differences are
alternative explanations. Treat this as a model-calibration inference,
not a direct mass measurement. `putting_dynamics` defaults to the bare head
and exposes inertia and attachment quantities for sensitivity analysis.
[INFERENCE from DERIV vs DATA]

### 3.2 Putter Head Mass, COR, and Contact Time

| Quantity                      | Value / band                                                                                            | Provenance                                                                                                                                                                                                            | Class                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Putter head mass              | modern blades ~340-350 g; historical norm ~300-310 g; e.g. Tiger Woods' Scotty Newport 2 GSS head 326 g | GOLF.com equipment column (J. Wall), fetched: <https://golf.com/gear/putters/blade-putters-weight-fully-equipped-mailbag/>                                                                                            | DATA (trade press; manufacturer-attributed) |
| Putter head mass default      | 0.350 kg (blade), 0.360 kg (mallet)                                                                     | Fleet convention (Tools `MINIMAL_PUTTERS`), consistent with the DATA band above                                                                                                                                       | CONV                                        |
| Putter static loft            | 3-4 deg ("most of the putters we measured showed a static loft between 3 and 4 degrees")                | Marquardt 2007, verified (see §2.2)                                                                                                                                                                                   | DATA                                        |
| Putter face COR at putt speed | 0.78 default; 0.73-0.82 across milled/insert faces                                                      | Widely repeated value; used by Tools and this repo's `putter_stroke.py` / `contact.rs`; **no public primary measurement verified**                                                                                    | CONV                                        |
| Contact time                  | ~0.5 ms order of magnitude                                                                              | Commonly quoted for golf impacts (e.g. R. Cross's impact publications, physics.usyd.edu.au — not verified at putting speeds specifically); used by Tools as a documented assumption to neglect gravity during contact | PTR/CONV                                    |
| Putter head MOI               | no openly published per-model values verified                                                           | —                                                                                                                                                                                                                     | gap (§6)                                    |

### 3.3 Putter Speed vs. Putt Distance (Derived Table)

Chain: target distance D -> stimp deceleration a -> required launch speed
v0 (skid + roll) -> head speed v0/q. Inputs: a = 5.49/S m/s^2 (§4),
mu_k = 0.4 [CONV], q = 1.57 (free head, e = 0.78, M = 0.350 kg,
3 deg loft). Level green, dying-at-the-hole pace. [DERIV]

Required launch speed solves
`D = (12/49) v0^2/(mu_k g) + (25/49) v0^2/(2a)`.

| D (m) | Stimp 8: v0 -> v_head (m/s) | Stimp 10: v0 -> v_head | Stimp 12: v0 -> v_head |
| ----- | --------------------------- | ---------------------- | ---------------------- |
| 1     | 1.52 -> 0.97                | 1.38 -> 0.88           | 1.27 -> 0.81           |
| 2     | 2.15 -> 1.37                | 1.95 -> 1.24           | 1.80 -> 1.14           |
| 3     | 2.63 -> 1.68                | 2.39 -> 1.52           | 2.20 -> 1.40           |
| 4     | 3.04 -> 1.93                | 2.76 -> 1.76           | 2.54 -> 1.62           |
| 5     | 3.39 -> 2.16                | 3.08 -> 1.96           | 2.84 -> 1.81           |
| 6     | 3.72 -> 2.37                | 3.37 -> 2.15           | 3.11 -> 1.98           |
| 7     | 4.02 -> 2.56                | 3.65 -> 2.32           | 3.36 -> 2.14           |
| 8     | 4.29 -> 2.74                | 3.90 -> 2.48           | 3.59 -> 2.29           |
| 9     | 4.55 -> 2.90                | 4.13 -> 2.63           | 3.81 -> 2.43           |
| 10    | 4.80 -> 3.06                | 4.36 -> 2.78           | 4.02 -> 2.56           |

Reading: practical putts live in head speeds ~0.8-3 m/s; the SAM tour
reference (1.51 m/s at 4 m) sits in-band; faster greens need slower
strokes at every distance (the whole content of "pace adjustment").
For hole-capture pace add the §5 arrival-speed margin to D.

### 3.4 Grip Forces

Nothing verifiable found. Putting grip-force studies exist (pressure-mat
and instrumented-grip work is commonly referenced in coaching material),
but no openly fetchable source with usable numbers was verified for this
review. Grip force therefore has **no entry** in the defaults table and
the model must not pretend to know it; the stroke model's honest inputs
are the impulse pair (Grober) or the measured kinematics (SAM). [gap, §6]

## 4. Roll Kinematics (Cross-Linked, Already Verified)

The skid -> roll physics is derived in full in the org and is not
re-derived here:

- **AffineDrift PR #3778** (`articles/putting-roll-models.qmd`, merged):
  impact -> skid -> 5/7 transition -> pure roll -> capture, with the
  stimpmeter as deceleration instrument, slope/break as gravity
  components, and an assumption ledger. Companion numerical survey in
  `articles/green-simulation.qmd`.
- **Tools `swing_sim.putting`** (`impact.py`, `roll.py`, `green.py`, epic
  Tools#4125 H3): the same closed forms implemented with DbC contracts and
  tests; UD's `putting_dynamics` must reproduce these closed forms in its
  flat-uniform limit (epic #8345 acceptance).

Headline closed forms (all DERIV from SPEC inputs; proofs in the article):

| Result                        | Value                                            | Note                                                        |
| ----------------------------- | ------------------------------------------------ | ----------------------------------------------------------- |
| Slide-to-roll speed ratio     | v\* = (5/7)(v0 + (2/5) omega0 R)                 | = (5/7) v0 for spinless launch; property of I = (2/5) m R^2 |
| Skid distance                 | (12/49) v0^2 / (mu_k g)                          | ends at t\* = 2 v0 / (7 mu_k g)                             |
| Skid share of putt length     | (24/25) a / (mu_k g) -> 10-15%                   | launch-speed independent                                    |
| Stimp -> deceleration         | `a = v_s^2 / (2 * 0.3048 * S)`                   | the stimp number is a deceleration in disguise              |
| Runaway grade                 | a/g = mu_r (7.0% / 5.6% / 4.7% at stimp 8/10/12) | downhill putt cannot stop beyond it                         |
| Dying-putt fractional break   | g sin(beta) / a                                  | launch-speed independent upper bound                        |
| Capture ceiling (dead centre) | v_cap = D_h sqrt(g / 2R) = 1.64 m/s              | pure SPEC geometry                                          |
| Effective hole half-width     | b_max(v) = sqrt(r_h^2 - v^2 R / 2g)              | 51/43/22 mm at 0.5/1.0/1.5 m/s                              |

### 4.1 Stimpmeter Release Speed — Geometry Now Verified, One Correction

The release-speed derivation (energy conservation down the groove,
enlarged effective inertia from the two-line contact) is in the AffineDrift
article. This review adds the **verified** groove geometry from the USGA
instructions (§1): included angle 145 deg, contact points 1/2 in apart.
That fixes the effective rolling radius with no free parameter:

```
contact half-separation = R sin(17.5 deg) = 6.4 mm  (2 x 6.4 = 12.8 mm ~ 1/2 in, consistent)
r_c = R cos(17.5 deg) = 0.954 R
v_release = sqrt(2 g L sin(20 deg) / (1 + (2/5)/(0.954)^2)) = 1.88 m/s
a = v_s^2 / (2 * 0.3048 * S) = 5.82 / S  m/s^2
```

The AffineDrift article and Tools `roll.py` use r_c ~ 0.87 R, giving the
conventional v_s = 1.83 m/s (6 ft/s) and a = 5.49/S. With the verified
145-degree spec the geometric value is r_c = 0.954 R and v_s = 1.88 m/s
(a = 5.82/S) — a 3% speed / 6% deceleration difference. Both lie inside
the commonly quoted 1.8-1.9 m/s band; the flat-ramp limit is 1.91 m/s.
[DERIV from SPEC]

**Default decision:** keep `v_s = 1.83 m/s` and `a = 5.49/S` as the fleet
default for cross-repo parity (AffineDrift article, Tools tests), and
carry the verified-geometry value 1.88 m/s / 5.82/S as the documented
upper edge of the test band. Reconciling Tools/AffineDrift to the
145-degree geometry is flagged as follow-up; it shifts every stimp-derived
number by ~6% in a known direction.

## 5. Model Defaults and Test Bands for `putting_dynamics`

| Parameter                                    | Default                                                                                                             | Test band                                                            | Class                                |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------ |
| Ball mass m                                  | 0.04593 kg nominal                                                                                                  | maximum conforming mass; actual mass may be lower                    | SPEC/CONV                            |
| Ball radius R                                | 0.021335 m nominal                                                                                                  | minimum conforming radius; actual radius may be larger               | SPEC/CONV                            |
| Ball inertia                                 | (2/5) m R^2                                                                                                         | inertia ratio 0.40 +/- 0.02                                          | DERIV/CONV                           |
| Hole diameter D_h                            | 0.10795 m                                                                                                           | exact                                                                | SPEC                                 |
| Hole depth                                   | >= 0.1016 m                                                                                                         | exact (minimum)                                                      | SPEC                                 |
| Gravity g                                    | 9.80665 m/s^2                                                                                                       | exact                                                                | SPEC                                 |
| Stimpmeter release speed v_s                 | 1.83 m/s (fleet parity)                                                                                             | [1.83, 1.91]; verified-geometry point 1.88                           | DERIV                                |
| Stimp -> rolling deceleration                | a = 5.49/S m/s^2                                                                                                    | [5.49/S, 5.82/S]; mu_r = a/g in 0.047-0.070 for stimp 8-12           | DERIV                                |
| Stimp range to support                       | 8-12 (USGA chart: tournament fast > 9'6")                                                                           | 6-14 hard limits                                                     | SPEC/CONV                            |
| Sliding friction mu_k (ball-turf)            | 0.40                                                                                                                | [0.3, 0.5] sensitivity band                                          | CONV (assumption — no public source) |
| Slide-to-roll transition                     | v\* = (5/7)(v0 + (2/5) omega0 R)                                                                                    | exact closed form; parameter-free test pin                           | DERIV                                |
| Skid share of putt                           | — (emergent)                                                                                                        | 10-15% at mu_k = 0.4, stimp 8-12                                     | DERIV                                |
| Putter head mass                             | 0.350 kg blade / 0.360 kg mallet                                                                                    | [0.30, 0.37] published band                                          | DATA/CONV                            |
| Putter effective striking mass M_eff         | head mass (free-head default)                                                                                       | [head mass, +inf); tour data favours the upper half (§3.1)           | CONV                                 |
| Putter loft                                  | 3.0 deg                                                                                                             | 2-4 deg static (tour static lofts 3-4 deg)                           | DATA/CONV                            |
| Putter face COR e                            | 0.78                                                                                                                | [0.73, 0.82]                                                         | CONV                                 |
| Contact time scale                           | 0.5 ms (neglect gravity during contact)                                                                             | order of magnitude only                                              | PTR/CONV                             |
| Ball/head speed ratio q                      | 1.57 (free head, 3 deg loft)                                                                                        | [1.4, 1.77] (COR band to rigid-coupling ceiling)                     | DERIV                                |
| Stroke tempo BST/TI                          | 2.1                                                                                                                 | 2.1 +/- 0.29 (group); +/- 0.11 within-player                         | DATA                                 |
| Backswing time                               | 670 ms                                                                                                              | +/- 90 ms                                                            | DATA                                 |
| Time to impact                               | 317 ms                                                                                                              | +/- 35 ms                                                            | DATA                                 |
| Impact timing TI/FST                         | 0.39                                                                                                                | +/- 0.04                                                             | DATA                                 |
| Stroke gain k = v_head/BSL                   | 6.3 s^-1                                                                                                            | [3.3, 6.6] (free pendulum to 2x resonance)                           | DATA/DERIV                           |
| Face rotation, impact zone (+/- 10 cm)       | 3.2 deg closing                                                                                                     | +/- 1.0 deg (group); 0.34 deg within-player                          | DATA                                 |
| Face angle at impact                         | 0.3 deg (of target)                                                                                                 | SD 0.59 deg                                                          | DATA                                 |
| Rise (attack) angle                          | 2.8 deg up                                                                                                          | +/- 1.8 deg                                                          | DATA                                 |
| Capture ceiling                              | 1.64 m/s dead-centre                                                                                                | parameter-free test pin; b_max(v) curve per §4                       | DERIV                                |
| Reference tour datum (validation comparison) | nominal 4 m level putt (shortened on slower greens): head 1.510 m/s from Table 2, BSL 241 mm, BST 670 ms, TI 317 ms | +/- 1 group SD each; same-source narrative impact speed is 1.570 m/s | DATA                                 |

Suggested contract tests: (a) flat-uniform surface reproduces the shared
closed forms (5/7 pin, skid share band, a = 5.49/S); (b) the SAM tour row
is used as a validation comparison across the documented effective-mass
range, not forced as an exact calibration target; (c) the full-chord
capture ceiling is exact to the geometry; (d) a stimpmeter simulation
launched _rolling at r_c contact_ (over-spinning by R/r_c) settles from
above, not below (sign test).

## 6. Gaps — No Public Verifiable Source

The model must mark these as assumptions (CONV) or leave them
unparameterised:

1. **Ball-turf sliding friction mu_k.** No openly published measurement
   verified. Default 0.40 is a fleet assumption; sensitivity 0.3-0.5
   moves the stimp-10 skid share 9.7-15.2%.
2. **Putter face COR primary data.** 0.78 is repeated across the fleet
   and trade literature but no primary measurement source was verified.
3. **Contact time at putting speeds.** Order 0.5 ms by analogy with
   full-speed golf impacts (R. Cross's open publications are the natural
   primary source — pointer, not verified for putts).
4. **Grip forces during the stroke.** Nothing verifiable found (§3.4).
5. **Putter head MOI values.** No openly published per-model numbers
   verified; treat MOI as a free parameter in the P3 collision model.
6. **Skid-phase field data** (high-speed footage statistics of skid
   length/transition on real greens) — the 5/7 law is parameter-free
   theory; no public measurement dataset verified to pin mu_k.
7. **Green-condition statistics** (spatial variability of stimp/friction
   on real greens) — the P2 stochastic fields have no public calibration
   dataset; seeded perturbations are modelling choices.

## 7. Unverified Pointers (Flagged; Numbers Not Used)

- R. Cross, physics of ball-implement impacts (open PDFs at
  physics.usyd.edu.au/~cross) — natural primary source for COR/contact
  time; not verified against a fetched copy for putting-speed values.
- D. Pelz, _Putt Like the Pros_ / putting research (cited by Marquardt as
  [3] for the 82%/18% face/path direction weighting and the "Perfy"
  robot) — the weighting is quoted here only as what Marquardt states.
- Blast Motion / TrueRoll coaching statistics on tempo (repeat the 2:1
  ratio; vendor blogs, no methodology published).
- Grober, "Resonance as a Means of Distance Control in Putting"
  (arXiv:1103.2827) — abstract verified ("driven at twice its natural
  resonance frequency"); full-text numbers not extracted; the companion
  paper arXiv:0903.1762 was fetched and used instead.
- Academic putting-biomechanics literature (e.g. Delay et al., Karlsen's
  green-reading work, Hurrion's putter-fitting studies) — commonly
  referenced; none fetched; no numbers reproduced.

## 8. Sources Actually Verified (Fetched During This Review)

1. R&A Equipment Rules, Part 4 (the ball) —
   <https://www.randa.org/en/roe/the-rules-of-equipment/part-4-conformance-of-balls>
2. R&A _Rules of Golf_ effective Jan 2023, official PDF, Definitions:
   "Hole" (p. 222) —
   <https://assets-us-01.kc-usercontent.com/c42c7bf4-dca7-00ea-4f2e-373223f80f76/48712d47-76dc-4fd3-add1-53972c021580/2023%20Rules%20of%20Golf.pdf>
3. USGA Stimpmeter Instructions (full reprint, _Hole Notes_ Aug 2000) —
   <https://archive.lib.msu.edu/tic/holen/article/2000aug24.pdf>
4. C. Marquardt, "The SAM PuttLab: Concept and PGA Tour Data", _Annual
   Review of Golf Coaching_ 2007 —
   <https://sam-academy.com/wp-content/uploads/2020/04/ARGC07-SAM-PuttLab-Concept-and-Tour-Data.pdf>
5. R. D. Grober, "Resonance in Putting", arXiv:0903.1762 —
   <https://arxiv.org/abs/0903.1762>
6. R. D. Grober, "Resonance as a Means of Distance Control in Putting",
   arXiv:1103.2827 (abstract only) — <https://arxiv.org/abs/1103.2827>
7. J. Wall, "Why blade putters have increased in weight", GOLF.com —
   <https://golf.com/gear/putters/blade-putters-weight-fully-equipped-mailbag/>

In-org verified derivations cross-linked: AffineDrift PR #3778
(`putting-roll-models.qmd`, `green-simulation.qmd`); Tools#4125 H3
(`src/shared/python/swing_sim/putting/`); this repo's
`docs/physics/GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md` (source-map pattern
this document follows).
