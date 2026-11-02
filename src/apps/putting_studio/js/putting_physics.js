/**
 * Temporary, hand-kept putting physics for the standalone HTML studio
 * (epic #14141, step 8a). THIS COPY IS TEMPORARY -- the single source
 * of truth is src/shared/python/putting_dynamics/; see ../SUNSET.md.
 * Classic script (no ES modules; opens from file://). Pure functions
 * only -- no reads of a shared mutable global.
 *
 * Alignment vs. the Gemini source: friction uses the USGA law
 * mu = v_release^2/(2 g S); the roll ODE drops the source's "(5/7) g"
 * factor for both friction and slope, matching solver._derivative()
 * (plain g for both), so a 1.83 m/s release rolls exactly S*0.3048 m
 * flat (Stimpmeter definition); restitution is 0.78 (PUTTER_FACE_COR);
 * the launch ratio uses the Python off-centre effective-mass model
 * instead of a Gaussian heuristic; capture uses
 * captureSpeedMps(entryOffsetM), a chord generalisation of
 * capture_speed_mps(); radii/gravity match Python exactly for
 * cross-language parity. Skid multiplier, gear-twist/lip-out and turf
 * chatter stay uncited (out of scope).
 */
(function (root) {
  "use strict";

  // Match src/shared/python/putting_dynamics + core/physics_constants.py.
  const GRAVITY_M_S2 = 9.80665;
  const STIMP_RELEASE_SPEED_MPS = 1.83;
  const STIMPMETER_RELEASE_ANGLE_DEG = 20.0;
  // Exact USGA value 0.053975; matched to the Python-rounded
  // solver.HOLE_RADIUS_M for the 1e-9 cross-language parity check.
  const HOLE_RADIUS_M = 0.054;
  const GOLF_BALL_RADIUS_M = 0.021335;
  const GOLF_BALL_MASS_KG = 0.04593;
  const PUTTER_FACE_COR = 0.78;
  const _FEET_TO_METERS = 0.3048;
  const _MIN_STIMP_FT = 3.0;
  const _MAX_STIMP_FT = 16.0;
  const _MAX_IMPACT_OFFSET_M = 0.08;
  // Uncited heuristics inherited from the source (unchanged, out of scope).
  const _SKID_MU_MULTIPLIER = 1.85;
  const _CAPTURE_SPEED_DAMPING = 0.6;
  const _LIP_OUT_DEFLECTION_SCALE = 0.38;
  const _CLUB_SPEED_EFFICIENCY_DERATE = 0.96;
  const _TYPICAL_PUTTER_HEAD_MASS_KG = 0.36;
  const _TYPICAL_PUTTER_MOI_KG_M2 = 4.5e-4;
  const _CATALOGUE_MOI_G_CM2_PER_KG_M2 = 1.0e7;
  function assertFinite(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new TypeError(name + " must be a finite number");
    }
  }
  function assertRange(value, min, max, name) {
    assertFinite(value, name);
    if (value < min || value > max) {
      throw new RangeError(name + " must be in [" + min + ", " + max + "]");
    }
  }

  /** Deterministic pseudo-random value in [0, 1). @throws {TypeError} */
  function seededRandom(seed) {
    assertFinite(seed, "seed");
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  }
  /** Approx. N(0,1), Box-Muller on two seeded uniforms. @throws {TypeError} */
  function gaussianRand(seed) {
    assertFinite(seed, "seed");
    const u1 = Math.max(1e-6, seededRandom(seed * 17.3 + 1.1));
    const u2 = seededRandom(seed * 31.7 + 2.9);
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }
  /**
   * Height and gradient of the undulating green at (x, y).
   * @returns {{z:number, dz_dx:number, dz_dy:number}}
   * @throws {TypeError|RangeError}
   */
  function evalUndulatingGreen(x, y, holeX, upPct, crossPct) {
    assertFinite(x, "x");
    assertFinite(y, "y");
    assertRange(holeX, 1e-6, 1000, "holeX");
    const sx0 = (upPct || 0.0) / 100.0,
      sy0 = (crossPct || 0.0) / 100.0;
    const scaleUnd = Math.abs(upPct) < 0.01 && Math.abs(crossPct) < 0.01 ? 0.0 : 1.0;
    const tierC = 0.52 * holeX,
      tierW = Math.max(0.45, 0.18 * holeX);
    const tierH = 0.016 * scaleUnd * Math.sign(upPct || 1.0);
    const uTier = Math.max(-20.0, Math.min(20.0, (x - tierC) / tierW));
    const sig = 1.0 / (1.0 + Math.exp(-uTier));
    const zTier = tierH * sig,
      dzTierDx = (tierH / tierW) * sig * (1.0 - sig);

    const m1x = 0.36 * holeX, m1y = -0.18, m1A = 0.011 * scaleUnd;
    const s1x = Math.max(0.45, 0.24 * holeX), s1y = 0.42;
    const g1 = m1A * Math.exp(-0.5 * (((x - m1x) / s1x) ** 2 + ((y - m1y) / s1y) ** 2));
    const g1x = -g1 * ((x - m1x) / (s1x * s1x)), g1y = -g1 * ((y - m1y) / (s1y * s1y));
    const m2x = 0.74 * holeX, m2y = 0.15, m2A = -0.009 * scaleUnd;
    const s2x = Math.max(0.45, 0.22 * holeX), s2y = 0.38;
    const g2 = m2A * Math.exp(-0.5 * (((x - m2x) / s2x) ** 2 + ((y - m2y) / s2y) ** 2));
    const g2x = -g2 * ((x - m2x) / (s2x * s2x)), g2y = -g2 * ((y - m2y) / (s2y * s2y));
    return {
      z: sx0 * x + sy0 * y + zTier + g1 + g2,
      dz_dx: sx0 + dzTierDx + g1x + g2x,
      dz_dy: sy0 + g1y + g2y,
    };
  }

  /**
   * Seeded micro-roughness lateral acceleration perturbation [m/s^2].
   * @returns {{ax:number, ay:number}} @throws {TypeError}
   */
  function sampleTurfMicroRoughness(x, y, speed, roughnessMm, seed) {
    assertFinite(x, "x");
    assertFinite(y, "y");
    assertFinite(speed, "speed");
    assertFinite(roughnessMm, "roughnessMm");
    assertFinite(seed, "seed");
    if (!roughnessMm || roughnessMm <= 0.001 || speed <= 0.01) return { ax: 0.0, ay: 0.0 };
    const amp = 0.0038 * (roughnessMm / 0.5);
    const p1 = 19.7 * x + 23.3 * y + 0.73 * seed;
    const p2 = 37.1 * x - 29.9 * y + 1.37 * seed;
    const p3 = 53.3 * (x + y) - 2.11 * seed;
    const dsx = amp * (0.6 * Math.sin(p1) + 0.4 * Math.cos(p2));
    const dsy = amp * (0.6 * Math.cos(p1) + 0.4 * Math.sin(p3));
    const gain = Math.min(1.3, 0.45 + 0.5 / Math.max(0.25, speed));
    return { ax: -GRAVITY_M_S2 * dsx * gain, ay: -GRAVITY_M_S2 * dsy * gain };
  }

  /**
   * USGA law: mu = v_release^2/(2 g S). Mirrors friction.stimp_to_rolling_mu.
   * @param {number} stimpFt In [3, 16]. @throws {TypeError|RangeError}
   */
  function stimpToRollingMu(stimpFt) {
    assertRange(stimpFt, _MIN_STIMP_FT, _MAX_STIMP_FT, "stimpFt");
    return STIMP_RELEASE_SPEED_MPS ** 2 / (2.0 * GRAVITY_M_S2 * stimpFt * _FEET_TO_METERS);
  }

  /**
   * Chord generalisation of the Holmes/Penner capture bound: the ball
   * sweeps chord 2*sqrt(r^2-b^2) through the hole at offset b; free
   * fall must drop it half a diameter within that chord, giving
   * v_capture(b) = chord(b) * sqrt(g / (2 R)). At b=0 this equals the
   * Python capture_speed_mps() exactly.
   * @param {number} entryOffsetM abs() <= HOLE_RADIUS_M.
   * @throws {TypeError|RangeError}
   */
  function captureSpeedMps(entryOffsetM) {
    assertFinite(entryOffsetM, "entryOffsetM");
    if (!(Math.abs(entryOffsetM) < HOLE_RADIUS_M)) {
      throw new RangeError("entry offset must lie inside the hole");
    }
    const chordM = 2.0 * Math.sqrt(HOLE_RADIUS_M ** 2 - entryOffsetM * entryOffsetM);
    return chordM * Math.sqrt(GRAVITY_M_S2 / (2.0 * GOLF_BALL_RADIUS_M));
  }

  /**
   * Putter effective mass at an off-centre strike: 1/m_eff = 1/M + d^2/I.
   * Mirrors collision.effective_head_mass. @throws {TypeError|RangeError}
   */
  function effectiveHeadMass({ headMassKg, moiKgM2, impactOffsetM }) {
    assertRange(headMassKg, 0.1, 1.0, "headMassKg");
    assertRange(moiKgM2, 1e-5, 1e-2, "moiKgM2");
    assertRange(Math.abs(impactOffsetM), 0.0, _MAX_IMPACT_OFFSET_M, "abs(impactOffsetM)");
    return 1.0 / (1.0 / headMassKg + (impactOffsetM * impactOffsetM) / moiKgM2);
  }

  /**
   * Ball/head launch ratio (1+e)*M/(M+m). Mirrors
   * collision.centre_strike_speed_ratio; the source's "1.595" is this
   * ratio at one (e,M,m) triple, valid only for a centred strike --
   * pass effectiveHeadMass()'s result as headMassKg off-centre.
   * @throws {TypeError|RangeError}
   */
  function centreStrikeSpeedRatio({
    headMassKg,
    ballMassKg = GOLF_BALL_MASS_KG,
    cor = PUTTER_FACE_COR,
  }) {
    assertRange(headMassKg, 0.0, 1.0, "headMassKg");
    assertRange(ballMassKg, 0.0, 1.0, "ballMassKg");
    assertRange(cor, 0.0, 1.0, "cor");
    return ((1.0 + cor) * headMassKg) / (headMassKg + ballMassKg);
  }

  /**
   * Integrate one putt with classic RK4 from strike (or stimp-ramp
   * release) to rest, capture or lip-out; see the destructured
   * defaults below for the full option list.
   * @returns {object} pts, finalX, finalY, holed, lippedOut, skidDist,
   *   v0, missDistCm, holeX, launchDeg, faceErrDeg, pathErrDeg.
   * @throws {TypeError|RangeError}
   */
  function simulateSinglePuttRK4(options) {
    const {
      vClubMps, aimDeg, pathErrDeg, faceErrDeg, strikeMm = 0.0,
      headMassKg = _TYPICAL_PUTTER_HEAD_MASS_KG,
      moiKgM2 = _TYPICAL_PUTTER_MOI_KG_M2,
      stimpFt, upPct = 0.0, crossPct = 0.0, trialSeed = 0.0, distFt,
      roughnessMm = 0.35, isStimpRamp = false,
      ballMassKg = GOLF_BALL_MASS_KG, cor = PUTTER_FACE_COR,
    } = options;
    assertFinite(vClubMps, "vClubMps"); assertFinite(aimDeg, "aimDeg");
    assertFinite(pathErrDeg, "pathErrDeg"); assertFinite(faceErrDeg, "faceErrDeg");
    assertRange(Math.abs(strikeMm), 0.0, 80.0, "abs(strikeMm)");
    assertRange(stimpFt, _MIN_STIMP_FT, _MAX_STIMP_FT, "stimpFt");
    assertRange(distFt, 0.5, 200.0, "distFt");
    assertRange(roughnessMm, 0.0, 5.0, "roughnessMm");
    const holeX = distFt * _FEET_TO_METERS, holeY = 0.0, holeR = HOLE_RADIUS_M;
    const v0 = isStimpRamp
      ? vClubMps
      : vClubMps *
        centreStrikeSpeedRatio({
          headMassKg: effectiveHeadMass({ headMassKg, moiKgM2, impactOffsetM: strikeMm / 1000.0 }),
          ballMassKg,
          cor,
        });
    const moiCatalogueGCm2 = moiKgM2 * _CATALOGUE_MOI_G_CM2_PER_KG_M2;
    const gearTwistDeg = isStimpRamp ? 0.0 : strikeMm * 0.085 * (4500.0 / moiCatalogueGCm2);
    const totalFaceDeg = aimDeg + faceErrDeg + gearTwistDeg, totalPathDeg = aimDeg + pathErrDeg;
    const launchDeg = isStimpRamp ? aimDeg : 0.84 * totalFaceDeg + 0.16 * totalPathDeg;
    const launchRad = (launchDeg * Math.PI) / 180.0;
    let x = 0.0, y = 0.0;
    let vx = v0 * Math.cos(launchRad), vy = v0 * Math.sin(launchRad);
    const muRoll = stimpToRollingMu(stimpFt);
    const dt = 0.012;
    const pts = [{ x: 0, y: 0, t: 0, mode: isStimpRamp ? "rolling" : "sliding" }];
    let t = 0.0, skidDist = 0.0, holed = false, lippedOut = false;
    const targetSkidDist = isStimpRamp ? 0.0 : 0.14 * holeX * (v0 / 1.75);
    while (t < 7.5) {
      const speed = Math.hypot(vx, vy);
      if (speed < 0.012) break;
      const distMoved = Math.hypot(x, y);
      const mode = !isStimpRamp && distMoved < targetSkidDist ? "sliding" : "rolling";
      const mu = mode === "sliding" ? muRoll * _SKID_MU_MULTIPLIER : muRoll;
      const grad = evalUndulatingGreen(x, y, holeX, upPct, crossPct);
      const chatter = sampleTurfMicroRoughness(x, y, speed, roughnessMm, trialSeed);
      const ax = -mu * GRAVITY_M_S2 * (vx / speed) - GRAVITY_M_S2 * grad.dz_dx + chatter.ax;
      const ay = -mu * GRAVITY_M_S2 * (vy / speed) - GRAVITY_M_S2 * grad.dz_dy + chatter.ay;
      const nx = x + vx * dt + 0.5 * ax * dt * dt, ny = y + vy * dt + 0.5 * ay * dt * dt;
      if (mode === "sliding") skidDist += Math.hypot(nx - x, ny - y);
      x = nx; y = ny; vx += ax * dt; vy += ay * dt; t += dt;
      if (!isStimpRamp) {
        const rx = x - holeX, ry = y - holeY;
        const dHole = Math.hypot(rx, ry);
        if (dHole <= holeR && !lippedOut) {
          const vHatX = vx / speed, vHatY = vy / speed;
          const bRaw = rx * vHatY - ry * vHatX;
          const b = Math.min(holeR, Math.abs(bRaw));
          if (speed <= captureSpeedMps(b)) {
            holed = true; x = holeX; y = holeY;
            pts.push({ x, y, t, mode: "stopped" });
            break;
          }
          lippedOut = true;
          const sign = bRaw >= 0 ? 1.0 : -1.0, bRatio = b / holeR;
          const defAng = sign * _LIP_OUT_DEFLECTION_SCALE * bRatio;
          const cA = Math.cos(defAng), sA = Math.sin(defAng);
          const vxNew = (cA * vx - sA * vy) * _CAPTURE_SPEED_DAMPING;
          const vyNew = (sA * vx + cA * vy) * _CAPTURE_SPEED_DAMPING;
          vx = vxNew; vy = vyNew;
        }
      }
      pts.push({ x, y, t, mode });
      if (x > holeX * 1.45 + 1.5) break;
    }
    const missDistCm = Math.hypot(x - holeX, y - holeY) * 100.0;
    return {
      pts, finalX: x, finalY: y, holed, lippedOut, skidDist,
      faceErrDeg, pathErrDeg, launchDeg, v0, missDistCm, holeX,
    };
  }

  /**
   * Required club speed for a target hole-entry speed, inverting the
   * rolling-friction law. @throws {TypeError|RangeError}
   */
  function computeClubSpeedForCapture({
    distFt, entrySpeedMps, stimpFt, upPct = 0.0,
    headMassKg = _TYPICAL_PUTTER_HEAD_MASS_KG,
    ballMassKg = GOLF_BALL_MASS_KG, cor = PUTTER_FACE_COR,
  }) {
    assertRange(distFt, 0.5, 200.0, "distFt");
    assertRange(entrySpeedMps, 0.0, 10.0, "entrySpeedMps");
    assertRange(stimpFt, _MIN_STIMP_FT, _MAX_STIMP_FT, "stimpFt");
    assertFinite(upPct, "upPct");
    const distM = distFt * _FEET_TO_METERS;
    const muRoll = stimpToRollingMu(stimpFt);
    const aEff = Math.max(0.08, GRAVITY_M_S2 * (muRoll + upPct / 100.0));
    const vBallReq = Math.sqrt(entrySpeedMps * entrySpeedMps + 2.0 * aEff * distM);
    const ratio = centreStrikeSpeedRatio({ headMassKg, ballMassKg, cor });
    return vBallReq / (ratio * _CLUB_SPEED_EFFICIENCY_DERATE);
  }

  const api = {
    GRAVITY_M_S2, STIMP_RELEASE_SPEED_MPS, STIMPMETER_RELEASE_ANGLE_DEG, HOLE_RADIUS_M,
    GOLF_BALL_RADIUS_M, GOLF_BALL_MASS_KG, PUTTER_FACE_COR, seededRandom, gaussianRand,
    evalUndulatingGreen, sampleTurfMicroRoughness, stimpToRollingMu, captureSpeedMps,
    effectiveHeadMass, centreStrikeSpeedRatio, simulateSinglePuttRK4, computeClubSpeedForCapture,
  };

  root.PuttingPhysics = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
