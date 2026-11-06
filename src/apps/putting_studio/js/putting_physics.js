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
 * cross-language parity; the skid length comes from the core table in
 * js/core_impact.js (#14647), passed in as skidTable; a core green preset
 * (js/core_greens.js, #14648) can replace the slider green as coreGreen; the
 * skid ends at the core roll-start speed from the same table (#14649).
 * Skid multiplier, gear-twist/lip-out and turf chatter stay uncited.
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
   * Height and gradient of a core green preset (js/core_greens.js, #14648)
   * at green coordinates (gx, gy): the clamped bilinear patch of
   * surfaces.HeightField.elevation() and gradient().
   * @returns {{z:number, dz_dx:number, dz_dy:number}} @throws {TypeError}
   */
  function coreGreenAt(green, gx, gy) {
    assertFinite(gx, "gx");
    assertFinite(gy, "gy");
    const h = green.heights_mm, s = green.spacing_m;
    const fx = (gx - green.origin_m[0]) / s, fy = (gy - green.origin_m[1]) / s;
    const ix = Math.min(Math.max(Math.floor(fx), 0), h[0].length - 2);
    const iy = Math.min(Math.max(Math.floor(fy), 0), h.length - 2);
    const tx = Math.min(Math.max(fx - ix, 0.0), 1.0), ty = Math.min(Math.max(fy - iy, 0.0), 1.0);
    const h00 = h[iy][ix] / 1e3, h10 = h[iy][ix + 1] / 1e3;
    const h01 = h[iy + 1][ix] / 1e3, h11 = h[iy + 1][ix + 1] / 1e3;
    return {
      z: h00 * (1 - tx) * (1 - ty) + h10 * tx * (1 - ty) + h01 * (1 - tx) * ty + h11 * tx * ty,
      dz_dx: ((h10 - h00) * (1 - ty) + (h11 - h01) * ty) / s,
      dz_dy: ((h01 - h00) * (1 - tx) + (h11 - h10) * tx) / s,
    };
  }

  /**
   * Uphill and cross slope [%] of a core green preset at the ball (putt
   * origin) for a hole holeX metres away: its dz/dx and dz/dy there (#14703).
   * @returns {{upPct:number, crossPct:number}} @throws {TypeError}
   */
  function coreSlopePctAtBall(green, holeX) {
    const g = evalGreen(0.0, 0.0, holeX, 0.0, 0.0, green);
    return { upPct: 100.0 * g.dz_dx, crossPct: 100.0 * g.dz_dy };
  }

  /**
   * The studio green at putt coordinates (x, y): a core preset centred on
   * the putt midpoint when coreGreen is given, else the slider green.
   * @returns {{z:number, dz_dx:number, dz_dy:number}} @throws {TypeError|RangeError}
   */
  function evalGreen(x, y, holeX, upPct, crossPct, coreGreen) {
    if (!coreGreen) return evalUndulatingGreen(x, y, holeX, upPct, crossPct);
    return coreGreenAt(coreGreen, x - holeX / 2.0, y);
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
   * Column `col` of the core skid table at a ball speed, interpolated in
   * [ballSpeedMps, skidM, rollStartSpeedMps] rows that rise from [0, 0, 0]
   * (js/core_impact.js, from putting_dynamics.simulate_strike). Past the
   * last row the last segment is extended. @throws {TypeError|RangeError}
   */
  function coreSkidTableAt(skidTable, ballSpeedMps, col) {
    if (!Array.isArray(skidTable) || skidTable.length < 2 || skidTable[0].length <= col) {
      throw new TypeError("skidTable must hold at least two [speed, skid, roll speed] rows");
    }
    assertRange(ballSpeedMps, 0.0, 20.0, "ballSpeedMps");
    let i = 1;
    while (i < skidTable.length - 1 && skidTable[i][0] < ballSpeedMps) i++;
    const v0 = skidTable[i - 1][0], v1 = skidTable[i][0];
    const s0 = skidTable[i - 1][col], s1 = skidTable[i][col];
    return s0 + ((s1 - s0) * (ballSpeedMps - v0)) / (v1 - v0);
  }

  /** Core skid distance [m] for a ball speed. @throws {TypeError|RangeError} */
  function coreSkidDistanceM(skidTable, ballSpeedMps) {
    return coreSkidTableAt(skidTable, ballSpeedMps, 1);
  }

  /** Core ball speed [m/s] when the skid ends and pure roll starts (#14649). */
  function coreRollStartSpeedMps(skidTable, ballSpeedMps) {
    return coreSkidTableAt(skidTable, ballSpeedMps, 2);
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
      roughnessMm = 0.35, isStimpRamp = false, skidTable = null, coreGreen = null,
      withHole = true,
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
    // A toe strike (+) opens the face, as in the core strike() (#14647);
    // studio angles are positive to the left, so the twist is negative.
    const gearTwistDeg = isStimpRamp ? 0.0 : -strikeMm * 0.085 * (4500.0 / moiCatalogueGCm2);
    const totalFaceDeg = aimDeg + faceErrDeg + gearTwistDeg, totalPathDeg = aimDeg + pathErrDeg;
    const launchDeg = isStimpRamp ? aimDeg : 0.84 * totalFaceDeg + 0.16 * totalPathDeg;
    const launchRad = (launchDeg * Math.PI) / 180.0;
    let x = 0.0, y = 0.0;
    let vx = v0 * Math.cos(launchRad), vy = v0 * Math.sin(launchRad);
    const muRoll = stimpToRollingMu(stimpFt);
    const dt = 0.012;
    const pts = [{ x: 0, y: 0, t: 0, mode: isStimpRamp ? "rolling" : "sliding" }];
    let t = 0.0, skidDist = 0.0, holed = false, lippedOut = false;
    const targetSkidDist = isStimpRamp ? 0.0 : coreSkidDistanceM(skidTable, v0);
    const rollStartSpeed = isStimpRamp ? 0.0 : coreRollStartSpeedMps(skidTable, v0);
    let skidEnded = isStimpRamp;
    while (t < 7.5) {
      let speed = Math.hypot(vx, vy);
      if (speed < 0.012) break;
      const distMoved = Math.hypot(x, y);
      const mode = !isStimpRamp && distMoved < targetSkidDist ? "sliding" : "rolling";
      if (mode === "rolling" && !skidEnded) {
        // The skid ends at the core's roll-start speed (about 5/7 of launch).
        skidEnded = true;
        vx *= rollStartSpeed / speed; vy *= rollStartSpeed / speed;
        speed = rollStartSpeed;
        if (speed < 0.012) break;
      }
      const mu = mode === "sliding" ? muRoll * _SKID_MU_MULTIPLIER : muRoll;
      const grad = evalGreen(x, y, holeX, upPct, crossPct, coreGreen);
      const chatter = sampleTurfMicroRoughness(x, y, speed, roughnessMm, trialSeed);
      const ax = -mu * GRAVITY_M_S2 * (vx / speed) - GRAVITY_M_S2 * grad.dz_dx + chatter.ax;
      const ay = -mu * GRAVITY_M_S2 * (vy / speed) - GRAVITY_M_S2 * grad.dz_dy + chatter.ay;
      const nx = x + vx * dt + 0.5 * ax * dt * dt, ny = y + vy * dt + 0.5 * ay * dt * dt;
      if (mode === "sliding") skidDist += Math.hypot(nx - x, ny - y);
      x = nx; y = ny; vx += ax * dt; vy += ay * dt; t += dt;
      if (!isStimpRamp && withHole) {
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

  // Search bounds and tolerances of green_reading.solve_aim.
  const _READ_SPEED_MIN_MPS = 0.05, _READ_SPEED_MAX_MPS = 8.0;
  const _READ_SPEED_ITERATIONS = 30, _READ_HEADING_ITERATIONS = 12;
  const _READ_MAX_HEADING_STEP_RAD = 0.25, _READ_TOL_M = 5e-3;

  /**
   * The core optimum read (putting_dynamics.green_reading.solve_aim) on the
   * studio putt: the head speed whose cup-free roll stops pastHoleM beyond
   * the hole (doubling bracket, then bisection) and the heading whose path
   * crosses the hole (clamped secant, safeguarded by bisection). Centred,
   * error-free strike. `options` takes the simulateSinglePuttRK4 green,
   * Stimp, distance and table options.
   * @returns {{clubSpeedMps:number, aimDeg:number, holed:boolean, converged:boolean}}
   * @throws {TypeError|RangeError}
   */
  function solveAimCore(options, pastHoleM = 0.3) {
    assertRange(pastHoleM, 1e-3, 5.0, "pastHoleM");
    const holeX = options.distFt * _FEET_TO_METERS;
    const run = (heading, speed, withHole) => simulateSinglePuttRK4({
      ...options, vClubMps: speed, aimDeg: (heading * 180.0) / Math.PI, pathErrDeg: 0,
      faceErrDeg: 0, strikeMm: 0, roughnessMm: 0, trialSeed: 0, isStimpRamp: false, withHole,
    });
    const along = (heading, speed) => {
      const r = run(heading, speed, false);
      return [r, r.finalX * Math.cos(heading) + r.finalY * Math.sin(heading)];
    };
    const pace = (heading, target) => {
      let lo = _READ_SPEED_MIN_MPS, hi = lo;
      let [r, at] = along(heading, hi);
      while (at < target && hi < _READ_SPEED_MAX_MPS) {
        lo = hi; hi = Math.min(2.0 * hi, _READ_SPEED_MAX_MPS);
        [r, at] = along(heading, hi);
      }
      let speed = hi;
      for (let i = 0; i < _READ_SPEED_ITERATIONS; i++) {
        if (Math.abs(at - target) <= _READ_TOL_M || hi <= lo) break;
        speed = 0.5 * (lo + hi);
        [r, at] = along(heading, speed);
        if (at < target) lo = speed; else hi = speed;
      }
      return [r, speed];
    };
    // Left offset where the path reaches the hole's distance (else at rest).
    const miss = (heading) => {
      const [r, speed] = pace(heading, holeX + pastHoleM);
      const pt = r.pts.find((p) => p.x >= holeX);
      return [pt ? pt.y : r.finalY, speed];
    };
    let heading = 0.0, prev = null;
    let [curMiss, speed] = miss(heading);
    const bracket = {};
    for (let i = 0; i < _READ_HEADING_ITERATIONS; i++) {
      bracket[curMiss > 0.0] = heading;
      if (Math.abs(curMiss) <= _READ_TOL_M || (prev && prev[1] === curMiss)) break;
      let step = curMiss / holeX;
      if (prev) step = (curMiss * (heading - prev[0])) / (curMiss - prev[1]);
      let next = heading - Math.max(-_READ_MAX_HEADING_STEP_RAD, Math.min(_READ_MAX_HEADING_STEP_RAD, step));
      if (true in bracket && false in bracket) {
        const lo = Math.min(bracket[true], bracket[false]), hi = Math.max(bracket[true], bracket[false]);
        if (!(lo < next && next < hi)) next = 0.5 * (bracket[true] + bracket[false]);
      }
      prev = [heading, curMiss];
      heading = next;
      [curMiss, speed] = miss(heading);
    }
    const final = run(heading, speed, true);
    return {
      clubSpeedMps: speed, aimDeg: (heading * 180.0) / Math.PI,
      holed: final.holed, converged: Math.abs(curMiss) <= _READ_TOL_M,
    };
  }

  /**
   * Required club speed for a target hole-entry speed, inverting the core
   * skid table and the rolling-friction law. @throws {TypeError|RangeError}
   */
  function computeClubSpeedForCapture({
    distFt, entrySpeedMps, stimpFt, upPct = 0.0, skidTable,
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
    // Bisect the launch speed whose core roll-start speed, after the core
    // skid, still carries entrySpeedMps to the hole; the excess rises with it.
    const excess = (v) => {
      const roll = coreRollStartSpeedMps(skidTable, v);
      const rest = Math.max(0.0, distM - coreSkidDistanceM(skidTable, v));
      return roll * roll - entrySpeedMps * entrySpeedMps - 2.0 * aEff * rest;
    };
    let lo = 0.0, hi = 20.0;
    for (let i = 0; i < 60; i++) {
      const mid = 0.5 * (lo + hi);
      if (excess(mid) < 0.0) lo = mid; else hi = mid;
    }
    const ratio = centreStrikeSpeedRatio({ headMassKg, ballMassKg, cor });
    return hi / ratio;
  }

  const api = {
    GRAVITY_M_S2, STIMP_RELEASE_SPEED_MPS, STIMPMETER_RELEASE_ANGLE_DEG, HOLE_RADIUS_M,
    GOLF_BALL_RADIUS_M, GOLF_BALL_MASS_KG, PUTTER_FACE_COR, seededRandom, gaussianRand,
    evalUndulatingGreen, coreGreenAt, evalGreen, coreSlopePctAtBall, sampleTurfMicroRoughness, stimpToRollingMu,
    captureSpeedMps, effectiveHeadMass, centreStrikeSpeedRatio, simulateSinglePuttRK4,
    computeClubSpeedForCapture, coreSkidDistanceM, coreRollStartSpeedMps, solveAimCore,
  };

  root.PuttingPhysics = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
