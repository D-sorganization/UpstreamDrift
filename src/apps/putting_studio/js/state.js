/**
 * Stroke kinematics & biomechanical state for the standalone HTML studio
 * (epic #14141, step 8e). Classic script (no ES modules; opens from
 * file://); loaded after vendor/three.min.js and js/putting_physics.js
 * and before the other js/*.js files, which share this file's top-level
 * `const`/`let` bindings (state, PUTTER_SPECS, evalPhysicsAtTime,
 * smoothOUNoise, T_STROKE/T_IMPACT/... timing constants) as globals.
 */
const T_STROKE = 1.05;
const T_TOTAL  = 2.80; // Full cycle: 0..1.05s stroke + 0.817..2.80s ball roll to cup
const T_BACK   = 0.60;
const T_IMPACT = 0.8168; // Exact analytical instant where downswing crosses theta(t_imp) == 0!
const THETA_MAX_DEG = 8.4; // Model backswing for a ~10-ft putt (illustrative, not calibrated)
// Head speed ratio across a centre strike, 1-D momentum with the shared COR:
// (m_h - e m_b) / (m_h + m_b). 0.35 kg is the shared typical head mass.
const POST_IMPACT_DECEL_RATIO = (0.35 - PuttingPhysics.PUTTER_FACE_COR * PuttingPhysics.GOLF_BALL_MASS_KG)
  / (0.35 + PuttingPhysics.GOLF_BALL_MASS_KG);
const d_L = 0.10, d_R = 0.10;
const s_L_pky = 0.07, s_L_idx = 0.17, s_M_L = 0.12;
const s_R_pky = 0.15, s_R_idx = 0.25, s_M_R = 0.20;
const L_CLUB = 0.86;

const state = {
  t: 0.817,
  playing: true,
  speed: 0.45,
  tauPreload: 0.60,
  tauOffPreload: 0.55,
  noiseScale: 0.25,
  pertImpulseNm: 0.25,
  toeOffsetMm: 10.0,
  psiLieDeg: 70.0,
  buttArcRadius: 0.36,
  vecMode: 'total',
  putterType: 'th',
  // Component visibility flags (controlled by checkboxes)
  showClub: true,
  showForceVecs: true,
  showHandTorques: true,
  showNetTorque: true,
  showImpactVecs: true,
  showButtArc: true,
  showStrokePlane: true,
  showBallTraces3D: true,
  showGreenEnv: true,
  showGhost: false,       // Orange unbalanced slack ghost OFF by default!
  showVecLabels: false,   // Floating 3D symbol tags OFF by default (no live torque values on screen)
  showLiveTorqueCards: false,
  showLegend: true,
  showAdv: true,
  showStartHoleLine: true,
  showGreenGrid: true,
  showPhotoFilter: true,
  // Green Roll, Undulation, Roughness, Capture Speed & Monte Carlo parameters
  nTrials: 40,
  puttDistFt: 10.0,
  captureSpeedMs: 0.58,
  surfaceRoughnessMm: 0.35,
  greenReadErrorPct: 0.25,
  stimpmeterMode: false,
  cumulativeMakes: 0,
  cumulativeAttempts: 0,
  crossSlopePct: 1.8,
  uphillSlopePct: 0.3,
  stimpFt: 11.5,
  aimAngleDeg: 4.8,
  rollT: 3.6,
  rollPlaying: true,
  mcBatchSeed: 1
};

// Smooth C^infinity analytical Ornstein-Uhlenbeck multi-sine synthesis (zero step discontinuities)
function smoothOUNoise(t, seedOffset) {
  return 0.45 * Math.sin(2.0 * Math.PI * 1.7 * t + seedOffset)
       + 0.32 * Math.cos(2.0 * Math.PI * 3.1 * t + 1.9 * seedOffset)
       + 0.23 * Math.sin(2.0 * Math.PI * 4.8 * t - 0.7 * seedOffset);
}

// sd4pt / sdSlk: assumed face-angle SD inputs [deg] per grip style (not measured).
const PUTTER_SPECS = {
  th:  { name: 'Toe-Hang Blade (45°)', e_face: 0.015, e_back: 0.012, moi: 3200, mHead: 0.345, buttKg: 0.000, sd4pt: 0.365, sdSlk: 1.303 },
  fb:  { name: 'Face-Balanced Mallet', e_face: 0.000, e_back: 0.026, moi: 5200, mHead: 0.360, buttKg: 0.000, sd4pt: 0.312, sdSlk: 0.957 },
  lab: { name: 'Zero-Torque / LAB',    e_face: 0.000, e_back: 0.000, moi: 4900, mHead: 0.365, buttKg: 0.000, sd4pt: 0.305, sdSlk: 0.804 },
  cb:  { name: 'Heavy Counterbalanced',e_face: 0.002, e_back: 0.018, moi: 5800, mHead: 0.385, buttKg: 0.075, sd4pt: 0.268, sdSlk: 0.894 }
};

function evalPhysicsAtTime(tiRaw, tauPre, psiDeg) {
  const ti = Math.min(tiRaw, T_STROKE); // Hold finish pose after T_STROKE (1.05s) while ball rolls
  const beta = (90.0 - psiDeg) * Math.PI / 180.0;
  const thMax = THETA_MAX_DEG * Math.PI / 180.0;

  let th = 0, om = 0, al = 0;
  if (ti <= T_BACK) {
    const s = ti / T_BACK;
    const p = 10*Math.pow(s,3) - 15*Math.pow(s,4) + 6*Math.pow(s,5);
    const dp = (30*Math.pow(s,2) - 60*Math.pow(s,3) + 30*Math.pow(s,4)) / T_BACK;
    const ddp = (60*s - 180*Math.pow(s,2) + 120*Math.pow(s,3)) / (T_BACK*T_BACK);
    th = -thMax * p;
    om = -thMax * dp;
    al = -thMax * ddp;
  } else {
    const T_rem = T_STROKE - T_BACK;
    const s = Math.min((ti - T_BACK) / T_rem, 1.0);
    const p = 10*Math.pow(s,3) - 15*Math.pow(s,4) + 6*Math.pow(s,5);
    const dp = (30*Math.pow(s,2) - 60*Math.pow(s,3) + 30*Math.pow(s,4)) / T_rem;
    const ddp = (60*s - 180*Math.pow(s,2) + 120*Math.pow(s,3)) / (T_rem*T_rem);
    if (ti <= T_IMPACT) {
      th = -thMax + 2.15 * thMax * p;
      om = 2.15 * thMax * dp;
      al = 2.15 * thMax * ddp;
    } else {
      // Post-impact follow-through: the head slows by POST_IMPACT_DECEL_RATIO at t_imp.
      const pImp = 1.0 / 2.15;
      th = 2.15 * thMax * (p - pImp) * POST_IMPACT_DECEL_RATIO;
      om = (tiRaw > T_STROKE) ? 0.0 : (2.15 * thMax * dp * POST_IMPACT_DECEL_RATIO);
      al = (tiRaw > T_STROKE) ? 0.0 : (2.15 * thMax * ddp * POST_IMPACT_DECEL_RATIO);
    }
  }

  const tauGravIn = -0.52 * 9.81 * 0.58 * Math.cos(beta) * Math.sin(th);
  const tauNet = 0.38 * al + 0.35 * om - tauGravIn;

  const nScale = 0.045 * state.noiseScale;
  const nL_in  = nScale * (0.8 + tauPre) * smoothOUNoise(ti, 1.1);
  const nR_in  = nScale * (0.8 + tauPre) * smoothOUNoise(ti, 2.7);
  const nL_off = nScale * (0.6 + state.tauOffPreload) * smoothOUNoise(ti, 4.3);
  const nR_off = nScale * (0.6 + state.tauOffPreload) * smoothOUNoise(ti, 5.9);

  const softPos = 0.5 * (tauNet + Math.sqrt(tauNet*tauNet + 0.12*0.12));
  const softNeg = 0.5 * (-tauNet + Math.sqrt(tauNet*tauNet + 0.12*0.12));
  const tauL_in = -(tauPre + softNeg) + nL_in;
  const tauR_in = +(tauPre + softPos) + nR_in;

  const tauGravOff = -0.52 * 9.81 * 0.58 * Math.sin(beta) * Math.cos(th);
  const tauCentOff = 0.22 * (om * om) * Math.cos(th) + 0.045 * al;
  const tauOffReq = -(tauGravOff + tauCentOff);
  const softPosOff = 0.5 * (tauOffReq + Math.sqrt(tauOffReq*tauOffReq + 0.08*0.08));
  const softNegOff = 0.5 * (-tauOffReq + Math.sqrt(tauOffReq*tauOffReq + 0.08*0.08));
  const tauL_off = +(state.tauOffPreload + softPosOff) + nL_off;
  const tauR_off = -(state.tauOffPreload + softNegOff) + nR_off;

  const FL_in = -tauL_in / d_L;
  const FR_in = +tauR_in / d_R;
  const FL_off = tauL_off / d_L;
  const FR_off = tauR_off / d_R;

  // Apples-to-Apples Perturbation & Recoil Response (Identical 0.000 deg trajectory when Perturbation=0 & Noise=0!)
  const pertStrength = (state.pertImpulseNm || 0.0) / 0.25;
  const pertWave = Math.exp(-Math.pow((ti - 0.72) / 0.065, 2)) * Math.sin(14.0 * Math.max(0, ti - 0.66));
  const kScaleStab = 1.80 / (1.80 + 14.5 * tauPre);
  const sdnTremor = (0.018 * state.noiseScale * Math.pow(Math.max(0.05, tauPre), 1.42)) * smoothOUNoise(ti, 3.7);
  const baseNoiseSlk = (0.045 * state.noiseScale) * smoothOUNoise(ti, 3.7);
  const phiPreDeg = (1.401 * pertStrength * kScaleStab) * pertWave + sdnTremor;
  const phiSlkDeg = (1.401 * pertStrength) * pertWave + baseNoiseSlk;

  // Instantaneous Impact Delivery Angles (Face Angle & Clubhead Path)
  const pSpec = PUTTER_SPECS[state.putterType] || PUTTER_SPECS.th;
  const preloadStiff = Math.max(0.22, tauPre + 0.5 * state.tauOffPreload);
  const faceErrDeg = (0.14 + 0.35 * state.noiseScale / preloadStiff) * smoothOUNoise(T_IMPACT, 2.1) + (pSpec.e_face * 8.0 / preloadStiff);
  const pathErrDeg = (0.08 + 0.22 * state.noiseScale / preloadStiff) * smoothOUNoise(T_IMPACT, 4.7);

  return {
    thDeg: th * 180 / Math.PI,
    thRad: th,
    phiPreDeg,
    phiSlkDeg,
    tauNet,
    tauOffReq,
    tauL_in, tauR_in,
    tauL_off, tauR_off,
    FL_in, FR_in, FL_off, FR_off,
    faceAngleImpDeg: faceErrDeg,
    clubPathImpDeg: pathErrDeg
  };
}

const N_PTS = 210;
const chartTimes = [];
for (let i = 0; i <= N_PTS; i++) chartTimes.push((i / N_PTS) * T_STROKE);

