/**
 * Monte Carlo green-roll dispersion batch, the 3D ball-trace renderer and
 * the per-frame club/force/torque 3D scene sync for the standalone HTML
 * studio (epic #14141, step 8k). Classic script (no ES modules; opens
 * from file://). The roll physics itself (seededRandom, gaussianRand,
 * evalUndulatingGreen, sampleTurfMicroRoughness, simulateSinglePuttRK4,
 * computeClubSpeedForCapture) now live in js/putting_physics.js
 * (globalThis.PuttingPhysics) -- see that file and ../SUNSET.md for the
 * physics corrections vs. this file's former inline copies (USGA
 * mu = v_release^2/(2 g S) instead of 0.560/stimpFt, plain g instead of
 * (5/7) g, 0.78 restitution instead of 0.80, the Holmes/Penner chord
 * capture bound instead of the 1.38*(1-(b/R)^2)^0.75 heuristic, and the
 * off-centre effective-mass launch ratio instead of a Gaussian "eff"
 * sweet-spot heuristic). Depends on globals from js/state.js and
 * js/scene3d_a.js + js/scene3d_b.js (state, PUTTER_SPECS, L_CLUB,
 * T_IMPACT, clubToThree, the club/scene meshes and groups) and exposes
 * greenBatchCB, greenBatchUN, lastBatchStats, rebuildGreenDispersionBatch,
 * rebuild3DBallTracesOnGreen and update3DScene as globals for
 * js/charts2d.js and the js/controls_*.js files.
 */
let greenBatchCB = [];
let greenBatchUN = [];
// Filled by rebuildGreenDispersionBatch(), which io.js runs at load.
let lastBatchStats = { makeCB: 0, lipOutCB: 0, makeUN: 0, faceSD: 0, pathSD: 0, radSD: 0, skidCm: 0 };

function rebuild3DBallTracesOnGreen() {
  while (ballTraces3DGroup.children.length) ballTraces3DGroup.remove(ballTraces3DGroup.children[0]);
  if (!state.showGreenEnv) return;

  const beta = (90.0 - state.psiLieDeg) * Math.PI / 180.0;
  const yGround = -L_CLUB * Math.cos(beta) - 0.02135;
  const zBall   =  L_CLUB * Math.sin(beta);
  const holeX   = (state.puttDistFt || 10.0) * 0.3048;
  const scale3DX = 3.048 / Math.max(0.6, holeX); // Maps active putt distance onto 3D cup at 3.048m
  const elapsedRoll = Math.max(0.0, state.t - T_IMPACT);

  // 1. Optional 3D Topographic Undulating Slope Grid Overlay on Green
  if (state.showGreenGrid) {
    const gridMat = new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.22 });
    for (let gx = 0.0; gx <= 3.30; gx += 0.38) {
      const pts = [];
      for (let gz = -0.85; gz <= 0.85; gz += 0.10) {
        const xm = (gx / scale3DX);
        const ym = (-gz / scale3DX);
        const elev = PuttingPhysics.evalUndulatingGreen(xm, ym, holeX, state.uphillSlopePct, state.crossSlopePct).z;
        pts.push(new THREE.Vector3(0.02135 + gx, yGround + 0.0025 + elev * 0.25, zBall + gz));
      }
      ballTraces3DGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), gridMat));
    }
    for (let gz = -0.76; gz <= 0.76; gz += 0.38) {
      const pts = [];
      for (let gx = 0.0; gx <= 3.30; gx += 0.12) {
        const xm = (gx / scale3DX);
        const ym = (-gz / scale3DX);
        const elev = PuttingPhysics.evalUndulatingGreen(xm, ym, holeX, state.uphillSlopePct, state.crossSlopePct).z;
        pts.push(new THREE.Vector3(0.02135 + gx, yGround + 0.0025 + elev * 0.25, zBall + gz));
      }
      ballTraces3DGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), gridMat));
    }
  }

  // 2. Optional Straight Line to the Hole & Initial Starting Direction Ray on 3D Green
  if (state.showStartHoleLine) {
    const holeLinePts = [
      new THREE.Vector3(0.02135, yGround + 0.005, zBall),
      new THREE.Vector3(CUP_X_3D, yGround + 0.005, zBall)
    ];
    const holeLineMat = new THREE.LineBasicMaterial({ color: 0xe2e8f0, transparent: true, opacity: 0.58 });
    ballTraces3DGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(holeLinePts), holeLineMat));

    const aimRad = (state.aimAngleDeg || 0.0) * Math.PI / 180.0;
    const startRayPts = [
      new THREE.Vector3(0.02135, yGround + 0.0055, zBall),
      new THREE.Vector3(0.02135 + 2.65 * Math.cos(aimRad), yGround + 0.0055, zBall - 2.65 * Math.sin(aimRad))
    ];
    const startRayMat = new THREE.LineBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.78 });
    ballTraces3DGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(startRayPts), startRayMat));
  }

  if (!state.showBallTraces3D) return;

  // 3. Render Monte Carlo trial traces on the 3D Oblong Green (thinner/faded for batch trials, bold for current putt idx=0)
  const max3DTraces = Math.min(greenBatchCB.length, 140);
  for (let idx = max3DTraces - 1; idx >= 0; idx--) {
    const tr = greenBatchCB[idx];
    const linePts = [];
    for (let k = 0; k < tr.pts.length; k++) {
      const p = tr.pts[k];
      if (p.t <= elapsedRoll || state.t < T_IMPACT * 0.5) {
        const x3 = 0.02135 + p.x * scale3DX;
        const z3 = zBall - p.y * scale3DX;
        linePts.push(new THREE.Vector3(x3, yGround + (idx === 0 ? 0.006 : 0.0038), z3));
      }
    }
    if (linePts.length >= 2) {
      const geom = new THREE.BufferGeometry().setFromPoints(linePts);
      const col = idx === 0 ? 0x38bdf8 : (tr.holed ? 0x22d3ee : (tr.lippedOut ? 0xfacc15 : 0xf87171));
      const op = idx === 0 ? 0.98 : (greenBatchCB.length > 80 ? 0.18 : 0.32);
      const mat = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: op });
      ballTraces3DGroup.add(new THREE.Line(geom, mat));
    }
  }
}

function rebuildGreenDispersionBatch() {
  greenBatchCB = [];
  greenBatchUN = [];
  const nTrials = state.nTrials || 40;
  const pSpec = PUTTER_SPECS[state.putterType] || PUTTER_SPECS.th;
  const distFt = state.puttDistFt || 10.0;
  const entrySpd = state.captureSpeedMs || 0.58;
  const roughnessMm = state.surfaceRoughnessMm !== undefined ? state.surfaceRoughnessMm : 0.35;
  const vNom = state.stimpmeterMode
    ? (1.83 / 1.595)
    : PuttingPhysics.computeClubSpeedForCapture({
        distFt, entrySpeedMps: entrySpd, stimpFt: state.stimpFt, upPct: state.uphillSlopePct,
      });
  const noiseMult = state.stimpmeterMode ? 0.0 : Math.max(0.12, 0.55 + 0.65 * state.noiseScale);
  // U-shaped counter-torque stabilization factor
  const tp = state.tauPreload;
  const preloadFactor = 0.44 + 0.56 * Math.exp(-tp / 0.32) + 0.14 * Math.pow(Math.max(0.0, tp - 0.80), 2);
  const readSdPct = state.stimpmeterMode ? 0.0 : (state.greenReadErrorPct !== undefined ? state.greenReadErrorPct : 0.25);

  let cbHoled = 0, cbLip = 0, unHoled = 0, skidSum = 0;
  const cbY = [], unY = [], faceArr = [], pathArr = [], radArr = [], cbLeaveFt = [], unLeaveFt = [];

  for (let i = 0; i < nTrials; i++) {
    const sBase = state.mcBatchSeed * 1000 + i * 13;
    const readErrDeg = (i === 0 ? 0 : PuttingPhysics.gaussianRand(sBase + 9)) * readSdPct * 2.45;
    const dvCB = (i === 0 ? 0 : PuttingPhysics.gaussianRand(sBase + 1)) * 0.022 * vNom * noiseMult * preloadFactor;
    const dpCB = (i === 0 ? 0 : PuttingPhysics.gaussianRand(sBase + 2)) * 0.32 * noiseMult * preloadFactor;
    const dfCB = (i === 0 ? 0 : PuttingPhysics.gaussianRand(sBase + 3)) * pSpec.sd4pt * noiseMult * preloadFactor + readErrDeg;
    const dxCB = (i === 0 ? 0 : PuttingPhysics.gaussianRand(sBase + 4)) * 2.1 * noiseMult * preloadFactor;
    const rCB = PuttingPhysics.simulateSinglePuttRK4({
      vClubMps: state.stimpmeterMode ? 1.83 : (vNom + dvCB),
      aimDeg: state.stimpmeterMode ? 0.0 : state.aimAngleDeg,
      pathErrDeg: dpCB,
      faceErrDeg: dfCB,
      strikeMm: dxCB,
      moiKgM2: pSpec.moi / 1.0e7,
      stimpFt: state.stimpFt,
      upPct: state.uphillSlopePct,
      crossPct: state.crossSlopePct,
      trialSeed: i,
      distFt: distFt,
      roughnessMm: roughnessMm,
      isStimpRamp: state.stimpmeterMode,
    });
    greenBatchCB.push(rCB);
    if (rCB.holed) cbHoled++;
    if (rCB.lippedOut) cbLip++;
    skidSum += rCB.skidDist;
    cbY.push(rCB.finalY * 100);
    faceArr.push(rCB.faceErrDeg);
    pathArr.push(rCB.pathErrDeg);
    radArr.push(rCB.missDistCm);
    cbLeaveFt.push(rCB.holed ? 0.0 : (rCB.missDistCm / 30.48));

    const readErrUN = PuttingPhysics.gaussianRand(sBase + 10) * (readSdPct * 1.35) * 2.45;
    const dvUN = PuttingPhysics.gaussianRand(sBase + 5) * 0.055 * vNom * noiseMult;
    const dpUN = PuttingPhysics.gaussianRand(sBase + 6) * 0.82 * noiseMult;
    const dfUN = PuttingPhysics.gaussianRand(sBase + 7) * pSpec.sdSlk * noiseMult + readErrUN;
    const dxUN = PuttingPhysics.gaussianRand(sBase + 8) * 6.8 * noiseMult;
    const rUN = PuttingPhysics.simulateSinglePuttRK4({
      vClubMps: state.stimpmeterMode ? 1.83 : (vNom + dvUN),
      aimDeg: state.stimpmeterMode ? 0.0 : state.aimAngleDeg,
      pathErrDeg: dpUN,
      faceErrDeg: dfUN,
      strikeMm: dxUN,
      moiKgM2: 3800 / 1.0e7,
      stimpFt: state.stimpFt,
      upPct: state.uphillSlopePct,
      crossPct: state.crossSlopePct,
      trialSeed: i + 5000,
      distFt: distFt,
      roughnessMm: roughnessMm,
      isStimpRamp: state.stimpmeterMode,
    });
    greenBatchUN.push(rUN);
    if (rUN.holed) unHoled++;
    unY.push(rUN.finalY * 100);
    unLeaveFt.push(rUN.holed ? 0.0 : (rUN.missDistCm / 30.48));
  }

  if (!state.stimpmeterMode) {
    state.cumulativeMakes = (state.cumulativeMakes || 0) + cbHoled;
    state.cumulativeAttempts = (state.cumulativeAttempts || 0) + nTrials;
  }

  const stdArr = (arr) => {
    if (!arr.length) return 0;
    const m = arr.reduce((a, b) => a + b, 0) / arr.length;
    return Math.sqrt(arr.reduce((a, b) => a + (b - m) * (b - m), 0) / arr.length);
  };

  const makePctCB = (cbHoled / nTrials) * 100.0;
  const lipPctCB  = (cbLip / nTrials) * 100.0;
  const makePctUN = (unHoled / nTrials) * 100.0;
  const fSD = stdArr(faceArr);
  const pSD = stdArr(pathArr);
  const rSD = stdArr(cbY);

  // Player-Specific Bellman Expected Strokes E_player(d) = 1 + (1 - P_make) * E_player(d_leave)
  const meanMissLeaveCB = cbLeaveFt.filter(v => v > 0).reduce((a, b) => a + b, 0) / Math.max(1, nTrials - cbHoled);
  const meanMissLeaveUN = unLeaveFt.filter(v => v > 0).reduce((a, b) => a + b, 0) / Math.max(1, nTrials - unHoled);
  const eComebackCB = 1.0 + Math.min(0.85, 0.065 * Math.pow(meanMissLeaveCB, 1.18));
  const eComebackUN = 1.0 + Math.min(0.95, 0.115 * Math.pow(meanMissLeaveUN, 1.25));
  const ePlayerCB = 1.0 + (1.0 - makePctCB / 100.0) * eComebackCB;
  const ePlayerUN = 1.0 + (1.0 - makePctUN / 100.0) * eComebackUN;
  const sgPlayerGain = ePlayerUN - ePlayerCB;

  lastBatchStats = {
    makeCB: makePctCB,
    lipOutCB: lipPctCB,
    makeUN: makePctUN,
    faceSD: fSD,
    pathSD: pSD,
    radSD: rSD,
    skidCm: (skidSum / nTrials) * 100.0,
    ePlayerCB,
    ePlayerUN,
    sgPlayerGain,
    meanMissLeaveCB,
    meanMissLeaveUN
  };

  const elCB = document.getElementById('greenMakeCB');
  const elUN = document.getElementById('greenMakeUN');
  const elSkid = document.getElementById('greenSkidDist');
  const elLat = document.getElementById('greenLatSD');
  if (elCB) elCB.textContent = `${makePctCB.toFixed(1)}%`;
  if (elUN) elUN.textContent = `${makePctUN.toFixed(1)}%`;
  if (elSkid) elSkid.textContent = `${lastBatchStats.skidCm.toFixed(1)} cm (SLIDING)`;
  if (elLat) elLat.textContent = `${rSD.toFixed(1)} cm vs ${stdArr(unY).toFixed(1)} cm`;

  const badgeMake = document.getElementById('makeRate3DBadge');
  if (badgeMake) badgeMake.textContent = `Make: ${makePctCB.toFixed(1)}% (${cbHoled}/${nTrials})`;
  const liveMake = document.getElementById('valLiveMakePct');
  if (liveMake) liveMake.textContent = `${makePctCB.toFixed(1)}% Holed (${cbLip} Lip-Out)`;
  const lblN = document.getElementById('lblNTrials');
  if (lblN) lblN.textContent = `${nTrials}`;

  const cumMakePct = state.cumulativeAttempts > 0 ? (100.0 * state.cumulativeMakes / state.cumulativeAttempts) : makePctCB;
  const cumBadge = document.getElementById('cumulativeMakeBadge');
  if (cumBadge) {
    cumBadge.textContent = `Cumulative Make: ${cumMakePct.toFixed(1)}% (${state.cumulativeMakes}/${state.cumulativeAttempts})`;
  }
  const elEPlayer = document.getElementById('valEPlayer');
  if (elEPlayer) elEPlayer.textContent = `${ePlayerCB.toFixed(2)} strokes (vs ${ePlayerUN.toFixed(2)} Slack)`;
  const elSGGain = document.getElementById('valSGPlayerGain');
  if (elSGGain) elSGGain.textContent = `${sgPlayerGain >= 0 ? '+' : ''}${sgPlayerGain.toFixed(2)} strokes / putt`;

  // Update 6-Channel 18-Hole Strokes Gained Variance Attribution
  const distScale = Math.sqrt(distFt / 12.0);
  const chCT = +(2.15 * (1.0 - preloadFactor) * distScale + 0.65).toFixed(2);
  const chFace = -(1.42 * preloadFactor * (0.5 + state.noiseScale) * distScale).toFixed(2);
  const chRead = -(1.10 * ((state.greenReadErrorPct || 0.25) / 0.25) * (Math.abs(state.crossSlopePct) / 2.0 + 0.3) * distScale).toFixed(2);
  const chSpot = -(0.78 * (1.0 + (state.toeOffsetMm || 0) / 15.0) * preloadFactor * distScale).toFixed(2);
  const chRough = -(0.52 * ((state.surfaceRoughnessMm || 0.35) / 0.35) * distScale).toFixed(2);
  const chPath = -(0.29 * preloadFactor * (0.5 + state.noiseScale) * distScale).toFixed(2);
  const setTxt = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
  setTxt('chGainCT', `+${Math.abs(chCT).toFixed(2)} SG`);
  setTxt('chLossFace', `${chFace.toFixed(2)} SG`);
  setTxt('chLossRead', `${chRead.toFixed(2)} SG`);
  setTxt('chLossSpot', `${chSpot.toFixed(2)} SG`);
  setTxt('chLossRough', `${chRough.toFixed(2)} SG`);
  setTxt('chLossPath', `${chPath.toFixed(2)} SG`);

  if (typeof drawSGCaptureChart === 'function') drawSGCaptureChart();
}
function update3DScene(phys) {
  const cButt = clubToThree(0.0, phys.thRad, phys.phiPreDeg, state.psiLieDeg);
  const cTop = clubToThree(0.03, phys.thRad, phys.phiPreDeg, state.psiLieDeg);
  const cGripEnd = clubToThree(0.29, phys.thRad, phys.phiPreDeg, state.psiLieDeg);
  const cHead = clubToThree(L_CLUB, phys.thRad, phys.phiPreDeg, state.psiLieDeg);

  courseGroup.visible = state.showGreenEnv;
  planeGroup.visible = state.showStrokePlane;
  buttArcGroup.visible = state.showButtArc && (state.buttArcRadius > 0.01);

  pivotSphere.visible = state.showClub;
  gripCylinder.visible = state.showClub;
  shaftCylinder.visible = state.showClub;
  putterHead.visible = state.showClub;

  pivotSphere.position.copy(cButt.pos);
  const shPos = getShoulderHubThree(state.psiLieDeg);
  shoulderSphere.position.copy(shPos);
  if (buttArcGroup.visible) {
    const cML = clubToThree(s_M_L, phys.thRad, phys.phiPreDeg, state.psiLieDeg);
    const cMR = clubToThree(s_M_R, phys.thRad, phys.phiPreDeg, state.psiLieDeg);
    alignCylinder(leftArmCylinder, shPos, cML.pos, true);
    alignCylinder(rightArmCylinder, shPos, cMR.pos, true);
  }

  alignCylinder(gripCylinder, cTop.pos, cGripEnd.pos);
  alignCylinder(shaftCylinder, cButt.pos, cHead.pos);
  putterHead.position.copy(cHead.pos);
  putterHead.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), cHead.e_th.clone().normalize());

  // Exact Ball Launch Synchronization at t_imp = 0.8168s (where theta == 0, ZERO delay!)
  const beta = (90.0 - state.psiLieDeg) * Math.PI / 180.0;
  const yGround = -L_CLUB * Math.cos(beta) - 0.02135;
  const zBall = L_CLUB * Math.sin(beta);
  // Exact 1:1 Metric Scale (scale3DX = 1.0): Ball launches at v_ball(0) = 1.79 m/s (1.60x pre-impact club speed,
  // 2.00x post-impact follow-through club speed!) and rolls 3.048 m (10.0 ft) to the USGA cup!
  const scale3DX = 1.0;

  if (state.t <= T_IMPACT || !greenBatchCB.length) {
    golfBallMesh.position.set(0.02135, yGround + 0.02135, zBall);
    golfBallMesh.rotation.set(0, 0, 0);
    impactFlashRing.visible = false;
  } else {
    const dtRoll = state.t - T_IMPACT;
    const heroPts = greenBatchCB[0].pts;
    let pt = heroPts[heroPts.length - 1];
    for (let k = 0; k < heroPts.length; k++) {
      if (heroPts[k].t >= dtRoll) { pt = heroPts[k]; break; }
    }
    golfBallMesh.position.set(0.02135 + pt.x * scale3DX, yGround + 0.02135, zBall - pt.y * scale3DX);
    // Forward rolling rotation of dimpled golf ball (theta_ball = x_travel / R_ball)
    golfBallMesh.rotation.z = -(pt.x / 0.02135);
    // Elastoplastic impact rebound shockwave ring during first 90 ms after strike
    if (dtRoll >= 0.0 && dtRoll <= 0.095) {
      impactFlashRing.visible = true;
      const prog = dtRoll / 0.095;
      impactFlashRing.position.set(0.02135, yGround + 0.02135, zBall);
      const sc = 1.0 + 3.2 * prog;
      impactFlashRing.scale.set(sc, sc, sc);
      impactFlashRing.material.opacity = 0.85 * (1.0 - prog);
    } else {
      impactFlashRing.visible = false;
    }
  }

  rebuild3DBallTracesOnGreen();

  // 3D Impact Delivery Vectors (Face Angle Normal, Clubhead Path, & Ball Launch Direction)
  impactVecsGroup.visible = state.showImpactVecs;
  if (state.showImpactVecs) {
    const impOrigin = new THREE.Vector3(0.02135, yGround + 0.022, zBall);
    const pathRad = (state.aimAngleDeg + phys.clubPathImpDeg) * Math.PI / 180.0;
    const faceRad = (state.aimAngleDeg + phys.faceAngleImpDeg) * Math.PI / 180.0;
    const launchRad = 0.84 * faceRad + 0.16 * pathRad;

    const vPath = new THREE.Vector3(Math.cos(pathRad), 0, -Math.sin(pathRad));
    const vFace = new THREE.Vector3(Math.cos(faceRad), 0, -Math.sin(faceRad));
    const vLnch = new THREE.Vector3(Math.cos(launchRad), 0, -Math.sin(launchRad));

    setThickForceArrow(impactPathArrow, impOrigin, vPath, 0.22);
    setThickForceArrow(impactFaceArrow, impOrigin, vFace, 0.25);
    setThickForceArrow(impactLaunchArrow, impOrigin, vLnch, 0.28);
  }

  // Orange Unbalanced Slack Ghost Club (OFF by default; controlled by chkGhostClub)
  ghostShaft.visible = state.showGhost;
  ghostHead.visible = state.showGhost;
  if (state.showGhost) {
    const gButt = clubToThree(0.0, phys.thRad, phys.phiSlkDeg * 12.0, state.psiLieDeg);
    const gHead = clubToThree(L_CLUB, phys.thRad, phys.phiSlkDeg * 12.0, state.psiLieDeg);
    alignCylinder(ghostShaft, gButt.pos, gHead.pos);
    ghostHead.position.copy(gHead.pos);
    ghostHead.quaternion.copy(putterHead.quaternion);
  }

  const inMult = (state.vecMode === 'offplane') ? 0.0 : 1.0;
  const offMult = (state.vecMode === 'inplane') ? 0.0 : 1.0;

  const sPts = [s_L_pky, s_L_idx, s_R_pky, s_R_idx];
  const fLocals = [
    [0, +phys.FL_in * inMult, -phys.FL_off * offMult],
    [0, -phys.FL_in * inMult, +phys.FL_off * offMult],
    [0, -phys.FR_in * inMult, -phys.FR_off * offMult],
    [0, +phys.FR_in * inMult, +phys.FR_off * offMult]
  ];
  const lblIds = ['lblFLpky', 'lblFLidx', 'lblFRpky', 'lblFRidx'];
  const lblSymbols = [
    '<span class="math-vec">F&#8407;</span><sub>L,pky</sub>',
    '<span class="math-vec">F&#8407;</span><sub>L,idx</sub>',
    '<span class="math-vec">F&#8407;</span><sub>R,pky</sub>',
    '<span class="math-vec">F&#8407;</span><sub>R,idx</sub>'
  ];

  for (let i = 0; i < 4; i++) {
    const res = clubToThree(sPts[i], phys.thRad, phys.phiPreDeg, state.psiLieDeg, fLocals[i]);
    contactSpheres[i].visible = state.showForceVecs;
    contactSpheres[i].position.copy(res.pos);
    const mag = res.v3.length();
    if (state.showForceVecs && mag > 0.01) {
      forceArrows[i].visible = true;
      const totalLen = Math.min(0.15, 0.035 + mag * 0.0044);
      const tipPos = setThickForceArrow(forceArrows[i], res.pos, res.v3, totalLen);
      project3DLabel(lblIds[i], tipPos, lblSymbols[i], true);
    } else {
      forceArrows[i].visible = false;
      project3DLabel(lblIds[i], res.pos, '', false);
    }
  }

  // 2 Midpoint Curved Torque Arrows at M_L (s=0.12m) and M_R (s=0.20m)
  const sMids = [s_M_L, s_M_R];
  const tauLocals = [
    [0, -phys.tauL_off * offMult, phys.tauL_in * inMult],
    [0, -phys.tauR_off * offMult, phys.tauR_in * inMult]
  ];
  const tauLblIds = ['lblTauL', 'lblTauR'];
  const tauSymbols = ['&#x21BA; <span class="math-vec">&tau;</span><sub>L</sub>', '&#x21BB; <span class="math-vec">&tau;</span><sub>R</sub>'];

  for (let m = 0; m < 2; m++) {
    const resM = clubToThree(sMids[m], phys.thRad, phys.phiPreDeg, state.psiLieDeg, tauLocals[m]);
    midDiamonds[m].visible = state.showHandTorques;
    midDiamonds[m].position.copy(resM.pos);
    const tMag = resM.v3.length();
    if (state.showHandTorques && tMag > 0.01) {
      torqueArrows[m].visible = true;
      torqueArrows[m].position.copy(resM.pos);
      torqueArrows[m].quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), resM.v3.clone().normalize());
      const sc = Math.min(1.55, 0.72 + tMag * 0.34);
      torqueArrows[m].scale.set(sc, sc, sc);
      const lblPos = resM.pos.clone().addScaledVector(resM.e_perp, (m === 0 ? -0.055 : 0.055));
      project3DLabel(tauLblIds[m], lblPos, tauSymbols[m], true);
    } else {
      torqueArrows[m].visible = false;
      project3DLabel(tauLblIds[m], resM.pos, '', false);
    }
  }

  // Resultant Net Torque Curved Arrow (3D Rotational Couple) at Grip Center (s=0.16m)
  const netTauLocal = [
    0,
    -(phys.tauL_off + phys.tauR_off) * offMult,
    (phys.tauL_in + phys.tauR_in) * inMult
  ];
  const resNet = clubToThree(0.16, phys.thRad, phys.phiPreDeg, state.psiLieDeg, netTauLocal);
  const netMag = resNet.v3.length();
  if (state.showNetTorque && netMag > 0.01) {
    netTorqueCurvedArrow.visible = true;
    netTorqueCurvedArrow.position.copy(resNet.pos);
    netTorqueCurvedArrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), resNet.v3.clone().normalize());
    const nSc = Math.min(1.65, 0.75 + netMag * 0.38);
    netTorqueCurvedArrow.scale.set(nSc, nSc, nSc);
    const netLblPos = resNet.pos.clone().add(new THREE.Vector3(0, 0.058, 0));
    project3DLabel('lblTauNet', netLblPos, '&#x21BB; <span class="math-vec">&tau;</span><sub>net</sub>', true);
  } else {
    netTorqueCurvedArrow.visible = false;
    project3DLabel('lblTauNet', resNet.pos, '', false);
  }

  renderer.render(scene, camera);
}
