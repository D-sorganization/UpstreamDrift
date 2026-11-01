/**
 * Saved-scenario comparison table, the main per-frame UI readout sync
 * (syncUI), the grip-pressure U-curve sweep chart and the clubface
 * strike mini-map for the standalone HTML studio (epic #14141, step
 * 8f). Classic script (no ES modules; opens from file://). Depends on
 * globals from js/state.js (state, PUTTER_SPECS, smoothOUNoise,
 * T_IMPACT, T_STROKE, evalPhysicsAtTime, L_CLUB), js/scene3d_b.js
 * (putterHead) and js/green3d.js (lastBatchStats, rebuildGreenDispersionBatch,
 * update3DScene) and exposes savedScenarios, renderScenarioTable,
 * saveCurrentScenarioSnapshot, fmtSign and syncUI as globals for
 * js/controls_b.js, js/controls_c.js and js/io.js.
 */
// Starts empty: rows are snapshots of this session's model runs, never
// baked-in results.
let savedScenarios = [];

function renderScenarioTable() {
  const tbody = document.getElementById('scenarioTableBody');
  if (!tbody) return;
  tbody.innerHTML = savedScenarios.map((sc, idx) => `
    <tr class="hover:bg-slate-900/70 transition">
      <td class="py-2 px-2 font-bold text-white">${sc.name}</td>
      <td class="py-2 px-2 text-sky-300">${sc.putter} (${sc.preload})</td>
      <td class="py-2 px-2 text-amber-200">${sc.noiseArc}</td>
      <td class="py-2 px-2 text-emerald-300">${sc.green}</td>
      <td class="py-2 px-2 text-amber-300">${sc.faceSD}</td>
      <td class="py-2 px-2 text-purple-300">${sc.pathSD}</td>
      <td class="py-2 px-2 text-cyan-300">${sc.radSD}</td>
      <td class="py-2 px-2 text-orange-300">${sc.lipOut}</td>
      <td class="py-2 px-2 font-black text-emerald-400 text-xs">${sc.makePct}</td>
    </tr>
  `).join('');
}

function saveCurrentScenarioSnapshot(customName) {
  rebuildGreenDispersionBatch();
  const pSpec = PUTTER_SPECS[state.putterType] || PUTTER_SPECS.th;
  const nameInput = document.getElementById('scenarioNameInput');
  const scName = customName || (nameInput && nameInput.value.trim()) || `Scenario #${savedScenarios.length + 1}`;
  savedScenarios.push({
    name: scName,
    putter: pSpec.name,
    preload: `${state.tauPreload.toFixed(2)} / ${state.tauOffPreload.toFixed(2)} N·m`,
    noiseArc: `${state.noiseScale.toFixed(2)}x • ${state.buttArcRadius.toFixed(2)}m Arc`,
    green: `${state.crossSlopePct.toFixed(1)}% Break • Stimp ${state.stimpFt.toFixed(1)}`,
    faceSD: `${lastBatchStats.faceSD.toFixed(2)}°`,
    pathSD: `${lastBatchStats.pathSD.toFixed(2)}°`,
    radSD: `${lastBatchStats.radSD.toFixed(1)} cm`,
    lipOut: `${lastBatchStats.lipOutCB.toFixed(1)}%`,
    makePct: `${lastBatchStats.makeCB.toFixed(1)}%`
  });
  renderScenarioTable();
}

// =========================================================================
// 6. UI UPDATES & EVENT LISTENERS
// =========================================================================
function fmtSign(v, dec=2) { return (v >= 0 ? '+' : '') + v.toFixed(dec); }

function syncUI() {
  const phys = evalPhysicsAtTime(state.t, state.tauPreload, state.psiLieDeg);
  document.getElementById('timeScrubber').value = state.t.toFixed(3);
  document.getElementById('timeDisplay').textContent = state.t.toFixed(3) + ' s';

  const phaseName = state.t < 0.56 ? 'BACKSWING' : (state.t <= 0.65 ? 'TOP TRANSITION' : (state.t < T_IMPACT - 0.015 ? 'DOWNSWING' : (state.t <= T_IMPACT + 0.045 ? 'BALL IMPACT (θ=0°)' : (state.t <= T_STROKE ? 'FOLLOW-THROUGH' : 'BALL ROLL TO CUP'))));
  document.getElementById('phaseBadge').textContent = `${phaseName} (t = ${state.t.toFixed(3)} s)`;
  document.getElementById('lieStatusBadge').innerHTML = `<span class="math-sym">&phi;</span><sub>lie</sub> = ${fmtSign(phys.phiPreDeg, 3)}&deg;`;

  const launchAngle = 0.84 * phys.faceAngleImpDeg + 0.16 * phys.clubPathImpDeg;
  const dPsi = phys.faceAngleImpDeg - phys.clubPathImpDeg;
  const impBadge = document.getElementById('impactParamsBadge');
  if (impBadge) {
    impBadge.innerHTML = `Face <span class="math-sym">&phi;</span><sub>f</sub>=${fmtSign(phys.faceAngleImpDeg, 2)}&deg; &bull; Path <span class="math-sym">&psi;</span><sub>p</sub>=${fmtSign(phys.clubPathImpDeg, 2)}&deg;`;
  }

  const elImpFace = document.getElementById('valImpactFace');
  const elImpPath = document.getElementById('valImpactPath');
  const elF2P = document.getElementById('valFaceToPath');
  if (elImpFace) elImpFace.textContent = `${fmtSign(phys.faceAngleImpDeg, 2)}° (SD ${lastBatchStats.faceSD.toFixed(2)}°)`;
  if (elImpPath) elImpPath.textContent = `${fmtSign(phys.clubPathImpDeg, 2)}° (SD ${lastBatchStats.pathSD.toFixed(2)}°)`;
  if (elF2P) elF2P.innerHTML = `${fmtSign(dPsi, 2)}&deg; &rarr; Launch ${fmtSign(launchAngle, 2)}&deg;`;

  document.getElementById('valFLpkyIn').textContent  = `${fmtSign(+phys.FL_in, 2)} N (Push Fwd)`;
  document.getElementById('valFLpkyOff').textContent = `${fmtSign(-phys.FL_off, 2)} N (Down-Plane)`;
  document.getElementById('valFLidxIn').textContent  = `${fmtSign(-phys.FL_in, 2)} N (Push Back)`;
  document.getElementById('valFLidxOff').textContent = `${fmtSign(+phys.FL_off, 2)} N (Up-Plane)`;
  document.getElementById('valTauML').textContent    = `${fmtSign(phys.tauL_in, 2)} N·m (in) | ${fmtSign(phys.tauL_off, 2)} N·m (off)`;

  document.getElementById('valFRpkyIn').textContent  = `${fmtSign(-phys.FR_in, 2)} N (Pull Back)`;
  document.getElementById('valFRpkyOff').textContent = `${fmtSign(-phys.FR_off, 2)} N (Down-Plane)`;
  document.getElementById('valFRidxIn').textContent  = `${fmtSign(+phys.FR_in, 2)} N (Push Fwd)`;
  document.getElementById('valFRidxOff').textContent = `${fmtSign(+phys.FR_off, 2)} N (Up-Plane)`;
  document.getElementById('valTauMR').textContent    = `${fmtSign(phys.tauR_in, 2)} N·m (in) | ${fmtSign(phys.tauR_off, 2)} N·m (off)`;

  document.getElementById('netTorqueReadout').textContent = state.showLiveTorqueCards ? `τ_net = ${fmtSign(phys.tauNet, 2)} N·m` : `Smooth C² Coupled`;
  const pSpec = PUTTER_SPECS[state.putterType] || PUTTER_SPECS.th;
  const betaRad = (90.0 - state.psiLieDeg) * Math.PI / 180.0;
  const alApprox = (phys.tauNet / 0.38);
  const tauS_mNm = 1000.0 * (-pSpec.mHead * L_CLUB * alApprox * pSpec.e_face + pSpec.mHead * 9.81 * Math.sin(betaRad) * (pSpec.e_back * Math.cos(phys.thRad) - pSpec.e_face * Math.sin(phys.thRad)));
  document.getElementById('valShaftTorque').textContent = `${fmtSign(tauS_mNm, 1)} mN·m`;
  const preloadScale = Math.sqrt(0.85 / Math.max(state.tauPreload, 0.25));
  const adjSd4pt = (pSpec.sd4pt * (0.7 + 0.3 * preloadScale)).toFixed(3);
  document.getElementById('valFaceSD').textContent = `${adjSd4pt}° vs ${pSpec.sdSlk.toFixed(2)}°`;
  document.getElementById('valMake15').textContent = `${lastBatchStats.makeCB.toFixed(1)}% vs ${lastBatchStats.makeUN.toFixed(1)}%`;
  putterHead.scale.set(pSpec.e_back > 0.015 ? 1.65 : (pSpec.e_face === 0 ? 1.45 : 1.0), 1.0, 1.0);
  drawStrikeFaceMiniMap(phys, pSpec);
  drawGripPressureSweepChart();
  update3DScene(phys);
  drawTelemetryCharts();
}

function evalGripSweepPoint(tp) {
  const K_face = 1.80 + 14.5 * tp;
  const I_eff = 4800.0 + 28500.0 * (tp / (0.45 + tp)) + 420.0 * K_face;
  const sigmaExt = 0.850 / Math.pow(1.0 + 3.45 * tp, 1.28);
  const sigmaSdn = Math.sqrt(0.280 * 0.280 + Math.pow(0.341 * Math.pow(tp, 1.42), 2));
  const sigmaTot = Math.sqrt(sigmaExt * sigmaExt + sigmaSdn * sigmaSdn);
  const sigmaDist = 2.38 + 1.67 * Math.exp(-3.5 * tp) + 1.68 * Math.pow(tp, 1.78);
  const sigmaEffLine = Math.sqrt(Math.pow(0.83 * sigmaTot, 2) + Math.pow(0.052 * sigmaDist, 2));
  const erfApprox = (x) => {
    const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    const sign = x < 0 ? -1 : 1;
    const ax = Math.abs(x);
    const t = 1.0 / (1.0 + p * ax);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
    return sign * y;
  };
  const pLine = erfApprox(0.6285 / (Math.SQRT2 * sigmaEffLine));
  const pDist = erfApprox(7.50 / (Math.SQRT2 * sigmaDist));
  const makePct = 100.0 * pLine * pDist;
  return { tp, K_face, I_eff, sigmaExt, sigmaSdn, sigmaTot, makePct };
}

function drawGripPressureSweepChart() {
  const cur = evalGripSweepPoint(state.tauPreload);
  const mof = evalGripSweepPoint(0.0);
  const pertRatio = (state.pertImpulseNm || 0.0) / 0.25;
  const toeRatio = (state.toeOffsetMm || 0.0) / 10.0;
  const pertFaceStab = 1.401 * pertRatio * (1.80 / cur.K_face) + 0.253 * toeRatio * (11521.0 / cur.I_eff);
  const pertFaceMof  = 1.401 * pertRatio + 0.253 * toeRatio;

  const elPert = document.getElementById('valPertFaceComp');
  const elMOI  = document.getElementById('valEffMOIComp');
  const elTot  = document.getElementById('valSweepTotErr');
  const elMake = document.getElementById('valSweepMakePct');
  const badge  = document.getElementById('gripRegimeBadge');

  if (elPert) {
    elPert.textContent = (pertRatio === 0 && toeRatio === 0)
      ? `0.000° vs 0.000° (Exact Match!)`
      : `+${pertFaceStab.toFixed(2)}° vs +${pertFaceMof.toFixed(2)}° (0 N·m)`;
  }
  if (elMOI) elMOI.textContent = `${Math.round(cur.I_eff).toLocaleString()} vs 11,521 g·cm²`;
  if (elTot) elTot.textContent = `${cur.sigmaTot.toFixed(3)}° (Ext ${cur.sigmaExt.toFixed(2)}° / SDN ${cur.sigmaSdn.toFixed(2)}°)`;
  if (elMake) elMake.textContent = `${cur.makePct.toFixed(1)}% vs ${mof.makePct.toFixed(1)}% (0 N·m)`;

  if (badge) {
    if (state.tauPreload < 0.25) {
      badge.textContent = 'Under-Stabilized (Perturbation Prone)';
      badge.className = 'text-[10px] mono font-bold px-2 py-0.5 rounded bg-orange-500/20 text-orange-300 border border-orange-500/40';
    } else if (state.tauPreload <= 0.90) {
      badge.textContent = `Model Mid-Range Preload (${cur.makePct.toFixed(1)}% Model Make)`;
      badge.className = 'text-[10px] mono font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';
    } else {
      badge.textContent = 'Over-Torqued (SDN Tremor Zone)';
      badge.className = 'text-[10px] mono font-bold px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/40';
    }
  }

  const c = document.getElementById('gripSweepCanvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 28, padR = 28, padT = 10, padB = 18;
  const pw = w - padL - padR, ph = h - padT - padB;

  // Highlight Optimal Counter-Torque Sweet Spot Band [0.40, 0.85] N*m
  const xOpt1 = padL + (0.40 / 2.0) * pw;
  const xOpt2 = padL + (0.85 / 2.0) * pw;
  ctx.fillStyle = 'rgba(16, 185, 129, 0.14)';
  ctx.fillRect(xOpt1, padT, xOpt2 - xOpt1, ph);

  const nSteps = 40;
  const pts = [];
  for (let i = 0; i <= nSteps; i++) pts.push(evalGripSweepPoint((i / nSteps) * 2.0));

  const drawSweepLine = (getter, yMin, yMax, color, dashed) => {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.0;
    if (dashed) ctx.setLineDash([4, 3]);
    ctx.beginPath();
    pts.forEach((p, idx) => {
      const x = padL + (p.tp / 2.0) * pw;
      const y = padT + ph * (1.0 - (getter(p) - yMin) / (yMax - yMin));
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  };

  drawSweepLine(p => p.sigmaExt, 0.0, 1.05, '#f59e0b', true);
  drawSweepLine(p => p.sigmaSdn, 0.0, 1.05, '#c084fc', true);
  drawSweepLine(p => p.sigmaTot, 0.0, 1.05, '#22d3ee', false);
  drawSweepLine(p => p.makePct, 20.0, 100.0, '#10b981', false);

  // Current preload vertical cursor
  const xCur = padL + (Math.min(2.0, state.tauPreload) / 2.0) * pw;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(xCur, padT); ctx.lineTo(xCur, padT + ph); ctx.stroke();

  ctx.fillStyle = '#94a3b8';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.fillText('0.0', padL - 6, h - 4);
  ctx.fillText('0.6* Opt', xOpt1 + 2, h - 4);
  ctx.fillText('2.0 N·m', w - padR - 24, h - 4);
}

function drawStrikeFaceMiniMap(phys, pSpec) {
  const sc = document.getElementById('strikeFaceCanvas');
  if (!sc) return;
  const ctx = sc.getContext('2d');
  const w = sc.width, h = sc.height;
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1;
  for (let gx = 40; gx <= w - 40; gx += 20) {
    ctx.beginPath(); ctx.moveTo(gx, 8); ctx.lineTo(gx, h - 8); ctx.stroke();
  }
  ctx.strokeStyle = '#475569';
  ctx.beginPath(); ctx.moveTo(20, h/2); ctx.lineTo(w - 20, h/2); ctx.stroke();
  ctx.strokeStyle = '#10b981';
  ctx.beginPath(); ctx.arc(w/2, h/2, 8, 0, Math.PI * 2); ctx.stroke();

  const tensionStiff = Math.max(0.18, state.tauPreload + 0.5 * state.tauOffPreload);
  const xStrikeMm = (state.noiseScale * 3.2 / tensionStiff) * smoothOUNoise(state.t + 0.15, 3.3);
  const yStrikeMm = (state.noiseScale * 1.4 / Math.max(0.2, state.tauOffPreload + 0.25)) * smoothOUNoise(state.t + 0.35, 7.1);
  const sweetSigmaMm = Math.sqrt(pSpec.moi / 4000.0) * 22.0;
  const corEff = 100.0 * Math.exp(-0.5 * Math.pow(xStrikeMm / sweetSigmaMm, 2));

  const xUnPx = w/2 + (xStrikeMm * 3.2) * 3.8;
  const yUnPx = h/2 - (yStrikeMm * 2.8) * 2.2;
  ctx.fillStyle = 'rgba(249, 115, 22, 0.55)';
  ctx.beginPath(); ctx.arc(Math.max(16, Math.min(w-16, xUnPx)), Math.max(10, Math.min(h-10, yUnPx)), 5.5, 0, Math.PI * 2); ctx.fill();

  const xCbPx = w/2 + xStrikeMm * 3.8;
  const yCbPx = h/2 - yStrikeMm * 2.2;
  ctx.fillStyle = '#22d3ee';
  ctx.beginPath(); ctx.arc(Math.max(16, Math.min(w-16, xCbPx)), Math.max(10, Math.min(h-10, yCbPx)), 5.0, 0, Math.PI * 2); ctx.fill();

  const lbl = document.getElementById('strikeSpotLbl');
  if (lbl) lbl.textContent = `x_hit = ${fmtSign(xStrikeMm, 1)} mm (Eff ${corEff.toFixed(1)}%)`;
}
