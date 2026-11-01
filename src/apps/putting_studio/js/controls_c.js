/**
 * Remaining physics/view sliders & selects, video-tab transport controls,
 * turnover-doc clipboard copy, the legend show/hide toggle, perturbation
 * & green-preset controls, window resize handling and the main
 * requestAnimationFrame loop for the standalone HTML studio (epic
 * #14141, step 8g). Classic script (no ES modules; opens from file://).
 * Depends on globals from js/state.js (state, T_TOTAL), js/scene3d_a.js
 * (container, camera, renderer), js/green3d.js (rebuildGreenDispersionBatch)
 * via js/putting_physics.js (PuttingPhysics.simulateSinglePuttRK4 for the
 * aim-point solver) and js/controls_a.js/js/controls_b.js (syncUI,
 * fmtSign, syncCheckboxesFromState); calls drawGreenRollSimulator,
 * defined in js/charts2d.js (loaded earlier). Exposes handleResize,
 * animate and toggleAllLegends as globals for js/io.js.
 */
// Sliders & Selects
document.getElementById('preloadSlider').addEventListener('input', (e) => {
  state.tauPreload = parseFloat(e.target.value);
  document.getElementById('preloadValLbl').textContent = `${state.tauPreload.toFixed(2)} N·m (${(state.tauPreload / d_L).toFixed(1)} N/finger)`;
  rebuildGreenDispersionBatch();
  syncUI();
});
document.getElementById('lieSlider').addEventListener('input', (e) => {
  state.psiLieDeg = parseFloat(e.target.value);
  document.getElementById('lieValLbl').textContent = `${state.psiLieDeg.toFixed(1)}° (${(90 - state.psiLieDeg).toFixed(0)}° tilt)`;
  rebuildPlaneMesh(state.psiLieDeg);
  syncUI();
});
document.getElementById('putterTypeSelect').addEventListener('change', (e) => {
  state.putterType = e.target.value;
  rebuildGreenDispersionBatch();
  syncUI();
});
document.getElementById('vecModeSelect').addEventListener('change', (e) => {
  state.vecMode = e.target.value;
  syncUI();
});
document.getElementById('toggleGhostBtn').addEventListener('click', () => {
  state.showGhost = !state.showGhost;
  syncCheckboxesFromState();
  syncUI();
});

// Video Player Tab Controls
const vid = document.getElementById('scientificVideo');
document.getElementById('vidStepBackBtn').addEventListener('click', () => { vid.pause(); vid.currentTime = Math.max(0, vid.currentTime - 1/30); });
document.getElementById('vidStepFwdBtn').addEventListener('click', () => { vid.pause(); vid.currentTime = Math.min(vid.duration || 6, vid.currentTime + 1/30); });
document.getElementById('vidSlowBtn').addEventListener('click', () => { vid.playbackRate = 0.5; vid.play(); });
document.getElementById('vidNormBtn').addEventListener('click', () => { vid.playbackRate = 1.0; vid.play(); });

// Turnover Clipboard Copy
document.getElementById('copyTurnoverBtn').addEventListener('click', (e) => {
  const text = document.getElementById('turnoverContentText').innerText;
  const ta = document.createElement('textarea');
  ta.value = text;
  document.body.appendChild(ta);
  ta.select();
  document.execCommand('copy');
  document.body.removeChild(ta);
  e.target.textContent = 'Copied to Clipboard!';
  setTimeout(() => { e.target.textContent = 'Copy Turnover Spec to Clipboard'; }, 2000);
});

// Unified Show / Hide Legend Toggle (Toolbar Button + Floating Viewport Pill + 'L' Key Shortcut)
function toggleAllLegends() {
  state.showLegend = !state.showLegend;
  const leg = document.getElementById('legendOverlay');
  const gLeg = document.getElementById('greenLegendBox');
  const topBtn = document.getElementById('toggleLegendBtn');
  const floatTxt = null;
  const floatBtn = null;
  if (leg) leg.style.display = state.showLegend ? 'block' : 'none';
  if (gLeg) gLeg.style.display = state.showLegend ? 'block' : 'none';
  if (topBtn) topBtn.innerHTML = state.showLegend ? 'Hide Legend' : 'Show Legend';
  if (floatTxt) floatTxt.textContent = state.showLegend ? 'Hide Legend' : 'Show Legend';
  if (floatBtn) {
    floatBtn.className = state.showLegend
      ? 'legend-pill-glow flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-950/92 hover:bg-slate-900 border-2 border-emerald-400/80 text-emerald-200 font-extrabold text-xs cursor-pointer transition'
      : 'flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-900/90 hover:bg-slate-800 border-2 border-amber-400/80 text-amber-200 font-extrabold text-xs cursor-pointer transition shadow-lg';
  }
}
document.getElementById('toggleLegendBtn').addEventListener('click', toggleAllLegends);

window.addEventListener('keydown', (e) => {
  if ((e.key === 'l' || e.key === 'L') && document.activeElement && document.activeElement.tagName !== 'INPUT') {
    toggleAllLegends();
  }
});

document.getElementById('toggleAdvBtn').addEventListener('click', (e) => {
  state.showAdv = !state.showAdv;
  const adv = document.getElementById('advancedFeaturesPanel');
  if (adv) adv.style.display = state.showAdv ? 'block' : 'none';
  e.target.textContent = state.showAdv ? 'Adv. Noise/Strike (ON)' : 'Adv. Noise/Strike (OFF)';
});

document.getElementById('offPreloadSlider').addEventListener('input', (e) => {
  state.tauOffPreload = parseFloat(e.target.value);
  const lSupport = (state.tauOffPreload + 0.985).toFixed(2);
  const rStab = (-state.tauOffPreload).toFixed(2);
  document.getElementById('offPreloadValLbl').textContent = `${state.tauOffPreload.toFixed(2)} N·m (Left +${lSupport} / Right ${rStab})`;
  rebuildGreenDispersionBatch();
  syncUI();
});

document.getElementById('pertImpulseSlider').addEventListener('input', (e) => {
  state.pertImpulseNm = parseFloat(e.target.value);
  document.getElementById('pertImpulseValLbl').textContent = `${state.pertImpulseNm.toFixed(2)} N·m`;
  rebuildGreenDispersionBatch();
  syncUI();
});
document.getElementById('toeOffsetSlider').addEventListener('input', (e) => {
  state.toeOffsetMm = parseFloat(e.target.value);
  document.getElementById('toeOffsetValLbl').textContent = `+${state.toeOffsetMm.toFixed(1)} mm`;
  rebuildGreenDispersionBatch();
  syncUI();
});
document.getElementById('pertZeroBtn').addEventListener('click', () => {
  state.pertImpulseNm = 0.0;
  state.toeOffsetMm = 0.0;
  state.noiseScale = 0.0;
  document.getElementById('pertImpulseSlider').value = 0.0;
  document.getElementById('toeOffsetSlider').value = 0.0;
  document.getElementById('noiseSlider').value = 0.0;
  document.getElementById('pertImpulseValLbl').textContent = '0.00 N·m';
  document.getElementById('toeOffsetValLbl').textContent = '+0.0 mm';
  document.getElementById('noiseValLbl').textContent = '0.00x (Unperturbed Exact Match)';
  rebuildGreenDispersionBatch();
  syncUI();
});
document.getElementById('pertImpulseBtn').addEventListener('click', () => {
  state.pertImpulseNm = 0.25;
  state.t = 0.817;
  document.getElementById('pertImpulseSlider').value = 0.25;
  document.getElementById('pertImpulseValLbl').textContent = '0.25 N·m';
  rebuildGreenDispersionBatch();
  syncUI();
});
document.getElementById('pertToeBtn').addEventListener('click', () => {
  state.toeOffsetMm = 10.0;
  state.t = 0.817;
  document.getElementById('toeOffsetSlider').value = 10.0;
  document.getElementById('toeOffsetValLbl').textContent = '+10.0 mm';
  rebuildGreenDispersionBatch();
  syncUI();
});
document.getElementById('solveAimpointBtn').addEventListener('click', () => {
  // Fast analytical-seed + secant search for optimal uphill aim compensation angle
  let bestAim = state.crossSlopePct * 2.68;
  let bestMiss = 999.0;
  const vNom = 1.32 + 0.03 * (state.uphillSlopePct / 0.5);
  for (let cand = bestAim - 2.0; cand <= bestAim + 2.0; cand += 0.2) {
    const r = PuttingPhysics.simulateSinglePuttRK4({
      vClubMps: vNom,
      aimDeg: cand,
      pathErrDeg: 0,
      faceErrDeg: 0,
      strikeMm: 0,
      moiKgM2: 5000 / 1.0e7,
      stimpFt: state.stimpFt,
      upPct: state.uphillSlopePct,
      crossPct: state.crossSlopePct,
      distFt: state.puttDistFt || 10.0,
      roughnessMm: state.surfaceRoughnessMm !== undefined ? state.surfaceRoughnessMm : 0.35,
    });
    const miss = r.holed ? r.missDistCm * 0.1 : r.missDistCm;
    if (miss < bestMiss) { bestMiss = miss; bestAim = cand; }
  }
  state.aimAngleDeg = parseFloat(bestAim.toFixed(1));
  document.getElementById('aimAngleSlider').value = state.aimAngleDeg;
  document.getElementById('aimAngleValLbl').textContent = `${state.aimAngleDeg >= 0 ? '+' : ''}${state.aimAngleDeg.toFixed(1)}° Model Aim`;
  state.rollT = 3.6;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

document.getElementById('noiseSlider').addEventListener('input', (e) => {
  state.noiseScale = parseFloat(e.target.value);
  document.getElementById('noiseValLbl').textContent = `${state.noiseScale.toFixed(2)}x (Smooth C² OU)`;
  rebuildGreenDispersionBatch();
  syncUI();
});

document.getElementById('presetFlatGreenBtn').addEventListener('click', () => {
  state.crossSlopePct = 0.0; state.uphillSlopePct = 0.0; state.aimAngleDeg = 0.0;
  document.getElementById('crossSlopeSlider').value = 0.0;
  document.getElementById('uphillSlopeSlider').value = 0.0;
  document.getElementById('aimAngleSlider').value = 0.0;
  document.getElementById('crossSlopeValLbl').textContent = '0.0% (Flat Green)';
  document.getElementById('uphillSlopeValLbl').textContent = '0.0% (Level)';
  document.getElementById('aimAngleValLbl').textContent = '0.0° Straight Aim';
  state.rollT = 3.6;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

document.getElementById('presetSlopedGreenBtn').addEventListener('click', () => {
  state.crossSlopePct = 2.2; state.uphillSlopePct = 0.4; state.aimAngleDeg = 6.0;
  document.getElementById('crossSlopeSlider').value = 2.2;
  document.getElementById('uphillSlopeSlider').value = 0.4;
  document.getElementById('aimAngleSlider').value = 6.0;
  document.getElementById('crossSlopeValLbl').textContent = '2.2% (Left-to-Right Break)';
  document.getElementById('uphillSlopeValLbl').textContent = '+0.4% (Slight Uphill)';
  document.getElementById('aimAngleValLbl').textContent = '+6.0° Uphill Aim';
  state.rollT = 3.6;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

document.getElementById('rerollMonteCarloBtn').addEventListener('click', () => {
  state.mcBatchSeed += 1;
  state.rollT = 0.0;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

document.getElementById('rollPlayPauseBtn').addEventListener('click', () => {
  state.rollT = 0.0;
  drawGreenRollSimulator();
});

document.getElementById('rollTimeScrubber').addEventListener('input', (e) => {
  state.rollT = parseFloat(e.target.value);
  document.getElementById('rollTimeLbl').textContent = `t = ${state.rollT.toFixed(2)} s`;
  drawGreenRollSimulator();
});

document.getElementById('crossSlopeSlider').addEventListener('input', (e) => {
  state.crossSlopePct = parseFloat(e.target.value);
  document.getElementById('crossSlopeValLbl').textContent = `${state.crossSlopePct.toFixed(1)}% Cross-Slope`;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

document.getElementById('uphillSlopeSlider').addEventListener('input', (e) => {
  state.uphillSlopePct = parseFloat(e.target.value);
  document.getElementById('uphillSlopeValLbl').textContent = `${fmtSign(state.uphillSlopePct, 1)}% Grade`;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

document.getElementById('stimpSlider').addEventListener('input', (e) => {
  state.stimpFt = parseFloat(e.target.value);
  document.getElementById('stimpValLbl').textContent = `${state.stimpFt.toFixed(1)} ft (μ = ${(0.560/state.stimpFt).toFixed(4)})`;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

document.getElementById('aimAngleSlider').addEventListener('input', (e) => {
  state.aimAngleDeg = parseFloat(e.target.value);
  document.getElementById('aimAngleValLbl').textContent = `${fmtSign(state.aimAngleDeg, 1)}° Aim Angle`;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

rebuildGreenDispersionBatch();
renderScenarioTable();

function handleResize() {
  if (container.clientWidth > 0 && container.clientHeight > 0) {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }
  syncUI();
}
window.addEventListener('resize', handleResize);

// Main Animation Loop
let lastTime = performance.now();
function animate(now) {
  const dtSec = Math.min(0.05, (now - lastTime) / 1000.0);
  lastTime = now;
  if (state.playing) {
    state.t += dtSec * state.speed;
    if (state.t > T_TOTAL) state.t = 0.0;
    syncUI();
  }
  if (!viewGreenRoll.classList.contains('hidden') && state.rollT < 3.6) {
    state.rollT = Math.min(3.6, state.rollT + dtSec * 1.1);
    document.getElementById('rollTimeScrubber').value = state.rollT.toFixed(2);
    document.getElementById('rollTimeLbl').textContent = `t = ${state.rollT.toFixed(2)} s`;
    drawGreenRollSimulator();
  }
  requestAnimationFrame(animate);
}

