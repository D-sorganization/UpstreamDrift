/**
 * Tab switching, playback transport (play/pause/scrubber/jump/speed),
 * camera-preset buttons and the scene-component visibility checkboxes
 * for the standalone HTML studio (epic #14141, step 8e). Classic script
 * (no ES modules; opens from file://). Depends on globals from
 * js/state.js (state), js/scene3d_a.js (camSpherical, updateCameraPosition,
 * zoomToFitScene), js/scene3d_b.js (rebuildPlaneMesh), js/green3d.js
 * (rebuildGreenDispersionBatch) and js/controls_a.js (syncUI,
 * savedScenarios, renderScenarioTable, saveCurrentScenarioSnapshot);
 * calls handleResize and drawGreenRollSimulator, which are defined in
 * js/controls_c.js (loaded after this file but only invoked later, from
 * inside event-handler closures, so the forward reference resolves at
 * call time through the shared global scope). Exposes activateTab,
 * setPlaying, compCheckboxMap and syncCheckboxesFromState as globals for
 * js/controls_c.js and js/io.js.
 */
// Tab switching
const tabBtn3D = document.getElementById('tabBtn3D');
const tabBtnGreenRoll = document.getElementById('tabBtnGreenRoll');
const tabBtnVideo = document.getElementById('tabBtnVideo');
const tabBtnTurnover = document.getElementById('tabBtnTurnover');
const view3DStudio = document.getElementById('view3DStudio');
const viewGreenRoll = document.getElementById('viewGreenRoll');
const viewVideoPlayer = document.getElementById('viewVideoPlayer');
const viewTurnoverDoc = document.getElementById('viewTurnoverDoc');

function activateTab(tabName) {
  view3DStudio.classList.add('hidden');
  viewGreenRoll.classList.add('hidden');
  viewVideoPlayer.classList.add('hidden');
  viewTurnoverDoc.classList.add('hidden');
  [tabBtn3D, tabBtnGreenRoll, tabBtnVideo, tabBtnTurnover].forEach(b => {
    b.className = 'px-3 py-1.5 rounded-md text-xs font-semibold text-slate-300 hover:text-white hover:bg-slate-800 transition';
  });
  if (tabName === '3d') {
    view3DStudio.classList.remove('hidden');
    tabBtn3D.className = 'px-3 py-1.5 rounded-md text-xs font-semibold bg-sky-600 text-white transition';
    handleResize();
  } else if (tabName === 'green') {
    viewGreenRoll.classList.remove('hidden');
    tabBtnGreenRoll.className = 'px-3 py-1.5 rounded-md text-xs font-semibold bg-emerald-600 text-white transition';
    rebuildGreenDispersionBatch();
    drawGreenRollSimulator();
    renderScenarioTable();
  } else if (tabName === 'video') {
    viewVideoPlayer.classList.remove('hidden');
    viewVideoPlayer.classList.add('flex');
    tabBtnVideo.className = 'px-3 py-1.5 rounded-md text-xs font-semibold bg-sky-600 text-white transition';
  } else {
    viewTurnoverDoc.classList.remove('hidden');
    viewTurnoverDoc.classList.add('flex');
    tabBtnTurnover.className = 'px-3 py-1.5 rounded-md text-xs font-semibold bg-sky-600 text-white transition';
  }
}

tabBtn3D.addEventListener('click', () => activateTab('3d'));
tabBtnGreenRoll.addEventListener('click', () => activateTab('green'));
tabBtnVideo.addEventListener('click', () => activateTab('video'));
tabBtnTurnover.addEventListener('click', () => activateTab('turnover'));

// Play/Pause & Scrubber
const playPauseBtn = document.getElementById('playPauseBtn');
function setPlaying(p) {
  state.playing = p;
  document.getElementById('playIcon').innerHTML = p ? '&#10074;&#10074;' : '&#9654;';
  document.getElementById('playText').textContent = p ? 'Pause' : 'Play';
}
setPlaying(true);

playPauseBtn.addEventListener('click', () => setPlaying(!state.playing));
document.getElementById('timeScrubber').addEventListener('input', (e) => {
  setPlaying(false);
  state.t = parseFloat(e.target.value);
  syncUI();
});

document.querySelectorAll('.jumpBtn').forEach(btn => {
  btn.addEventListener('click', () => {
    setPlaying(false);
    state.t = parseFloat(btn.getAttribute('data-jump'));
    syncUI();
  });
});

document.querySelectorAll('.speedBtn').forEach(btn => {
  btn.addEventListener('click', () => {
    state.speed = parseFloat(btn.getAttribute('data-speed'));
    document.querySelectorAll('.speedBtn').forEach(b => {
      b.className = 'speedBtn px-2 py-1 rounded bg-slate-900 border border-slate-700 text-slate-300';
    });
    btn.className = 'speedBtn px-2 py-1 rounded bg-sky-600/30 border border-sky-500 text-sky-200 font-semibold';
  });
});

// Camera Presets & Zoom-to-Fit Controls (Expanded up to 8.50m)
document.getElementById('camGripBtn').addEventListener('click', () => {
  camSpherical.theta = -0.65; camSpherical.phi = 1.15;
  zoomToFitScene('grip');
});
document.getElementById('camFullBtn').addEventListener('click', () => {
  camSpherical.theta = -0.42; camSpherical.phi = 1.18;
  zoomToFitScene('full');
});
document.getElementById('zoomFitBtn').addEventListener('click', () => zoomToFitScene('full'));
document.getElementById('zoomFitOverlayBtn').addEventListener('click', () => zoomToFitScene('full'));
document.getElementById('zoomInBtn').addEventListener('click', () => {
  camSpherical.radius = Math.max(0.22, camSpherical.radius * 0.80);
  updateCameraPosition();
});
document.getElementById('zoomOutBtn').addEventListener('click', () => {
  camSpherical.radius = Math.min(22.00, camSpherical.radius * 1.25);
  updateCameraPosition();
});
document.getElementById('zoomRangeSlider').addEventListener('input', (e) => {
  camSpherical.radius = parseFloat(e.target.value);
  updateCameraPosition();
});
document.getElementById('buttArcSlider').addEventListener('input', (e) => {
  state.buttArcRadius = parseFloat(e.target.value);
  document.getElementById('buttArcValLbl').textContent = state.buttArcRadius > 0.01
    ? `${state.buttArcRadius.toFixed(2)} m Arc (Natural Hand Path)`
    : `0.00 m (Fixed Butt Pivot)`;
  rebuildPlaneMesh(state.psiLieDeg);
  syncUI();
});
document.getElementById('camDtlBtn').addEventListener('click', () => {
  camSpherical.radius = 2.10; camSpherical.theta = -Math.PI / 2; camSpherical.phi = 1.32;
  camSpherical.target.set(0.35, -0.38, 0.14); updateCameraPosition();
});
document.getElementById('camFaceBtn').addEventListener('click', () => {
  camSpherical.radius = 2.20; camSpherical.theta = 0.0; camSpherical.phi = 1.22;
  camSpherical.target.set(0.25, -0.38, 0.14); updateCameraPosition();
});

// Component Visibility Checkboxes Wiring
const compCheckboxMap = [
  ['chkClub', 'showClub'],
  ['chkForceVecs', 'showForceVecs'],
  ['chkHandTorques', 'showHandTorques'],
  ['chkNetTorque', 'showNetTorque'],
  ['chkImpactVecs', 'showImpactVecs'],
  ['chkButtArc', 'showButtArc'],
  ['chkStrokePlane', 'showStrokePlane'],
  ['chkBallTraces3D', 'showBallTraces3D'],
  ['chkGreenEnv', 'showGreenEnv'],
  ['chkStartHoleLine', 'showStartHoleLine'],
  ['chkGreenGrid', 'showGreenGrid'],
  ['chkGhostClub', 'showGhost'],
  ['chkVecLabels', 'showVecLabels'],
  ['chkLiveTorqueCards', 'showLiveTorqueCards']
];

function syncCheckboxesFromState() {
  compCheckboxMap.forEach(([id, key]) => {
    const el = document.getElementById(id);
    if (el) el.checked = !!state[key];
  });
  const wrapper = document.getElementById('liveTorqueCardsWrapper');
  if (wrapper) wrapper.classList.toggle('hidden', !state.showLiveTorqueCards);
  const ghostBtn = document.getElementById('toggleGhostBtn');
  if (ghostBtn) {
    ghostBtn.textContent = state.showGhost ? 'Showing Ghost Droop (ON)' : 'Ghost Club Hidden (OFF)';
    ghostBtn.className = state.showGhost
      ? 'w-full py-1.5 px-2 rounded bg-orange-500/20 border border-orange-500/50 text-orange-300 text-xs font-semibold'
      : 'w-full py-1.5 px-2 rounded bg-slate-900 border border-slate-700 text-slate-400 text-xs font-semibold';
  }
}
syncCheckboxesFromState();

compCheckboxMap.forEach(([id, key]) => {
  const el = document.getElementById(id);
  if (el) {
    el.addEventListener('change', (e) => {
      state[key] = e.target.checked;
      syncCheckboxesFromState();
      syncUI();
    });
  }
});

document.getElementById('showAllCompBtn').addEventListener('click', () => {
  compCheckboxMap.forEach(([_, key]) => { state[key] = true; });
  syncCheckboxesFromState();
  syncUI();
});

document.getElementById('hideAllCompBtn').addEventListener('click', () => {
  compCheckboxMap.forEach(([_, key]) => { state[key] = false; });
  state.showClub = true;
  state.showGreenEnv = true;
  syncCheckboxesFromState();
  syncUI();
});

document.getElementById('nTrialsSlider').addEventListener('input', (e) => {
  state.nTrials = parseInt(e.target.value, 10);
  document.getElementById('nTrialsValLbl').textContent = `${state.nTrials} Trials Overlaid`;
  rebuildGreenDispersionBatch();
  syncUI();
});

document.getElementById('saveScenarioQuickBtn').addEventListener('click', () => {
  saveCurrentScenarioSnapshot();
  activateTab('green');
});
document.getElementById('saveScenarioBtn').addEventListener('click', () => {
  saveCurrentScenarioSnapshot();
});
document.getElementById('clearScenariosBtn').addEventListener('click', () => {
  savedScenarios = savedScenarios.slice(0, 2);
  renderScenarioTable();
});

