/**
 * V4.0 overlay/cumulative-stats controls, Stimpmeter Ramp Mode toggle, JSON
 * export/load and the window.PuttingSimulatorAPI / page bootstrap for the
 * standalone HTML studio (epic #14141, step 8e). Classic script (no ES
 * modules; opens from file://). Depends on globals from js/state.js (state),
 * js/putting_physics.js (PuttingPhysics.simulateSinglePuttRK4 for the
 * Stimpmeter calibration roll), js/green3d.js (rebuildGreenDispersionBatch),
 * js/charts2d.js (drawGreenRollSimulator, drawSGCaptureChart),
 * js/controls_a.js (savedScenarios, syncUI) and js/controls_c.js
 * (handleResize, animate); this file is loaded last and runs
 * window.onload, which kicks off the main requestAnimationFrame loop.
 */
// 7. V4.0 CONTROLS, STIMPMETER MODE, JSON EXPORT/LOAD & MCP AGENT API
// =========================================================================
function sync2DOverlayCheckboxes() {
  const cHole3D = document.getElementById('chkStartHoleLine');
  const cHole2D = document.getElementById('chkStartHoleLine2D');
  const cGrid3D = document.getElementById('chkGreenGrid');
  const cGrid2D = document.getElementById('chkGreenGrid2D');
  const cFilt2D = document.getElementById('chkPhotoFilter2D');
  if (cHole3D) cHole3D.checked = !!state.showStartHoleLine;
  if (cHole2D) cHole2D.checked = !!state.showStartHoleLine;
  if (cGrid3D) cGrid3D.checked = !!state.showGreenGrid;
  if (cGrid2D) cGrid2D.checked = !!state.showGreenGrid;
  if (cFilt2D) cFilt2D.checked = !!state.showPhotoFilter;
}

['chkStartHoleLine', 'chkStartHoleLine2D'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', (e) => {
    state.showStartHoleLine = e.target.checked;
    sync2DOverlayCheckboxes();
    drawGreenRollSimulator();
    syncUI();
  });
});

['chkGreenGrid', 'chkGreenGrid2D'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', (e) => {
    state.showGreenGrid = e.target.checked;
    sync2DOverlayCheckboxes();
    drawGreenRollSimulator();
    syncUI();
  });
});

const elPhoto2D = document.getElementById('chkPhotoFilter2D');
if (elPhoto2D) elPhoto2D.addEventListener('change', (e) => {
  state.showPhotoFilter = e.target.checked;
  drawGreenRollSimulator();
});

const elDistSlider = document.getElementById('puttDistSlider');
if (elDistSlider) elDistSlider.addEventListener('input', (e) => {
  state.puttDistFt = parseFloat(e.target.value);
  document.getElementById('puttDistValLbl').textContent = `${state.puttDistFt.toFixed(1)} ft`;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

const elCapSlider = document.getElementById('captureSpeedSlider');
if (elCapSlider) elCapSlider.addEventListener('input', (e) => {
  state.captureSpeedMs = parseFloat(e.target.value);
  document.getElementById('captureSpeedValLbl').textContent = `${state.captureSpeedMs.toFixed(2)} m/s`;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

const elRoughSlider = document.getElementById('roughnessSlider');
if (elRoughSlider) elRoughSlider.addEventListener('input', (e) => {
  state.surfaceRoughnessMm = parseFloat(e.target.value);
  document.getElementById('roughnessValLbl').textContent = `${state.surfaceRoughnessMm.toFixed(2)} mm`;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

const elReadSlider = document.getElementById('greenReadSlider');
if (elReadSlider) elReadSlider.addEventListener('input', (e) => {
  state.greenReadErrorPct = parseFloat(e.target.value);
  document.getElementById('greenReadValLbl').textContent = `${state.greenReadErrorPct.toFixed(2)}%`;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

const btn100 = document.getElementById('runBatch100Btn');
if (btn100) btn100.addEventListener('click', () => {
  state.stimpmeterMode = false;
  state.nTrials = 100;
  state.mcBatchSeed += 1;
  state.rollT = 3.6;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

const btn1000 = document.getElementById('runBatch1000Btn');
if (btn1000) btn1000.addEventListener('click', () => {
  state.stimpmeterMode = false;
  state.nTrials = 1000;
  state.mcBatchSeed += 1;
  state.rollT = 3.6;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

const btnSpread45 = document.getElementById('runSpread45Btn');
if (btnSpread45) btnSpread45.addEventListener('click', () => {
  state.stimpmeterMode = false;
  state.nTrials = 250;
  const distNodes = [3, 6, 10, 15, 25, 35, 45];
  distNodes.forEach(dFt => {
    state.puttDistFt = dFt;
    state.mcBatchSeed += 1;
    rebuildGreenDispersionBatch();
  });
  state.puttDistFt = 12.0;
  if (elDistSlider) elDistSlider.value = 12.0;
  document.getElementById('puttDistValLbl').textContent = '12.0 ft (0-45 ft Spread Complete)';
  state.rollT = 3.6;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

const btnResetCum = document.getElementById('resetCumulativeBtn');
if (btnResetCum) btnResetCum.addEventListener('click', () => {
  state.cumulativeMakes = 0;
  state.cumulativeAttempts = 0;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
});

const btnStimpMode = document.getElementById('stimpmeterModeBtn');
if (btnStimpMode) btnStimpMode.addEventListener('click', () => {
  state.stimpmeterMode = !state.stimpmeterMode;
  btnStimpMode.textContent = state.stimpmeterMode ? 'Exit Stimpmeter Mode' : 'Stimpmeter Ramp Mode';
  if (state.stimpmeterMode) {
    state.nTrials = 12;
    const rRoll = PuttingPhysics.simulateSinglePuttRK4({
      vClubMps: 1.83,
      aimDeg: 0.0,
      pathErrDeg: 0,
      faceErrDeg: 0,
      strikeMm: 0,
      moiKgM2: 5000 / 1.0e7,
      stimpFt: state.stimpFt,
      upPct: state.uphillSlopePct,
      crossPct: state.crossSlopePct,
      trialSeed: 0,
      distFt: state.puttDistFt,
      roughnessMm: state.surfaceRoughnessMm !== undefined ? state.surfaceRoughnessMm : 0.35,
      isStimpRamp: true,
    });
    const effStimp = (rRoll.finalX / 0.3048).toFixed(1);
    const optAim = (state.crossSlopePct * 2.65 * ((state.puttDistFt || 10.0) / 10.0)).toFixed(1);
    const optCap = (state.tauPreload >= 0.40 ? 0.65 : 0.38).toFixed(2);
    const elStimp = document.getElementById('valStimpReadout');
    if (elStimp) elStimp.textContent = `Eff. Stimp ${effStimp} ft • Opt Aim +${optAim}° • v_ent* ${optCap} m/s`;
  } else {
    state.nTrials = 40;
  }
  state.rollT = 3.6;
  rebuildGreenDispersionBatch();
  drawGreenRollSimulator();
  syncUI();
});

// Export & Load Simulation JSON
function exportSimulationJSON() {
  const payload = {
    schema_version: '4.0.0',
    suite: 'UpstreamDrift_3D_Putting_Studio_v4',
    timestamp: new Date().toISOString(),
    state: { ...state },
    lastBatchStats: { ...lastBatchStats },
    savedScenarios: [...savedScenarios]
  };
  return JSON.stringify(payload, null, 2);
}

function loadSimulationJSON(jsonStr) {
  try {
    const parsed = JSON.parse(jsonStr);
    if (parsed && parsed.state) {
      Object.assign(state, parsed.state);
      if (parsed.savedScenarios && Array.isArray(parsed.savedScenarios)) {
        savedScenarios = parsed.savedScenarios;
        renderScenarioTable();
      }
      syncCheckboxesFromState();
      sync2DOverlayCheckboxes();
      rebuildPlaneMesh(state.psiLieDeg);
      rebuildGreenDispersionBatch();
      drawGreenRollSimulator();
      syncUI();
      return { status: 'ok', state };
    }
  } catch (err) {
    return { status: 'error', message: String(err) };
  }
  return { status: 'error', message: 'Invalid JSON payload' };
}

const btnExport = document.getElementById('exportJsonBtn');
if (btnExport) btnExport.addEventListener('click', () => {
  const jsonStr = exportSimulationJSON();
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `putting_simulation_v4_${Math.round(state.puttDistFt)}ft.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

const btnLoad = document.getElementById('loadJsonBtn');
const inpLoad = document.getElementById('loadJsonInput');
if (btnLoad && inpLoad) {
  btnLoad.addEventListener('click', () => inpLoad.click());
  inpLoad.addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => loadSimulationJSON(String(ev.target.result || ''));
    reader.readAsText(file);
  });
}

// Programmatic MCP & JSON-RPC 2.0 Agent Control Interface (window.PuttingSimulatorAPI)
window.PuttingSimulatorAPI = {
  version: '4.0.0',
  getState: () => ({ ...state, lastBatchStats: { ...lastBatchStats } }),
  setState: (patch) => {
    Object.assign(state, patch || {});
    syncCheckboxesFromState();
    sync2DOverlayCheckboxes();
    rebuildPlaneMesh(state.psiLieDeg);
    rebuildGreenDispersionBatch();
    drawGreenRollSimulator();
    syncUI();
    return window.PuttingSimulatorAPI.getState();
  },
  runBatch: (nTrials = 100) => {
    state.stimpmeterMode = false;
    state.nTrials = Math.max(10, Math.min(2000, nTrials));
    state.mcBatchSeed += 1;
    state.rollT = 3.6;
    rebuildGreenDispersionBatch();
    drawGreenRollSimulator();
    syncUI();
    return { nTrials: state.nTrials, ...lastBatchStats };
  },
  runDistanceSpread0to45: () => {
    if (btnSpread45) btnSpread45.click();
    return window.PuttingSimulatorAPI.getState();
  },
  runStimpmeterCalibration: () => {
    state.stimpmeterMode = false;
    if (btnStimpMode) btnStimpMode.click();
    return window.PuttingSimulatorAPI.getState();
  },
  exportResultsJSON: exportSimulationJSON,
  loadResultsJSON: loadSimulationJSON,
  listTools: () => [
    { name: 'set_simulation_parameters', description: 'Set preload torque, distance (0-45 ft), slope, roughness, capture speed, or green-read error.' },
    { name: 'run_parallel_putt_batch', description: 'Run 100 or 1000 Monte Carlo putts on the undulating green.' },
    { name: 'run_stimpmeter_calibration', description: 'Roll USGA Stimpmeter calibration balls and compute optimal green read.' },
    { name: 'export_simulation_json', description: 'Export full simulation state and Strokes Gained metrics as JSON.' }
  ],
  callTool: (name, args = {}) => {
    if (name === 'set_simulation_parameters') return window.PuttingSimulatorAPI.setState(args);
    if (name === 'run_parallel_putt_batch') return window.PuttingSimulatorAPI.runBatch(args.nTrials || 100);
    if (name === 'run_stimpmeter_calibration') return window.PuttingSimulatorAPI.runStimpmeterCalibration();
    if (name === 'export_simulation_json') return { json: exportSimulationJSON() };
    throw new Error('Unknown MCP tool: ' + name);
  },
  handleJsonRpc: (request) => {
    const id = request && request.id !== undefined ? request.id : 1;
    try {
      if (request.method === 'tools/list') return { jsonrpc: '2.0', id, result: { tools: window.PuttingSimulatorAPI.listTools() } };
      if (request.method === 'tools/call') return { jsonrpc: '2.0', id, result: window.PuttingSimulatorAPI.callTool(request.params.name, request.params.arguments) };
      return { jsonrpc: '2.0', id, error: { code: -32601, message: 'Method not found' } };
    } catch (err) {
      return { jsonrpc: '2.0', id, error: { code: -32602, message: String(err) } };
    }
  }
};

window.onload = () => {
  handleResize();
  rebuildGreenDispersionBatch();
  drawSGCaptureChart();
  requestAnimationFrame(animate);
};
