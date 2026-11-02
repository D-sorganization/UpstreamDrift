/**
 * 2D telemetry canvases: torque/off-plane strip charts, the 2D oblong
 * green-roll simulator canvas (with its own undulation shading via
 * globalThis.PuttingPhysics.evalUndulatingGreen) and the Strokes-Gained
 * capture-speed chart for the standalone HTML studio (epic #14141, step
 * 8j). Classic script (no ES modules; opens from file://). Depends on
 * globals from js/state.js (state, evalPhysicsAtTime, chartTimes, N_PTS,
 * T_STROKE), js/putting_physics.js (PuttingPhysics.evalUndulatingGreen)
 * and js/green3d.js (greenBatchCB, greenBatchUN) and exposes
 * drawTelemetryCharts, drawGreenRollSimulator and drawSGCaptureChart as
 * globals for the js/controls_*.js and js/io.js files.
 */
function drawTelemetryCharts() {
  const c1 = document.getElementById('torqueChartCanvas');
  const c2 = document.getElementById('phiChartCanvas');
  if (!c1 || !c2) return;

  [c1, c2].forEach(c => {
    const rect = c.parentElement.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      c.width = rect.width * window.devicePixelRatio;
      c.height = rect.height * window.devicePixelRatio;
    }
  });

  const ctx1 = c1.getContext('2d');
  const ctx2 = c2.getContext('2d');
  const w1 = c1.width, h1 = c1.height;
  const w2 = c2.width, h2 = c2.height;
  ctx1.clearRect(0, 0, w1, h1);
  ctx2.clearRect(0, 0, w2, h2);

  const padL = 36, padR = 12, padT = 12, padB = 22;
  const pw1 = w1 - padL - padR, ph1 = h1 - padT - padB;
  const pw2 = w2 - padL - padR, ph2 = h2 - padT - padB;

  ctx1.strokeStyle = '#334155'; ctx1.lineWidth = 1;
  const yZero1 = padT + ph1 * 0.5;
  ctx1.beginPath(); ctx1.moveTo(padL, yZero1); ctx1.lineTo(w1 - padR, yZero1); ctx1.stroke();

  ctx2.strokeStyle = '#334155'; ctx2.lineWidth = 1;
  const yZero2 = padT + ph2 * 0.5;
  ctx2.beginPath(); ctx1.moveTo(padL, yZero1);
  ctx2.beginPath(); ctx2.moveTo(padL, yZero2); ctx2.lineTo(w2 - padR, yZero2); ctx2.stroke();

  function plotCurve(ctx, pw, ph, yMin, yMax, color, dashed, getter) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2 * window.devicePixelRatio;
    if (dashed) ctx.setLineDash([5, 4]);
    ctx.beginPath();
    for (let i = 0; i <= N_PTS; i++) {
      const ti = chartTimes[i];
      const p = evalPhysicsAtTime(ti, state.tauPreload, state.psiLieDeg);
      const val = getter(p);
      const x = padL + (ti / T_STROKE) * pw;
      const y = padT + ph * (1.0 - (val - yMin) / (yMax - yMin));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  plotCurve(ctx1, pw1, ph1, -2.6, 2.6, '#f97316', false, p => p.tauR_in);
  plotCurve(ctx1, pw1, ph1, -2.6, 2.6, '#38bdf8', false, p => p.tauL_in);
  plotCurve(ctx1, pw1, ph1, -2.6, 2.6, '#4ade80', false, p => p.tauNet);
  plotCurve(ctx1, pw1, ph1, -2.6, 2.6, '#c084fc', true,  p => p.tauOffReq);

  plotCurve(ctx2, pw2, ph2, -0.55, 0.55, '#f97316', true,  p => p.phiSlkDeg);
  plotCurve(ctx2, pw2, ph2, -0.55, 0.55, '#22d3ee', false, p => p.phiPreDeg);

  const tClamped = Math.min(state.t, T_STROKE);
  const xPlay1 = padL + (tClamped / T_STROKE) * pw1;
  ctx1.strokeStyle = '#facc15'; ctx1.lineWidth = 2 * window.devicePixelRatio;
  ctx1.beginPath(); ctx1.moveTo(xPlay1, padT); ctx1.lineTo(xPlay1, padT + ph1); ctx1.stroke();

  const xPlay2 = padL + (tClamped / T_STROKE) * pw2;
  ctx2.strokeStyle = '#facc15'; ctx2.lineWidth = 2 * window.devicePixelRatio;
  ctx2.beginPath(); ctx2.moveTo(xPlay2, padT); ctx2.lineTo(xPlay2, padT + ph2); ctx2.stroke();
}

function drawOblongPath2D(ctx, cx, cy, rx, ry) {
  ctx.beginPath();
  const nPts = 120;
  for (let i = 0; i <= nPts; i++) {
    const ang = (i / nPts) * Math.PI * 2.0;
    const rMod = 1.0 + 0.12 * Math.cos(2.0 * ang) - 0.06 * Math.sin(3.0 * ang);
    const px = cx + rx * rMod * Math.cos(ang);
    const py = cy + ry * rMod * Math.sin(ang);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
}

function drawGreenRollSimulator() {
  const c = document.getElementById('greenRollCanvas');
  if (!c) return;
  c.classList.toggle('photoreal-green-canvas', !!state.showPhotoFilter);
  const rect = c.parentElement.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    c.width = rect.width * window.devicePixelRatio;
    c.height = rect.height * window.devicePixelRatio;
  }
  const dpr = window.devicePixelRatio || 1;
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);

  // Deep Augusta Dark Forest Green Rough Background
  const bgGrad = ctx.createRadialGradient(w * 0.50, h * 0.50, w * 0.08, w * 0.50, h * 0.50, w * 0.65);
  bgGrad.addColorStop(0.0, '#062614');
  bgGrad.addColorStop(0.6, '#03170c');
  bgGrad.addColorStop(1.0, '#010b05');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, w, h);

  // Sculpted Sand Bunkers with AO Lip Shadows flanking the oblong green
  ctx.save();
  ctx.shadowColor = 'rgba(0, 0, 0, 0.65)';
  ctx.shadowBlur = 10 * dpr;
  ctx.fillStyle = '#d6c088';
  drawOblongPath2D(ctx, w * 0.30, h * 0.13, w * 0.12, h * 0.08);
  ctx.fill();
  drawOblongPath2D(ctx, w * 0.74, h * 0.87, w * 0.13, h * 0.08);
  ctx.fill();
  ctx.restore();

  // Oblong Fringe Collar Apron
  ctx.fillStyle = '#09351c';
  drawOblongPath2D(ctx, w * 0.50, h * 0.50, w * 0.455, h * 0.435);
  ctx.fill();

  // Oblong Putting Green Interior (Clipped)
  ctx.save();
  drawOblongPath2D(ctx, w * 0.50, h * 0.50, w * 0.43, h * 0.405);
  const turfGrad = ctx.createLinearGradient(0, 0, w, h);
  turfGrad.addColorStop(0.0, '#0d4726');
  turfGrad.addColorStop(0.5, '#115e32');
  turfGrad.addColorStop(1.0, '#0a3a1e');
  ctx.fillStyle = turfGrad;
  ctx.fill();
  ctx.clip();

  // Mowed Tournament Stripes inside Oblong Green
  const nStripes = 18;
  const sw = w / nStripes;
  for (let i = 0; i < nStripes; i++) {
    if (i % 2 === 0) {
      ctx.fillStyle = 'rgba(52, 211, 153, 0.085)';
      ctx.fillRect(i * sw, 0, sw, h);
    }
  }

  const holeX = (state.puttDistFt || 10.0) * 0.3048;
  const xSpan = holeX * 1.26;
  const ySpan = Math.max(0.85, holeX * 0.28);
  const mapX = (xm) => (0.12 + (xm / xSpan) * 0.76) * w;
  const mapY = (ym) => (0.50 - (ym / ySpan) * 0.78) * h;

  // Non-Planar Mound & Swale Photorealistic Elevation Shading + Optional Topographic Slope Grid Overlay
  const nx = 22, ny = 12;
  for (let ix = 0; ix <= nx; ix++) {
    for (let iy = 0; iy <= ny; iy++) {
      const xm = (ix / nx) * xSpan;
      const ym = ((iy / ny) - 0.5) * 2.0 * ySpan;
      const g = PuttingPhysics.evalUndulatingGreen(xm, ym, holeX, state.uphillSlopePct, state.crossSlopePct);
      const px = mapX(xm), py = mapY(ym);
      if (state.showPhotoFilter) {
        const shade = g.dz_dx * 8.0 + g.dz_dy * 6.0;
        ctx.fillStyle = shade >= 0
          ? `rgba(110, 231, 183, ${Math.min(0.14, Math.abs(shade) * 0.22)})`
          : `rgba(2, 16, 8, ${Math.min(0.22, Math.abs(shade) * 0.30)})`;
        ctx.fillRect(px - w / nx * 0.55, py - h / ny * 0.55, w / nx * 1.1, h / ny * 1.1);
      }
      if (state.showGreenGrid) {
        // Local slope fall-line arrow
        const sMag = Math.hypot(g.dz_dx, g.dz_dy);
        if (sMag > 0.0005) {
          const ux = (-g.dz_dx / sMag) * 8.5 * dpr;
          const uy = (g.dz_dy / sMag) * 8.5 * dpr;
          ctx.strokeStyle = 'rgba(167, 243, 208, 0.28)';
          ctx.lineWidth = 1.0 * dpr;
          ctx.beginPath();
          ctx.moveTo(px - ux * 0.5, py - uy * 0.5);
          ctx.lineTo(px + ux * 0.5, py + uy * 0.5);
          ctx.stroke();
          ctx.fillStyle = 'rgba(52, 211, 153, 0.42)';
          ctx.beginPath();
          ctx.arc(px + ux * 0.5, py + uy * 0.5, 1.4 * dpr, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }

  if (state.showGreenGrid) {
    ctx.strokeStyle = 'rgba(52, 211, 153, 0.16)';
    ctx.lineWidth = 0.8 * dpr;
    for (let ix = 0; ix <= nx; ix += 2) {
      const xm = (ix / nx) * xSpan;
      ctx.beginPath(); ctx.moveTo(mapX(xm), 0); ctx.lineTo(mapX(xm), h); ctx.stroke();
    }
    for (let iy = 0; iy <= ny; iy += 2) {
      const ym = ((iy / ny) - 0.5) * 2.0 * ySpan;
      ctx.beginPath(); ctx.moveTo(0, mapY(ym)); ctx.lineTo(w, mapY(ym)); ctx.stroke();
    }
  }

  // Straight Line to the Hole & Initial Starting Direction Line (Toggleable via chkStartHoleLine)
  if (state.showStartHoleLine) {
    // 1. Straight Line directly from Ball (0,0) to Cup (holeX, 0)
    ctx.save();
    ctx.strokeStyle = 'rgba(248, 250, 252, 0.68)';
    ctx.setLineDash([7 * dpr, 5 * dpr]);
    ctx.lineWidth = 1.6 * dpr;
    ctx.beginPath();
    ctx.moveTo(mapX(0), mapY(0));
    ctx.lineTo(mapX(holeX), mapY(0));
    ctx.stroke();
    ctx.fillStyle = 'rgba(248, 250, 252, 0.85)';
    ctx.font = `600 ${10 * dpr}px "JetBrains Mono", monospace`;
    ctx.fillText(`Straight Line to Hole (${(state.puttDistFt || 10).toFixed(0)} ft)`, mapX(holeX * 0.36), mapY(0) + 13 * dpr);
    ctx.restore();

    // 2. Initial Starting Direction Ray (Aim / Launch Tangent)
    if (!state.stimpmeterMode) {
      const aimRad = (state.aimAngleDeg || 0.0) * Math.PI / 180.0;
      const rayLen = holeX * 0.82;
      const rxEnd = rayLen * Math.cos(aimRad);
      const ryEnd = rayLen * Math.sin(aimRad);
      ctx.save();
      ctx.strokeStyle = 'rgba(250, 204, 21, 0.85)';
      ctx.setLineDash([4 * dpr, 3 * dpr]);
      ctx.lineWidth = 1.8 * dpr;
      ctx.beginPath();
      ctx.moveTo(mapX(0), mapY(0));
      ctx.lineTo(mapX(rxEnd), mapY(ryEnd));
      ctx.stroke();
      ctx.fillStyle = '#fde047';
      ctx.font = `600 ${10 * dpr}px "JetBrains Mono", monospace`;
      ctx.fillText(`Starting Direction (${state.aimAngleDeg >= 0 ? '+' : ''}${state.aimAngleDeg.toFixed(1)}°)`, mapX(rxEnd * 0.52), mapY(ryEnd * 0.52) - 6 * dpr);
      ctx.restore();
    }
  }

  // Skid-to-Roll transition line
  if (!state.stimpmeterMode) {
    const xSkid = mapX(0.12 * holeX);
    ctx.save();
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.72)';
    ctx.setLineDash([6 * dpr, 5 * dpr]);
    ctx.lineWidth = 1.6 * dpr;
    ctx.beginPath(); ctx.moveTo(xSkid, 0); ctx.lineTo(xSkid, h); ctx.stroke();
    ctx.restore();
  }

  const isLargeBatch = greenBatchCB.length >= 100;
  const bgAlphaUN = isLargeBatch ? (greenBatchCB.length >= 500 ? 0.08 : 0.15) : 0.28;
  const bgAlphaCB = isLargeBatch ? (greenBatchCB.length >= 500 ? 0.12 : 0.24) : 0.48;
  const bgWidth = (isLargeBatch ? 0.75 : 1.25) * dpr;

  // Draw Unbalanced Slack Trajectories (Faded Thin Red/Orange Overlays)
  if (!state.stimpmeterMode) {
    greenBatchUN.forEach((tr) => {
      ctx.strokeStyle = `rgba(239, 68, 68, ${bgAlphaUN})`;
      ctx.lineWidth = bgWidth;
      ctx.beginPath();
      let lastPt = tr.pts[0];
      tr.pts.forEach((p, idx) => {
        if (p.t <= state.rollT) {
          if (idx === 0) ctx.moveTo(mapX(p.x), mapY(p.y));
          else ctx.lineTo(mapX(p.x), mapY(p.y));
          lastPt = p;
        }
      });
      ctx.stroke();
      if (!isLargeBatch) {
        ctx.fillStyle = 'rgba(239, 68, 68, 0.65)';
        ctx.beginPath(); ctx.arc(mapX(lastPt.x), mapY(lastPt.y), 2.8 * dpr, 0, Math.PI * 2); ctx.fill();
      }
    });
  }

  // Draw 4-Point Counterbalanced Background Trial Overlays (Faded & Thinner than Current Putt)
  for (let idx = greenBatchCB.length - 1; idx >= 1; idx--) {
    const tr = greenBatchCB[idx];
    ctx.strokeStyle = tr.holed
      ? `rgba(34, 211, 238, ${bgAlphaCB})`
      : `rgba(250, 204, 21, ${bgAlphaCB * 0.85})`;
    ctx.lineWidth = bgWidth;
    ctx.beginPath();
    let lastPt = tr.pts[0];
    for (let k = 0; k < tr.pts.length; k++) {
      const p = tr.pts[k];
      if (p.t <= state.rollT) {
        if (k === 0) ctx.moveTo(mapX(p.x), mapY(p.y));
        else ctx.lineTo(mapX(p.x), mapY(p.y));
        lastPt = p;
      }
    }
    ctx.stroke();
    ctx.fillStyle = tr.holed ? `rgba(34, 211, 238, ${Math.min(0.8, bgAlphaCB * 2.2)})` : `rgba(250, 204, 21, ${Math.min(0.75, bgAlphaCB * 2.0)})`;
    ctx.beginPath(); ctx.arc(mapX(lastPt.x), mapY(lastPt.y), (isLargeBatch ? 2.0 : 3.2) * dpr, 0, Math.PI * 2); ctx.fill();
  }

  // Draw Current Foreground Putt (idx === 0) with Bold High-Contrast Line & Crisp Ball Marker
  if (greenBatchCB.length > 0) {
    const hero = greenBatchCB[0];
    ctx.save();
    ctx.shadowColor = 'rgba(56, 189, 248, 0.75)';
    ctx.shadowBlur = 8 * dpr;
    ctx.strokeStyle = state.stimpmeterMode ? '#c084fc' : '#38bdf8';
    ctx.lineWidth = 3.0 * dpr;
    ctx.beginPath();
    let lastPt = hero.pts[0];
    hero.pts.forEach((p, idx) => {
      if (p.t <= state.rollT) {
        if (idx === 0) ctx.moveTo(mapX(p.x), mapY(p.y));
        else ctx.lineTo(mapX(p.x), mapY(p.y));
        lastPt = p;
      }
    });
    ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 2.0 * dpr;
    ctx.beginPath(); ctx.arc(mapX(lastPt.x), mapY(lastPt.y), 5.2 * dpr, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.restore();
  }

  // If Stimpmeter Ramp Mode is active, draw the 36-inch USGA V-groove Aluminum Stimpmeter Ramp at (0,0)
  if (state.stimpmeterMode) {
    const sx0 = mapX(-0.18), sy0 = mapY(0.0);
    const sx1 = mapX(0.18),  sy1 = mapY(0.0);
    ctx.save();
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 6.0 * dpr;
    ctx.beginPath(); ctx.moveTo(sx0, sy0); ctx.lineTo(sx1, sy1); ctx.stroke();
    ctx.fillStyle = '#f8fafc';
    ctx.font = `700 ${10.5 * dpr}px "JetBrains Mono", monospace`;
    ctx.fillText('USGA Stimpmeter Ramp (20.0° Release • v_0 = 1.83 m/s)', sx0, sy0 - 10 * dpr);
    ctx.restore();
  }

  // Draw USGA Hole Cup at (holeX, 0.0 m)
  const hx = mapX(holeX), hy = mapY(0.0);
  ctx.fillStyle = '#04080e';
  ctx.strokeStyle = '#f8fafc';
  ctx.lineWidth = 2.5 * dpr;
  ctx.beginPath(); ctx.arc(hx, hy, 11.5 * dpr, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.restore();
}

function drawSGCaptureChart() {
  const c = document.getElementById('sgCaptureChartCanvas');
  if (!c) return;
  const rect = c.parentElement.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    c.width = rect.width * window.devicePixelRatio;
    c.height = rect.height * window.devicePixelRatio;
  }
  const dpr = window.devicePixelRatio || 1;
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 38 * dpr, padR = 14 * dpr, padT = 18 * dpr, padB = 24 * dpr;
  const midGap = 28 * dpr;
  const subW = (w - padL - padR - midGap) * 0.5;
  const subH = h - padT - padB;

  // Left Subplot: Expected Strokes E_player(v_entry) vs Capture Speed v_entry in [0.20, 1.35] m/s
  ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1 * dpr;
  ctx.strokeRect(padL, padT, subW, subH);
  ctx.fillStyle = '#94a3b8';
  ctx.font = `600 ${9.5 * dpr}px "JetBrains Mono", monospace`;
  ctx.fillText('E_player vs Capture Speed v_entry (m/s)', padL + 4 * dpr, padT - 5 * dpr);

  const vMin = 0.20, vMax = 1.35;
  const eMin = 1.15, eMax = 2.45;
  const mapVX = (v) => padL + ((v - vMin) / (vMax - vMin)) * subW;
  const mapEY = (e) => padT + subH * (1.0 - (e - eMin) / (eMax - eMin));

  // Stabilized Curve (Cyan, minimum at v_entry* = 0.65 m/s)
  ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 2.2 * dpr;
  ctx.beginPath();
  for (let i = 0; i <= 40; i++) {
    const v = vMin + (i / 40) * (vMax - vMin);
    const eStab = 1.24 + 1.15 * Math.pow(v - 0.65, 2) + (v < 0.45 ? 0.35 * Math.pow(0.45 - v, 1.5) : 0.0);
    const x = mapVX(v), y = mapEY(eStab);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Unstabilized Poor Putter Curve (Orange/Red, minimum at v_entry* = 0.38 m/s due to 3-putt comebacker penalty!)
  ctx.strokeStyle = '#f97316'; ctx.lineWidth = 2.0 * dpr;
  ctx.setLineDash([5 * dpr, 3 * dpr]);
  ctx.beginPath();
  for (let i = 0; i <= 40; i++) {
    const v = vMin + (i / 40) * (vMax - vMin);
    const ePoor = 1.66 + 1.55 * Math.pow(Math.max(0, v - 0.38), 1.65) + 1.80 * Math.pow(Math.max(0, 0.38 - v), 2);
    const x = mapVX(v), y = mapEY(Math.min(eMax, ePoor));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Vertical Marker for Selected Capture Speed
  const curV = state.captureSpeedMs || 0.58;
  const xv = mapVX(curV);
  ctx.strokeStyle = '#facc15'; ctx.lineWidth = 1.6 * dpr;
  ctx.beginPath(); ctx.moveTo(xv, padT); ctx.lineTo(xv, padT + subH); ctx.stroke();

  // Right Subplot: 0-45 ft Player-Specific Expected Strokes E_player(d) (Cyan Stabilized vs Orange Slack)
  const rx0 = padL + subW + midGap;
  ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1 * dpr;
  ctx.strokeRect(rx0, padT, subW, subH);
  ctx.fillStyle = '#94a3b8';
  ctx.fillText('0-45 ft Bellman Curve E_player(d)', rx0 + 4 * dpr, padT - 5 * dpr);

  const mapDX = (dFt) => rx0 + (dFt / 45.0) * subW;
  const mapDEY = (e) => padT + subH * (1.0 - (e - 1.0) / 1.65);

  ctx.strokeStyle = '#34d399'; ctx.lineWidth = 2.2 * dpr;
  ctx.beginPath();
  for (let d = 0; d <= 45; d += 1.5) {
    const eS = 1.0 + 1.08 * (1.0 - Math.exp(-d / 14.5)) + 0.004 * d;
    const x = mapDX(d), y = mapDEY(eS);
    if (d === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2.0 * dpr;
  ctx.setLineDash([5 * dpr, 3 * dpr]);
  ctx.beginPath();
  for (let d = 0; d <= 45; d += 1.5) {
    const eU = 1.0 + 1.38 * (1.0 - Math.exp(-d / 9.5)) + 0.0085 * d;
    const x = mapDX(d), y = mapDEY(Math.min(2.62, eU));
    if (d === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  const xd = mapDX(state.puttDistFt || 10.0);
  ctx.strokeStyle = '#facc15'; ctx.lineWidth = 1.6 * dpr;
  ctx.beginPath(); ctx.moveTo(xd, padT); ctx.lineTo(xd, padT + subH); ctx.stroke();
}
