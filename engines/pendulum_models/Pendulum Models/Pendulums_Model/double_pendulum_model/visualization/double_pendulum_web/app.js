const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const pixelsPerMeter = 300;
const planeGuideLengthPx = pixelsPerMeter * 3;
const pivot = {
  x: canvas.width * 0.32,
  y: canvas.height * 0.28,
};
let animationId = null;

const gravity = 9.80665;

const state = {
  theta1: -45 * Math.PI / 180,
  theta2: -90 * Math.PI / 180,
  omega1: 0,
  omega2: 0,
  time: 0,
};

const params = {
  l1: 0.75,
  l2: 1.0,
  m1: 7.5,
  mShaft: 0.35,
  mHead: 0.2,
  com1: 0.45,
  com2: 0.43,
  plane: 35,
  damping1: 0.4,
  damping2: 0.25,
};

function rotatePoint(point, angleRad, center = pivot) {
  const dx = point.x - center.x;
  const dy = point.y - center.y;
  const cosA = Math.cos(angleRad);
  const sinA = Math.sin(angleRad);

  return {
    x: center.x + dx * cosA - dy * sinA,
    y: center.y + dx * sinA + dy * cosA,
  };
}

function resetStateFromInputs() {
  state.theta1 = Number(document.getElementById('theta1').value) * Math.PI / 180;
  state.theta2 = Number(document.getElementById('theta2').value) * Math.PI / 180;
  state.omega1 = 0;
  state.omega2 = 0;
  state.time = 0;
}

function updateParamsFromInputs() {
  params.l1 = Number(document.getElementById('l1').value);
  params.l2 = Number(document.getElementById('l2').value);
  params.m1 = Number(document.getElementById('m1').value);
  params.mShaft = Number(document.getElementById('mshaft').value);
  params.mHead = Number(document.getElementById('mhead').value);
  params.com1 = Number(document.getElementById('com1').value);
  params.com2 = Number(document.getElementById('com2').value);
  params.plane = Number(document.getElementById('plane').value);
  params.tau1Expr = document.getElementById('tau1').value || '0';
  params.tau2Expr = document.getElementById('tau2').value || '0';
}

function safeEval(expr, context) {
  try {
    const fn = new Function(...Object.keys(context), `return ${expr};`);
    return Number(fn(...Object.values(context)));
  } catch (err) {
    return 0;
  }
}

function massMatrix(theta2) {
  const m2 = params.mShaft + params.mHead;
  const lc1 = params.l1 * params.com1;
  const lc2 = params.l2 * params.com2;
  const I1 = (1 / 12) * params.m1 * params.l1 * params.l1 + params.m1 * lc1 * lc1;
  const I2 = (1 / 12) * m2 * params.l2 * params.l2 + m2 * lc2 * lc2;
  const cos2 = Math.cos(theta2);
  const m11 = I1 + I2 + params.m1 * lc1 * lc1 + m2 * (params.l1 ** 2 + lc2 ** 2 + 2 * params.l1 * lc2 * cos2);
  const m12 = I2 + m2 * (lc2 ** 2 + params.l1 * lc2 * cos2);
  const m22 = I2 + m2 * lc2 ** 2;
  return [[m11, m12], [m12, m22]];
}

function coriolis(theta2, omega1, omega2) {
  const m2 = params.mShaft + params.mHead;
  const lc2 = params.l2 * params.com2;
  const h = -m2 * params.l1 * lc2 * Math.sin(theta2);
  return [h * (2 * omega1 * omega2 + omega2 ** 2), h * omega1 ** 2];
}

function gravityVector(theta1, theta2) {
  const gProj = gravity * Math.cos(params.plane * Math.PI / 180);
  const m2 = params.mShaft + params.mHead;
  const lc1 = params.l1 * params.com1;
  const lc2 = params.l2 * params.com2;
  const g1 = (params.m1 * lc1 + m2 * params.l1) * gProj * Math.sin(theta1) + m2 * lc2 * gProj * Math.sin(theta1 + theta2);
  const g2 = m2 * lc2 * gProj * Math.sin(theta1 + theta2);
  return [g1, g2];
}

function damping(omega1, omega2) {
  return [params.damping1 * omega1, params.damping2 * omega2];
}

function torques(t, s) {
  const ctx = { t, theta1: s.theta1, theta2: s.theta2, omega1: s.omega1, omega2: s.omega2, Math };
  return [safeEval(params.tau1Expr, ctx), safeEval(params.tau2Expr, ctx)];
}

function invert2x2(m) {
  const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  return [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]];
}

function derivatives(t, s) {
  const tau = torques(t, s);
  const c = coriolis(s.theta2, s.omega1, s.omega2);
  const g = gravityVector(s.theta1, s.theta2);
  const d = damping(s.omega1, s.omega2);
  const inv = invert2x2(massMatrix(s.theta2));
  const acc1 = inv[0][0] * (tau[0] - c[0] - g[0] - d[0]) + inv[0][1] * (tau[1] - c[1] - g[1] - d[1]);
  const acc2 = inv[1][0] * (tau[0] - c[0] - g[0] - d[0]) + inv[1][1] * (tau[1] - c[1] - g[1] - d[1]);
  return [s.omega1, s.omega2, acc1, acc2];
}

function rk4(dt) {
  const k1 = derivatives(state.time, state);
  const s2 = {
    theta1: state.theta1 + dt / 2 * k1[0],
    theta2: state.theta2 + dt / 2 * k1[1],
    omega1: state.omega1 + dt / 2 * k1[2],
    omega2: state.omega2 + dt / 2 * k1[3],
  };
  const k2 = derivatives(state.time + dt / 2, s2);
  const s3 = {
    theta1: state.theta1 + dt / 2 * k2[0],
    theta2: state.theta2 + dt / 2 * k2[1],
    omega1: state.omega1 + dt / 2 * k2[2],
    omega2: state.omega2 + dt / 2 * k2[3],
  };
  const k3 = derivatives(state.time + dt / 2, s3);
  const s4 = {
    theta1: state.theta1 + dt * k3[0],
    theta2: state.theta2 + dt * k3[1],
    omega1: state.omega1 + dt * k3[2],
    omega2: state.omega2 + dt * k3[3],
  };
  const k4 = derivatives(state.time + dt, s4);

  state.theta1 += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
  state.theta2 += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
  state.omega1 += dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
  state.omega2 += dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]);
  state.time += dt;
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const planeRad = params.plane * Math.PI / 180;
  ctx.strokeStyle = '#2a3545';
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  const planeDirection = { x: Math.sin(planeRad), y: Math.cos(planeRad) };
  ctx.moveTo(
    pivot.x - planeDirection.x * planeGuideLengthPx,
    pivot.y - planeDirection.y * planeGuideLengthPx,
  );
  ctx.lineTo(
    pivot.x + planeDirection.x * planeGuideLengthPx,
    pivot.y + planeDirection.y * planeGuideLengthPx,
  );
  ctx.stroke();
  ctx.setLineDash([]);

  const elbowUnrotated = {
    x: pivot.x + Math.sin(state.theta1) * params.l1 * pixelsPerMeter,
    y: pivot.y + Math.cos(state.theta1) * params.l1 * pixelsPerMeter,
  };
  const wristUnrotated = {
    x: pivot.x + (
      Math.sin(state.theta1) * params.l1 +
      Math.sin(state.theta1 + state.theta2) * params.l2
    ) * pixelsPerMeter,
    y: pivot.y + (
      Math.cos(state.theta1) * params.l1 +
      Math.cos(state.theta1 + state.theta2) * params.l2
    ) * pixelsPerMeter,
  };

  const elbow = rotatePoint(elbowUnrotated, -planeRad);
  const wrist = rotatePoint(wristUnrotated, -planeRad);

  ctx.strokeStyle = '#66fcf1';
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.moveTo(pivot.x, pivot.y);
  ctx.lineTo(elbow.x, elbow.y);
  ctx.stroke();

  ctx.strokeStyle = '#ff6b6b';
  ctx.beginPath();
  ctx.moveTo(elbow.x, elbow.y);
  ctx.lineTo(wrist.x, wrist.y);
  ctx.stroke();

  ctx.fillStyle = '#ff6b6b';
  ctx.beginPath();
  ctx.arc(wrist.x, wrist.y, 10, 0, Math.PI * 2);
  ctx.fill();

  const contactRadius = (params.l1 + params.l2) * pixelsPerMeter * 0.95;
  const contactPoint = {
    x: pivot.x + planeDirection.x * contactRadius,
    y: pivot.y + planeDirection.y * contactRadius,
  };
  ctx.fillStyle = '#fcd34d';
  ctx.beginPath();
  ctx.arc(contactPoint.x, contactPoint.y, 8, 0, Math.PI * 2);
  ctx.fill();
}

function updateButtons() {
  const isRunning = animationId !== null;
  const isPaused = !isRunning && state.time > 0;

  const startBtn = document.getElementById('start');
  const pauseBtn = document.getElementById('pause');
  const startSpan = startBtn.querySelector('span');

  startBtn.disabled = isRunning;
  pauseBtn.disabled = !isRunning;

  if (startSpan) {
    startSpan.textContent = isPaused ? "Resume" : "Start";
  }
}

function announce(message) {
  const region = document.getElementById('status-announcer');
  if (region) region.textContent = message;
}

function step() {
  rk4(0.01);
  draw();
  const tau = torques(state.time, state);
  document.getElementById('torques').textContent = `Applied Nm: shoulder=${tau[0].toFixed(2)}, wrist=${tau[1].toFixed(2)}`;
  animationId = requestAnimationFrame(step);
}

function start() {
  if (animationId) return;
  if (state.time === 0) {
    resetStateFromInputs();
  }
  updateParamsFromInputs();
  step();
  updateButtons();
  announce('Simulation started');
}

function pause() {
  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  updateButtons();
  announce('Simulation paused');
}

function reset() {
  pause();
  resetStateFromInputs();
  updateParamsFromInputs();
  draw();
  document.getElementById('torques').textContent = 'Torques: --';
  updateButtons();
  announce('Simulation reset');
}

['start', 'pause', 'reset'].forEach(id => document.getElementById(id).addEventListener('click', () => {
  ({ start, pause, reset })[id]();
}));

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;

  if (e.key === ' ' || e.key === 'Spacebar') {
    e.preventDefault();
    animationId ? pause() : start();
  } else if (e.key === 'r' || e.key === 'R') {
    reset();
  }
});

resetStateFromInputs();
updateParamsFromInputs();
reset();
