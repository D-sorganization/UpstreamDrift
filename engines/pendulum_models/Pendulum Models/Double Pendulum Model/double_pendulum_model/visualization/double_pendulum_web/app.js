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

function parseInputs() {
  state.theta1 = Number(document.getElementById('theta1').value) * Math.PI / 180;
  state.theta2 = Number(document.getElementById('theta2').value) * Math.PI / 180;
  state.omega1 = 0; state.omega2 = 0; state.time = 0;
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

  // Pre-calculate derived physical constants to avoid recomputation in loop
  params.derived = {};
  params.derived.m2 = params.mShaft + params.mHead;
  params.derived.lc1 = params.l1 * params.com1;
  params.derived.lc2 = params.l2 * params.com2;
  params.derived.I1 = (1 / 12) * params.m1 * params.l1 * params.l1 + params.m1 * params.derived.lc1 * params.derived.lc1;
  params.derived.I2 = (1 / 12) * params.derived.m2 * params.l2 * params.l2 + params.derived.m2 * params.derived.lc2 * params.derived.lc2;
  const planeRad = params.plane * Math.PI / 180;
  params.derived.gProj = gravity * Math.cos(planeRad);

  // Pre-compile torque functions
  const contextKeys = ['t', 'theta1', 'theta2', 'omega1', 'omega2', 'Math'];
  try {
    params.tau1Fn = new Function(...contextKeys, `return ${params.tau1Expr};`);
  } catch (e) {
    params.tau1Fn = () => 0;
  }
  try {
    params.tau2Fn = new Function(...contextKeys, `return ${params.tau2Expr};`);
  } catch (e) {
    params.tau2Fn = () => 0;
  }
}

function validateExpression(expr) {
  try {
    // Check for syntax errors using the same variables as safeEval
    const vars = ['t', 'theta1', 'theta2', 'omega1', 'omega2', 'Math'];
    new Function(...vars, `return ${expr};`);
    return true;
  } catch (err) {
    return false;
  }
}

function safeEval(expr, context) {
  // Keeping safeEval for compatibility if needed, but not used in loop anymore
  try {
    const fn = new Function(...Object.keys(context), `return ${expr};`);
    return Number(fn(...Object.values(context)));
  } catch (err) {
    return 0;
  }
}

function massMatrix(theta2) {
  const { m2, lc1, lc2, I1, I2 } = params.derived;
  const cos2 = Math.cos(theta2);
  const m11 = I1 + I2 + params.m1 * lc1 * lc1 + m2 * (params.l1 ** 2 + lc2 ** 2 + 2 * params.l1 * lc2 * cos2);
  const m12 = I2 + m2 * (lc2 ** 2 + params.l1 * lc2 * cos2);
  const m22 = I2 + m2 * lc2 ** 2;
  return [[m11, m12], [m12, m22]];
}

function coriolis(theta2, omega1, omega2) {
  const { m2, lc2 } = params.derived;
  const h = -m2 * params.l1 * lc2 * Math.sin(theta2);
  return [h * (2 * omega1 * omega2 + omega2 ** 2), h * omega1 ** 2];
}

function gravityVector(theta1, theta2) {
  const { m2, lc1, lc2, gProj } = params.derived;
  const g1 = (params.m1 * lc1 + m2 * params.l1) * gProj * Math.sin(theta1) + m2 * lc2 * gProj * Math.sin(theta1 + theta2);
  const g2 = m2 * lc2 * gProj * Math.sin(theta1 + theta2);
  return [g1, g2];
}

function damping(omega1, omega2) {
  return [params.damping1 * omega1, params.damping2 * omega2];
}

function torques(t, s) {
  // Use pre-compiled functions for performance
  try {
    return [
      Number(params.tau1Fn(t, s.theta1, s.theta2, s.omega1, s.omega2, Math)),
      Number(params.tau2Fn(t, s.theta1, s.theta2, s.omega1, s.omega2, Math))
    ];
  } catch (e) {
    return [0, 0];
  }
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

  // Optimization: Pre-calculate net torques
  const net1 = tau[0] - c[0] - g[0] - d[0];
  const net2 = tau[1] - c[1] - g[1] - d[1];

  const acc1 = inv[0][0] * net1 + inv[0][1] * net2;
  const acc2 = inv[1][0] * net1 + inv[1][1] * net2;
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

function step() {
  rk4(0.01);
  draw();
  const tau = torques(state.time, state);
  document.getElementById('torques').textContent = `Applied Nm: shoulder=${tau[0].toFixed(2)}, wrist=${tau[1].toFixed(2)}`;
  animationId = requestAnimationFrame(step);
}

function updateA11yStatus(message) {
  const status = document.getElementById('a11y-status');
  if (status) status.textContent = message;
}

function start() {
  cancelAnimationFrame(animationId);
  parseInputs();
  step();
  updateA11yStatus('Simulation started');
}

function pause() {
  cancelAnimationFrame(animationId);
  updateA11yStatus('Simulation paused');
}

function reset() {
  pause();
  parseInputs();
  draw();
  document.getElementById('torques').textContent = 'Torques: --';
  updateA11yStatus('Simulation reset to initial state');
}

['start', 'pause', 'reset'].forEach(id => document.getElementById(id).addEventListener('click', () => {
  ({ start, pause, reset })[id]();
}));

['tau1', 'tau2'].forEach(id => {
  const input = document.getElementById(id);
  const help = document.getElementById(id + '-help');
  input.addEventListener('input', () => {
    const isValid = validateExpression(input.value);
    input.classList.toggle('input-error', !isValid);
    input.setAttribute('aria-invalid', !isValid);

    if (isValid) {
      help.textContent = "Vars: t, theta1, theta2, omega1, omega2";
      help.className = "input-help";
    } else {
      help.textContent = "Invalid expression";
      help.className = "input-help error";
    }
  });
});

reset();
