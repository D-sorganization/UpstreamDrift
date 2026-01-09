const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const pixelsPerMeter = 300;
const planeGuideLengthPx = pixelsPerMeter * 3;
const pivot = {
  x: canvas.width * 0.32,
  y: canvas.height * 0.28,
};
let animationId = null;
const defaultValues = {};

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
  state.omega1 = 0; state.omega2 = 0; state.time = 0;
}

function validateExpr(expr) {
  try {
    // Check syntax with dummy context keys matching torques()
    new Function('t', 'theta1', 'theta2', 'omega1', 'omega2', 'Math', `return ${expr};`);
    return null;
  } catch (e) {
    return e.message;
  }
}

function updateParamsFromInputs() {
  // Validate numeric inputs
  document.querySelectorAll('input[type="number"]').forEach(input => {
    if (!input.checkValidity()) {
      input.classList.add('error');
      input.setAttribute('title', input.validationMessage);
      input.setAttribute('aria-invalid', 'true');
    } else {
      input.classList.remove('error');
      input.removeAttribute('title');
      input.setAttribute('aria-invalid', 'false');
    }
  });

  params.l1 = Number(document.getElementById('l1').value);
  params.l2 = Number(document.getElementById('l2').value);
  params.m1 = Number(document.getElementById('m1').value);
  params.mShaft = Number(document.getElementById('mshaft').value);
  params.mHead = Number(document.getElementById('mhead').value);
  params.com1 = Number(document.getElementById('com1').value);
  params.com2 = Number(document.getElementById('com2').value);
  params.plane = Number(document.getElementById('plane').value);

  const tau1Input = document.getElementById('tau1');
  const tau2Input = document.getElementById('tau2');

  [tau1Input, tau2Input].forEach(input => {
    const errorMsg = validateExpr(input.value || '0');
    const errorEl = document.getElementById(`${input.id}-error`);

    if (!errorMsg) {
      input.classList.remove('error');
      input.setAttribute('aria-invalid', 'false');
      if (errorEl) {
        errorEl.classList.add('hidden');
        errorEl.textContent = '';
      }
      input.setAttribute('aria-describedby', 'math-hint');
    } else {
      input.classList.add('error');
      input.setAttribute('aria-invalid', 'true');
      if (errorEl) {
        errorEl.textContent = errorMsg;
        errorEl.classList.remove('hidden');
        input.setAttribute('aria-describedby', `math-hint ${input.id}-error`);
      }
    }
  });

  params.tau1Expr = tau1Input.value || '0';
  params.tau2Expr = tau2Input.value || '0';
}

function restoreDefaults() {
  pause();
  document.querySelectorAll('.grid input').forEach(input => {
    if (defaultValues[input.id] !== undefined) {
      input.value = defaultValues[input.id];
    }
  });
  resetStateFromInputs();
  updateParamsFromInputs();
  draw();
  announce('Parameters restored to defaults');
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

function step() {
  rk4(0.01);
  draw();
  const tau = torques(state.time, state);
  document.getElementById('torques').textContent = `Applied Nm: shoulder=${tau[0].toFixed(2)}, wrist=${tau[1].toFixed(2)}`;
  animationId = requestAnimationFrame(step);
}

function updateButtonStates(isRunning) {
  const btn = document.getElementById('play-pause');
  const label = btn.querySelector('span');
  const iconPlay = document.getElementById('icon-play');
  const iconPause = document.getElementById('icon-pause');

  if (isRunning) {
    iconPlay.classList.add('hidden');
    iconPause.classList.remove('hidden');
    label.textContent = 'Pause';
    btn.title = 'Pause simulation (Space)';
  } else {
    iconPlay.classList.remove('hidden');
    iconPause.classList.add('hidden');
    if (state.time !== 0) {
      label.textContent = 'Resume';
      btn.title = 'Resume simulation (Space)';
    } else {
      label.textContent = 'Start';
      btn.title = 'Start simulation (Space)';
    }
  }
}

function showToast(message) {
  const toast = document.getElementById('toast');
  if (toast) {
    toast.textContent = message;
    toast.classList.add('visible');
    setTimeout(() => toast.classList.remove('visible'), 3000);
  }
}

function announce(message) {
  const region = document.getElementById('status-announcer');
  if (region) region.textContent = message;
  showToast(message);
}

function copyShareLink() {
  const inputs = document.querySelectorAll('.grid input');
  const params = new URLSearchParams();
  inputs.forEach(input => params.set(input.id, input.value));
  // Handle file:// protocol where host is empty
  const origin = window.location.protocol === 'file:' ? window.location.pathname : `${window.location.protocol}//${window.location.host}${window.location.pathname}`;
  const newUrl = `${origin}?${params.toString()}`;

  try {
    window.history.replaceState(null, '', newUrl);
  } catch (e) {
    // Ignore history errors on file://
  }

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(newUrl).then(() => {
      announce('Link copied to clipboard');
    }).catch(() => {
      fallbackCopy(newUrl);
    });
  } else {
    fallbackCopy(newUrl);
  }
}

function fallbackCopy(text) {
  try {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed'; // Avoid scrolling to bottom
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    const successful = document.execCommand('copy');
    document.body.removeChild(textArea);
    if (successful) {
      announce('Link copied to clipboard');
      return;
    }
  } catch (err) {
    // Fallback failed
  }
  prompt('Copy this link:', text);
}

function initFromUrl() {
  const params = new URLSearchParams(window.location.search);
  let changed = false;
  params.forEach((value, key) => {
    const input = document.getElementById(key);
    if (input) {
      input.value = value;
      changed = true;
    }
  });
  if (changed) {
    resetStateFromInputs();
    updateParamsFromInputs();
    draw();
    announce('Configuration loaded from URL');
  }
}

function start() {
  cancelAnimationFrame(animationId);
  if (state.time === 0) {
    resetStateFromInputs();
  }
  updateParamsFromInputs();
  step();
  updateButtonStates(true);
  announce('Simulation started');
}

function pause() {
  cancelAnimationFrame(animationId);
  animationId = null;
  updateButtonStates(false);
  announce('Simulation paused');
}

function reset() {
  pause();
  resetStateFromInputs();
  updateParamsFromInputs();
  draw();
  document.getElementById('torques').textContent = 'Torques: --';
  updateButtonStates(false);
  announce('Simulation reset');
}

document.getElementById('play-pause').addEventListener('click', () => {
  if (animationId) pause();
  else start();
});

document.getElementById('share').addEventListener('click', copyShareLink);

['reset', 'defaults'].forEach(id => document.getElementById(id).addEventListener('click', () => {
  ({ reset, defaults: restoreDefaults })[id]();
}));

document.querySelectorAll('.grid input').forEach(input => {
  defaultValues[input.id] = input.value;
});

document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, button, select, textarea')) return;

  if (e.key === ' ' || e.key === 'Spacebar') {
    e.preventDefault();
    animationId ? pause() : start();
  } else if (e.key === 'r' || e.key === 'R') {
    reset();
  }
});

document.querySelectorAll('.grid input').forEach(input => {
  if (input.id !== 'theta1' && input.id !== 'theta2') {
    input.addEventListener('input', () => {
      updateParamsFromInputs();
      if (!animationId) draw();
    });
  }
});

reset();
initFromUrl();
