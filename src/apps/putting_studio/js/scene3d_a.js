/**
 * Three.js scene bootstrap: renderer, HD procedural textures, lighting,
 * camera-sphere state and mouse/trackpad/touch orbit-pan-zoom controls for
 * the standalone HTML studio (epic #14141, step 8h). Classic script (no ES
 * modules; opens from file://). Depends on globals from js/state.js
 * (state, L_CLUB, evalPhysicsAtTime) loaded immediately before it, and
 * exposes scene, camera, renderer, container, camSpherical,
 * updateCameraPosition, clubToThree, getShoulderHubThree, zoomToFitScene
 * and the HD textures as globals for js/scene3d_b.js and later files.
 */
// 2. THREE.JS 3D SCENE SETUP (FULL CLUB DEFAULT VIEW + OBLONG GREEN + TRACES)
// =========================================================================
const container = document.getElementById('threeContainer');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x93c5fd, 0.0085);

const camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.01, 260);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 2, 2.5));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
if (THREE.ACESFilmicToneMapping !== undefined) {
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.12;
}
if (THREE.sRGBEncoding !== undefined) {
  renderer.outputEncoding = THREE.sRGBEncoding;
}
container.appendChild(renderer.domElement);

// High-Definition Procedural Texture Generators (Anisotropic Filtered)
const maxAniso = renderer.capabilities.getMaxAnisotropy ? renderer.capabilities.getMaxAnisotropy() : 8;

function createHDBentgrassTexture() {
  const c = document.createElement('canvas');
  c.width = 1024; c.height = 1024;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#15803d';
  ctx.fillRect(0, 0, 1024, 1024);
  // Tournament double-cut mowing bands + fine bentgrass micro-blades
  for (let s = 0; s < 16; s++) {
    ctx.fillStyle = (s % 2 === 0) ? 'rgba(34, 197, 94, 0.22)' : 'rgba(20, 83, 45, 0.18)';
    ctx.fillRect(s * 64, 0, 64, 1024);
  }
  for (let i = 0; i < 28000; i++) {
    const x = (Math.sin(i * 12.9898) * 43758.5453 % 1 + 1) % 1 * 1024;
    const y = (Math.sin(i * 78.2330) * 43758.5453 % 1 + 1) % 1 * 1024;
    ctx.fillStyle = (i % 3 === 0) ? 'rgba(74, 222, 128, 0.24)' : (i % 3 === 1 ? 'rgba(21, 128, 61, 0.28)' : 'rgba(6, 78, 59, 0.22)');
    ctx.fillRect(x, y, 2, 3);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(6, 4);
  tex.anisotropy = maxAniso;
  return tex;
}

function createGolfBallDimpleTexture() {
  const c = document.createElement('canvas');
  c.width = 512; c.height = 256;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(0, 0, 512, 256);
  for (let row = 1; row < 14; row++) {
    const y = (row / 14) * 256;
    const cols = 28;
    for (let col = 0; col < cols; col++) {
      const x = ((col + (row % 2) * 0.5) / cols) * 512;
      const grad = ctx.createRadialGradient(x, y, 0.5, x, y, 5.2);
      grad.addColorStop(0, '#94a3b8');
      grad.addColorStop(0.7, '#cbd5e1');
      grad.addColorStop(1, '#f8fafc');
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.arc(x, y, 5.2, 0, Math.PI * 2); ctx.fill();
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.anisotropy = maxAniso;
  return tex;
}

function createHDSkyDomeTexture() {
  const c = document.createElement('canvas');
  c.width = 1024; c.height = 512;
  const ctx = c.getContext('2d');
  const skyGrad = ctx.createLinearGradient(0, 0, 0, 275);
  skyGrad.addColorStop(0.0, '#082f49');
  skyGrad.addColorStop(0.38, '#0284c7');
  skyGrad.addColorStop(0.76, '#7dd3fc');
  skyGrad.addColorStop(0.95, '#e0f2fe');
  skyGrad.addColorStop(1.0, '#dcfce7');
  ctx.fillStyle = skyGrad;
  ctx.fillRect(0, 0, 1024, 275);
  // Soft volumetric cirrus/cumulus cloud wisps
  for (let cl = 0; cl < 45; cl++) {
    const cx = ((cl * 73) % 1024);
    const cy = 95 + ((cl * 29) % 110);
    const rx = 45 + (cl % 5) * 22;
    const ry = 8 + (cl % 3) * 5;
    const cg = ctx.createRadialGradient(cx, cy, 2, cx, cy, rx);
    cg.addColorStop(0, 'rgba(255, 255, 255, 0.42)');
    cg.addColorStop(1, 'rgba(255, 255, 255, 0.0)');
    ctx.fillStyle = cg;
    ctx.beginPath(); ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2); ctx.fill();
  }
  // Distant Blue-Green Mountain Range Silhouette at Horizon (y ~ 260)
  ctx.fillStyle = '#1e3a5f';
  ctx.beginPath(); ctx.moveTo(0, 275);
  for (let x = 0; x <= 1024; x += 16) {
    const mh = 18 + 14 * Math.sin(x * 0.015) + 8 * Math.cos(x * 0.042);
    ctx.lineTo(x, 272 - mh);
  }
  ctx.lineTo(1024, 275); ctx.closePath(); ctx.fill();
  // Lower hemisphere turf tint
  ctx.fillStyle = '#064e3b';
  ctx.fillRect(0, 275, 1024, 237);
  return new THREE.CanvasTexture(c);
}

const hdBentgrassTex = createHDBentgrassTexture();
const hdDimpleTex = createGolfBallDimpleTexture();
const hdSkyTex = createHDSkyDomeTexture();

const hemiLight = new THREE.HemisphereLight(0xe0f2fe, 0x15803d, 0.95);
scene.add(hemiLight);
const dirLight = new THREE.DirectionalLight(0xfffbeb, 1.15);
dirLight.position.set(2.5, 4.5, 3.0);
scene.add(dirLight);
const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.45);
fillLight.position.set(-2.0, 1.5, -1.5);
scene.add(fillLight);

// Default Camera shows the FULL CLUB, HAND ARC, AND OBLONG GREEN comfortably!
// Zoom range expanded from 0.22 m up to 8.50 m!
const camSpherical = { radius: 2.35, theta: -0.42, phi: 1.18, target: new THREE.Vector3(0.42, -0.40, 0.14) };
let isDragging = false, isPanning = false, prevX = 0, prevY = 0;

function syncZoomSlider() {
  const zs = document.getElementById('zoomRangeSlider');
  if (zs) zs.value = camSpherical.radius.toFixed(2);
}

function updateCameraPosition() {
  const sinPhi = Math.sin(camSpherical.phi);
  camera.position.x = camSpherical.target.x + camSpherical.radius * sinPhi * Math.sin(camSpherical.theta);
  camera.position.y = camSpherical.target.y + camSpherical.radius * Math.cos(camSpherical.phi);
  camera.position.z = camSpherical.target.z + camSpherical.radius * sinPhi * Math.cos(camSpherical.theta);
  camera.lookAt(camSpherical.target);
  syncZoomSlider();
}
updateCameraPosition();

function getButtArcPositionThree(thRad, psiDeg) {
  const beta = (90.0 - psiDeg) * Math.PI / 180.0;
  const rArm = state.buttArcRadius || 0.0;
  if (rArm <= 0.001) return new THREE.Vector3(0, 0, 0);
  const thHand = 0.55 * thRad;
  const X_hat = new THREE.Vector3(1, 0, 0);
  const e_rho0 = new THREE.Vector3(0, -Math.cos(beta), Math.sin(beta));
  const dx = rArm * Math.sin(thHand);
  const dUp = rArm * (1.0 - Math.cos(thHand));
  return new THREE.Vector3().addScaledVector(X_hat, dx).addScaledVector(e_rho0, -dUp);
}

function getShoulderHubThree(psiDeg) {
  const beta = (90.0 - psiDeg) * Math.PI / 180.0;
  const rArm = Math.max(0.28, state.buttArcRadius || 0.36);
  const e_rho0 = new THREE.Vector3(0, -Math.cos(beta), Math.sin(beta));
  return e_rho0.clone().multiplyScalar(-rArm);
}

function clubToThree(sDist, thRad, phiDeg, psiDeg, vecLocal) {
  const beta = (90.0 - psiDeg) * Math.PI / 180.0;
  const ph = phiDeg * Math.PI / 180.0;

  const X_hat = new THREE.Vector3(1, 0, 0);
  const e_rho0 = new THREE.Vector3(0, -Math.cos(beta), Math.sin(beta));
  const n_p = new THREE.Vector3(0, Math.sin(beta), Math.cos(beta));

  const e_s_in = new THREE.Vector3().addScaledVector(X_hat, Math.sin(thRad)).addScaledVector(e_rho0, Math.cos(thRad));
  const e_th = new THREE.Vector3().addScaledVector(X_hat, Math.cos(thRad)).addScaledVector(e_rho0, -Math.sin(thRad));

  const e_s = new THREE.Vector3().addScaledVector(e_s_in, Math.cos(ph)).addScaledVector(n_p, Math.sin(ph));
  const e_perp = new THREE.Vector3().addScaledVector(e_s_in, -Math.sin(ph)).addScaledVector(n_p, Math.cos(ph));

  const buttPos = getButtArcPositionThree(thRad, psiDeg);
  const pos = buttPos.clone().addScaledVector(e_s, sDist);
  if (!vecLocal) return { pos, buttPos, e_s, e_th, e_perp };

  const v3 = new THREE.Vector3()
    .addScaledVector(e_s, vecLocal[0])
    .addScaledVector(e_th, vecLocal[1])
    .addScaledVector(e_perp, vecLocal[2]);
  return { pos, buttPos, v3, e_s, e_th, e_perp };
}

function zoomToFitScene(mode = 'full') {
  if (mode === 'grip') {
    const midGrip = clubToThree(0.16, evalPhysicsAtTime(state.t, state.tauPreload, state.psiLieDeg).thRad, 0, state.psiLieDeg).pos;
    camSpherical.target.copy(midGrip);
    camSpherical.radius = 0.58;
  } else {
    // Frame full club + shoulder arc + oblong putting green & cup
    const shPos = getShoulderHubThree(state.psiLieDeg);
    const hPos  = clubToThree(L_CLUB, 0, 0, state.psiLieDeg).pos;
    camSpherical.target.set(0.42, (shPos.y + hPos.y) * 0.52, hPos.z * 0.65);
    const aspect = Math.max(0.60, container.clientWidth / Math.max(1, container.clientHeight));
    camSpherical.radius = Math.max(1.85, Math.min(22.00, 2.35 / Math.min(1.0, aspect)));
  }
  updateCameraPosition();
}

// Mouse Orbit & Shift/Right-Click Pan
renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());
renderer.domElement.addEventListener('mousedown', (e) => {
  isDragging = true;
  isPanning = (e.button === 2 || e.shiftKey);
  prevX = e.clientX; prevY = e.clientY;
});
window.addEventListener('mouseup', () => { isDragging = false; isPanning = false; });
window.addEventListener('mousemove', (e) => {
  if (!isDragging) return;
  const dx = e.clientX - prevX, dy = e.clientY - prevY;
  prevX = e.clientX; prevY = e.clientY;
  if (isPanning) {
    const panScale = camSpherical.radius * 0.0015;
    camSpherical.target.x -= dx * panScale;
    camSpherical.target.y += dy * panScale;
  } else {
    camSpherical.theta -= dx * 0.008;
    camSpherical.phi = Math.max(0.15, Math.min(Math.PI - 0.15, camSpherical.phi - dy * 0.008));
  }
  updateCameraPosition();
});

// Trackpad Pinch-to-Zoom (e.ctrlKey) + Smooth Wheel Zoom up to 8.50 m
renderer.domElement.addEventListener('wheel', (e) => {
  e.preventDefault();
  const factor = e.ctrlKey ? 0.0065 : 0.0015;
  const delta = e.deltaY * factor * camSpherical.radius;
  camSpherical.radius = Math.max(0.22, Math.min(22.00, camSpherical.radius + delta));
  updateCameraPosition();
}, { passive: false });

let gestureStartRadius = 2.35;
renderer.domElement.addEventListener('gesturestart', (e) => {
  e.preventDefault();
  gestureStartRadius = camSpherical.radius;
}, { passive: false });
renderer.domElement.addEventListener('gesturechange', (e) => {
  e.preventDefault();
  if (e.scale && e.scale > 0) {
    camSpherical.radius = Math.max(0.22, Math.min(22.00, gestureStartRadius / e.scale));
    updateCameraPosition();
  }
}, { passive: false });

let lastTouchDist = 0, lastTouchX = 0, lastTouchY = 0;
renderer.domElement.addEventListener('touchstart', (e) => {
  if (e.touches.length === 1) {
    lastTouchX = e.touches[0].clientX;
    lastTouchY = e.touches[0].clientY;
  } else if (e.touches.length === 2) {
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    lastTouchDist = Math.hypot(dx, dy);
  }
}, { passive: false });

renderer.domElement.addEventListener('touchmove', (e) => {
  e.preventDefault();
  if (e.touches.length === 1) {
    const dx = e.touches[0].clientX - lastTouchX;
    const dy = e.touches[0].clientY - lastTouchY;
    lastTouchX = e.touches[0].clientX;
    lastTouchY = e.touches[0].clientY;
    camSpherical.theta -= dx * 0.008;
    camSpherical.phi = Math.max(0.15, Math.min(Math.PI - 0.15, camSpherical.phi - dy * 0.008));
    updateCameraPosition();
  } else if (e.touches.length === 2) {
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    const dist = Math.hypot(dx, dy);
    if (lastTouchDist > 5) {
      const ratio = lastTouchDist / Math.max(5, dist);
      camSpherical.radius = Math.max(0.22, Math.min(22.00, camSpherical.radius * ratio));
      updateCameraPosition();
    }
    lastTouchDist = dist;
  }
}, { passive: false });
