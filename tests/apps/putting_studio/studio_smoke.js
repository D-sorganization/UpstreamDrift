'use strict';
// Headless smoke load of the standalone studio: runs every classic script in
// a vm with DOM/WebGL stubs, then exercises one batch and one physics call.
const vm = require('vm');
const fs = require('fs');
const path = require('path');

// Usage: node studio_smoke.js <studio dir>; prints SMOKE_LOAD_MS=<ms>.
const BASE = process.argv[2];
const T0 = process.hrtime.bigint();

function makeStubElement(id, withParent = true) {
  const listeners = {};
  const el = {
    id,
    value: '0',
    checked: false,
    textContent: '',
    innerText: '',
    innerHTML: '',
    className: '',
    style: {},
    classList: {
      add() {}, remove() {}, toggle() {}, contains() { return false; },
    },
    children: [],
    // One level deep only -- avoids infinite recursion building
    // grandparents nobody in this smoke test needs.
    parentElement: withParent ? makeStubElement(`${id}-parent`, false) : null,
    dataset: {},
    disabled: false,
    width: 300,
    height: 150,
    duration: 6,
    currentTime: 0,
    playbackRate: 1.0,
    addEventListener(type, fn) {
      listeners[type] = listeners[type] || [];
      listeners[type].push(fn);
    },
    removeEventListener() {},
    dispatchEvent(evt) {
      const fns = listeners[evt && evt.type] || [];
      fns.forEach((fn) => fn(evt));
      return true;
    },
    appendChild(child) { el.children.push(child); return child; },
    removeChild() {},
    getContext(kind) {
      if (kind === '2d') {
        return {
          clearRect() {}, fillRect() {}, strokeRect() {}, beginPath() {},
          moveTo() {}, lineTo() {}, stroke() {}, fill() {}, arc() {},
          ellipse() {}, quadraticCurveTo() {}, bezierCurveTo() {}, rect() {},
          closePath() {}, save() {}, restore() {}, translate() {},
          rotate() {}, scale() {}, setTransform() {}, fillText() {},
          strokeText() {}, measureText() { return { width: 0 }; },
          createLinearGradient() {
            return { addColorStop() {} };
          },
          createRadialGradient() {
            return { addColorStop() {} };
          },
          drawImage() {}, getImageData() { return { data: [] }; },
          putImageData() {}, setLineDash() {},
          canvas: el,
          fillStyle: '', strokeStyle: '', lineWidth: 1, font: '',
          textAlign: '', textBaseline: '', globalAlpha: 1,
          shadowColor: '', shadowBlur: 0,
        };
      }
      if (kind === 'webgl' || kind === 'webgl2' || kind === 'experimental-webgl') {
        return makeWebGLStub(el);
      }
      return null;
    },
    getBoundingClientRect() {
      return { left: 0, top: 0, right: 800, bottom: 600, width: 800, height: 600 };
    },
    focus() {}, blur() {}, click() {}, select() {},
    pause() {}, play() { return Promise.resolve(); },
    setAttribute() {}, getAttribute() { return null; },
    querySelector() { return makeStubElement('stub-child'); },
    querySelectorAll() { return []; },
  };
  return el;
}

// Minimal headless WebGL stand-in. THREE.WebGLRenderer's constructor calls a
// large, version-dependent set of gl.* methods purely to probe capabilities;
// none of those probe results matter for this smoke test (we never call
// renderer.render()/animate() here), so unknown methods are no-ops and
// unknown ALL_CAPS properties are treated as opaque enum tokens.
const glConstants = new Map();
let glConstantCounter = 1;
function glConstant(name) {
  if (!glConstants.has(name)) glConstants.set(name, glConstantCounter++);
  return glConstants.get(name);
}
const glMethodCache = new Map();
function makeWebGLStub(canvasEl) {
  const known = {
    getContextAttributes() { return { alpha: true, antialias: true, stencil: true, depth: true }; },
    getExtension() { return null; },
    getSupportedExtensions() { return []; },
    getParameter(p) {
      // Common capability queries THREE reads as plain numbers/strings.
      if (p === glConstant('VERSION')) return 'WebGL 1.0 (stub)';
      if (p === glConstant('SHADING_LANGUAGE_VERSION')) return 'WebGL GLSL ES 1.0 (stub)';
      if (p === glConstant('VENDOR') || p === glConstant('RENDERER')) return 'stub';
      // Single-element array: coerces to the number 16 via Array.prototype
      // toString()+ToNumber for arithmetic/comparison call sites, while
      // still supporting .indexOf(...) for call sites that treat the
      // result as array-like (e.g. COMPRESSED_TEXTURE_FORMATS).
      return [16];
    },
    getShaderPrecisionFormat() { return { rangeMin: 127, rangeMax: 127, precision: 23 }; },
    getContext() { return null; },
    isContextLost() { return false; },
    createProgram() { return {}; },
    createShader() { return {}; },
    createBuffer() { return {}; },
    createTexture() { return {}; },
    createFramebuffer() { return {}; },
    createRenderbuffer() { return {}; },
    getProgramParameter() { return true; },
    getShaderParameter() { return true; },
    getProgramInfoLog() { return ''; },
    getShaderInfoLog() { return ''; },
    getActiveAttrib() { return { name: 'a', type: glConstant('FLOAT_VEC3'), size: 1 }; },
    getActiveUniform() { return { name: 'u', type: glConstant('FLOAT_VEC3'), size: 1 }; },
    getAttribLocation() { return 0; },
    getUniformLocation() { return {}; },
    checkFramebufferStatus() { return glConstant('FRAMEBUFFER_COMPLETE'); },
    canvas: canvasEl || null,
    drawingBufferWidth: 800,
    drawingBufferHeight: 600,
  };
  const handler = {
    get(target, prop) {
      if (prop in known) return known[prop];
      if (typeof prop !== 'string') return undefined;
      if (/^[A-Z0-9_]+$/.test(prop)) return glConstant(prop);
      if (!glMethodCache.has(prop)) {
        glMethodCache.set(prop, function glStubMethod() { return undefined; });
      }
      return glMethodCache.get(prop);
    },
    has() { return true; },
  };
  return new Proxy({}, handler);
}

const elementRegistry = new Map();
function getElementById(id) {
  if (!elementRegistry.has(id)) {
    elementRegistry.set(id, makeStubElement(id));
  }
  return elementRegistry.get(id);
}

const documentStub = {
  getElementById,
  createElement(tag) { return makeStubElement(`created-${tag}`); },
  createElementNS() { return makeStubElement('created-ns'); },
  querySelector() { return makeStubElement('stub-qs'); },
  querySelectorAll() { return []; },
  addEventListener() {},
  removeEventListener() {},
  body: makeStubElement('body'),
  documentElement: makeStubElement('documentElement'),
  activeElement: makeStubElement('activeElement'),
  execCommand() { return true; },
  visibilityState: 'visible',
};

const windowStub = {
  innerWidth: 1280,
  innerHeight: 800,
  devicePixelRatio: 1,
  addEventListener() {},
  removeEventListener() {},
  requestAnimationFrame(cb) { return 0; },
  cancelAnimationFrame() {},
  onload: null,
  PuttingSimulatorAPI: undefined,
  navigator: { userAgent: 'node-vm-smoke-test' },
  location: { href: 'file:///stub/index.html' },
};

const sandbox = {
  console,
  window: windowStub,
  document: documentStub,
  navigator: windowStub.navigator,
  requestAnimationFrame: windowStub.requestAnimationFrame,
  cancelAnimationFrame: windowStub.cancelAnimationFrame,
  performance: { now: () => Date.now() },
  Math,
  JSON,
  Date,
  Image: function Image() { return makeStubElement('image'); },
  localStorage: {
    getItem() { return null; },
    setItem() {}, removeItem() {}, clear() {},
  },
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  module: { exports: {} },
  globalThis: undefined,
};
sandbox.globalThis = sandbox;
sandbox.window.document = documentStub;

const context = vm.createContext(sandbox);

// Load in the page's own order, so the smoke test follows index.html.
const scriptOrder = [...fs.readFileSync(path.join(BASE, 'index.html'), 'utf8')
  .matchAll(/<script src="([^"]+)"><\/script>/g)].map((m) => m[1]);
if (scriptOrder.length === 0) throw new Error('index.html loads no scripts');

let failed = false;
for (const rel of scriptOrder) {
  const full = path.join(BASE, rel);
  const code = fs.readFileSync(full, 'utf8');
  try {
    vm.runInContext(code, context, { filename: rel });
    console.log(`OK   load: ${rel}`);
  } catch (err) {
    failed = true;
    console.log(`FAIL load: ${rel} -> ${err.name}: ${err.message}`);
    // V8 prefixes vm.runInContext stack traces with an inlined source-line
    // code frame; for the one-line minified vendor bundle that frame IS the
    // entire 600KB file, so keep only the actual "at ..." call-site lines.
    if (err.stack) {
      const frames = err.stack.split('\n').filter((l) => l.trim().startsWith('at '));
      console.log(frames.slice(0, 12).join('\n'));
    }
  }
}

console.log(`SMOKE_LOAD_MS=${Number(process.hrtime.bigint() - T0) / 1e6}`);

if (failed) {
  console.log('--- ABORT: load phase had errors, skipping behavior calls ---');
  process.exit(1);
}

// Exercise the dispersion-batch rebuild and one physics evaluation.
try {
  vm.runInContext('rebuildGreenDispersionBatch();', context, { filename: 'smoke:rebuildGreenDispersionBatch' });
  console.log('OK   call: rebuildGreenDispersionBatch()');
} catch (err) {
  failed = true;
  console.log(`FAIL call: rebuildGreenDispersionBatch() -> ${err.name}: ${err.message}`);
  console.log(err.stack.split('\n').filter((l) => l.trim().startsWith('at ')).slice(0, 8).join('\n'));
}

try {
  vm.runInContext('globalThis.__smokeLastBatch = { greenBatchCB_len: greenBatchCB.length, greenBatchUN_len: greenBatchUN.length, lastBatchStats: lastBatchStats };', context, { filename: 'smoke:readBatch' });
  const info = context.__smokeLastBatch;
  console.log(`OK   data : greenBatchCB.length=${info.greenBatchCB_len} greenBatchUN.length=${info.greenBatchUN_len} lastBatchStats.makePct=${info.lastBatchStats ? info.lastBatchStats.makeCB : 'n/a'}`);
} catch (err) {
  failed = true;
  console.log(`FAIL data : reading batch globals -> ${err.name}: ${err.message}`);
}

try {
  vm.runInContext('globalThis.__smokePhys = evalPhysicsAtTime(state.t !== undefined ? state.t : 0.5);', context, { filename: 'smoke:evalPhysicsAtTime' });
  const phys = context.__smokePhys;
  console.log(`OK   call: evalPhysicsAtTime() -> keys=${Object.keys(phys || {}).slice(0, 6).join(',')}`);
} catch (err) {
  failed = true;
  console.log(`FAIL call: evalPhysicsAtTime() -> ${err.name}: ${err.message}`);
  console.log(err.stack.split('\n').filter((l) => l.trim().startsWith('at ')).slice(0, 8).join('\n'));
}

try {
  vm.runInContext('globalThis.__smokeSingle = PuttingPhysics.simulateSinglePuttRK4({ vClubMps: 1.6, aimDeg: 0, pathErrDeg: 0, faceErrDeg: 0, strikeMm: 0, moiKgM2: 5000/1.0e7, stimpFt: 10.5, upPct: 0, crossPct: 0, trialSeed: 1, distFt: 10, roughnessMm: 0.35 });', context, { filename: 'smoke:directPhysics' });
  const single = context.__smokeSingle;
  console.log(`OK   call: PuttingPhysics.simulateSinglePuttRK4() -> finalX=${single.finalX}, holed=${single.holed}`);
} catch (err) {
  failed = true;
  console.log(`FAIL call: PuttingPhysics.simulateSinglePuttRK4() -> ${err.name}: ${err.message}`);
}

process.exit(failed ? 1 : 0);
