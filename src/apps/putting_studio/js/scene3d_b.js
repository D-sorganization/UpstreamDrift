/**
 * Oblong putting-green course environment, club/grip/ball meshes, the
 * force & torque curved-arrow factories and the 2D-label projector for
 * the standalone HTML studio (epic #14141, step 8i). Classic script (no
 * ES modules; opens from file://). Depends on globals from js/state.js
 * and js/scene3d_a.js (scene, camera, container, clubToThree, textures,
 * L_CLUB, THETA_MAX_DEG, POST_IMPACT_DECEL_RATIO) and exposes
 * rebuildGolfCourseEnvironment, rebuildPlaneMesh, golfBallMesh,
 * impactFlashRing, ballTraces3DGroup, alignCylinder, setThickForceArrow,
 * project3DLabel and the club/arrow meshes as globals for js/green3d.js
 * and later files.
 */
// =========================================================================
// 2B. REALISTIC OBLONG / KIDNEY GOLF GREEN, MOWED STRIPES, BUNKERS & PIN
// =========================================================================
const courseGroup = new THREE.Group();
scene.add(courseGroup);

// TRUE 1:1 METRIC SCALE: 1 Three.js unit = 1.000 meter!
// Putter = 0.86 m (34 in), Ball Radius = 0.02135 m, 10-ft Cup at X = 3.048 m,
// Championship Oblong Green = 28 m x 18 m, USGA Flagstick = 2.13 m tall, Mature Pine/Oak Trees = 11 m to 18 m tall!
const CUP_X_3D = 3.048;

function createOblongGreenShape(rx, rz, cx, cz) {
  const shape = new THREE.Shape();
  const nPts = 140;
  for (let i = 0; i <= nPts; i++) {
    const ang = (i / nPts) * Math.PI * 2.0;
    const rMod = 1.0 + 0.12 * Math.cos(2.0 * ang) - 0.06 * Math.sin(3.0 * ang) + 0.04 * Math.cos(ang);
    const x = cx + rx * rMod * Math.cos(ang);
    const y = cz + rz * rMod * Math.sin(ang);
    if (i === 0) shape.moveTo(x, y);
    else shape.lineTo(x, y);
  }
  return shape;
}

function rebuildGolfCourseEnvironment(psiDeg) {
  while (courseGroup.children.length) courseGroup.remove(courseGroup.children[0]);
  const beta = (90.0 - psiDeg) * Math.PI / 180.0;
  const yGround = -L_CLUB * Math.cos(beta) - 0.02135;
  const zBall   =  L_CLUB * Math.sin(beta);

  // 0. 3D HD Atmospheric Sky Dome (160m radius) with Distant Mountain Ridge & Cirrus Clouds
  const skyDome = new THREE.Mesh(
    new THREE.SphereGeometry(155.0, 48, 32),
    new THREE.MeshBasicMaterial({ map: hdSkyTex, side: THREE.BackSide, fog: false })
  );
  skyDome.position.set(2.0, yGround + 4.0, zBall);
  courseGroup.add(skyDome);

  // 1. Expansive Rolling Fairway & Primary Rough Surrounding Terrain (180m diameter)
  const roughGeo = new THREE.CircleGeometry(140.0, 72);
  const roughMat = new THREE.MeshStandardMaterial({ color: 0x0f3f22, roughness: 0.92, metalness: 0.02 });
  const roughMesh = new THREE.Mesh(roughGeo, roughMat);
  roughMesh.rotation.x = -Math.PI / 2;
  roughMesh.position.set(4.5, yGround - 0.012, zBall);
  courseGroup.add(roughMesh);

  // Fairway Approach Apron (48m x 32m)
  const fairwayShape = createOblongGreenShape(26.0, 18.0, 4.5, 0.0);
  const fairwayMesh = new THREE.Mesh(
    new THREE.ShapeGeometry(fairwayShape),
    new THREE.MeshStandardMaterial({ color: 0x14532d, roughness: 0.85, metalness: 0.02 })
  );
  fairwayMesh.rotation.x = -Math.PI / 2;
  fairwayMesh.position.set(0, yGround - 0.006, zBall);
  courseGroup.add(fairwayMesh);

  // 2. True-Scale Oblong First-Cut Fringe / Collar Apron (31m x 21m, 1.5m collar width around green)
  const collarShape = createOblongGreenShape(15.5, 10.5, 4.2, 0.0);
  const collarGeo = new THREE.ShapeGeometry(collarShape);
  const collarMat = new THREE.MeshStandardMaterial({ color: 0x15803d, roughness: 0.78, metalness: 0.03 });
  const collarMesh = new THREE.Mesh(collarGeo, collarMat);
  collarMesh.rotation.x = -Math.PI / 2;
  collarMesh.position.set(0, yGround - 0.003, zBall);
  courseGroup.add(collarMesh);

  // 3. True-Scale Championship Oblong Putting Green (28m long x 18m wide, Rx = 14.0m, Rz = 9.0m)
  const greenRx = 14.0, greenRz = 9.0, greenCx = 4.2;
  const greenShape = createOblongGreenShape(greenRx, greenRz, greenCx, 0.0);
  const greenGeo = new THREE.ShapeGeometry(greenShape);
  const greenMat = new THREE.MeshStandardMaterial({
    color: 0x22c55e,
    map: hdBentgrassTex,
    bumpMap: hdBentgrassTex,
    bumpScale: 0.0025,
    roughness: 0.62,
    metalness: 0.04
  });
  const greenMesh = new THREE.Mesh(greenGeo, greenMat);
  greenMesh.rotation.x = -Math.PI / 2;
  greenMesh.position.set(0, yGround, zBall);
  courseGroup.add(greenMesh);

  // 4. Clipped Tournament 0.85m Walk-Mower Stripes inside the 28m x 18m Oblong Green
  const stripeWidth = 0.85;
  for (let sx = -8.5; sx <= 17.0; sx += stripeWidth * 2.0) {
    const dxNorm = (sx - greenCx) / greenRx;
    if (Math.abs(dxNorm) < 0.95) {
      const halfChordZ = greenRz * Math.sqrt(Math.max(0.02, 1.0 - dxNorm * dxNorm)) * 0.94;
      const sShape = new THREE.Shape();
      const sw = stripeWidth * 0.48;
      sShape.absarc(sx, -halfChordZ + sw, sw, Math.PI, 0, false);
      sShape.absarc(sx,  halfChordZ - sw, sw, 0, Math.PI, false);
      const sGeo = new THREE.ShapeGeometry(sShape);
      const sMesh = new THREE.Mesh(
        sGeo,
        new THREE.MeshStandardMaterial({ color: 0x4ade80, transparent: true, opacity: 0.15, roughness: 0.55 })
      );
      sMesh.rotation.x = -Math.PI / 2;
      sMesh.position.set(0, yGround + 0.001, zBall);
      courseGroup.add(sMesh);
    }
  }

  // 5. True-Scale Sculpted Greenside Sand Bunkers (7m x 4m, flanking the green at 10-14m distance)
  const bunkerSpecs = [
    { rx: 3.8, rz: 2.1, cx: 3.5, zOff: -11.4, col: 0xfde68a },
    { rx: 4.2, rz: 2.3, cx: 7.2, zOff:  11.2, col: 0xfef08a },
    { rx: 3.2, rz: 1.8, cx: 18.2, zOff: -5.5, col: 0xfde68a }
  ];
  bunkerSpecs.forEach(b => {
    const lipShape = createOblongGreenShape(b.rx * 1.08, b.rz * 1.08, b.cx, 0.0);
    const lipMesh = new THREE.Mesh(new THREE.ShapeGeometry(lipShape), new THREE.MeshStandardMaterial({ color: 0x14532d, roughness: 0.9 }));
    lipMesh.rotation.x = -Math.PI / 2;
    lipMesh.position.set(0, yGround - 0.0025, zBall + b.zOff);
    const sandShape = createOblongGreenShape(b.rx, b.rz, b.cx, 0.0);
    const sandMesh = new THREE.Mesh(new THREE.ShapeGeometry(sandShape), new THREE.MeshStandardMaterial({ color: b.col, roughness: 0.85 }));
    sandMesh.rotation.x = -Math.PI / 2;
    sandMesh.position.set(0, yGround - 0.0015, zBall + b.zOff);
    courseGroup.add(lipMesh, sandMesh);
  });

  // 6. True-Scale Mature Pine, Cypress & Oak Trees (11m to 18m tall — 15x taller than the 0.86m putter!)
  const treePositions = [
    [-14.0, -18.5, 13.5], [-6.5, -22.0, 15.2], [4.0, -21.5, 14.0], [14.5, -19.0, 16.5],
    [24.0, -14.5, 15.0], [28.5, -4.0, 17.0], [26.0, 9.5, 14.8], [16.5, 18.5, 15.6],
    [5.5, 20.0, 13.8], [-7.5, 18.0, 14.5], [-16.5, 12.0, 12.8], [35.0, -25.0, 18.0], [38.0, 16.0, 17.5]
  ];
  treePositions.forEach(([tx, tz, treeHeightM], idx) => {
    const trunkH = treeHeightM * 0.24;
    const trunkR = treeHeightM * 0.022;
    const trunk = new THREE.Mesh(
      new THREE.CylinderGeometry(trunkR * 0.7, trunkR, trunkH, 10),
      new THREE.MeshStandardMaterial({ color: 0x5c2d0c, roughness: 0.9 })
    );
    trunk.position.set(tx, yGround + trunkH * 0.5, zBall + tz);

    const c1H = treeHeightM * 0.45, c1R = treeHeightM * 0.20;
    const foliage1 = new THREE.Mesh(
      new THREE.ConeGeometry(c1R, c1H, 14),
      new THREE.MeshStandardMaterial({ color: 0x064e3b, roughness: 0.82 })
    );
    foliage1.position.set(tx, yGround + trunkH + c1H * 0.38, zBall + tz);

    const c2H = treeHeightM * 0.38, c2R = treeHeightM * 0.15;
    const foliage2 = new THREE.Mesh(
      new THREE.ConeGeometry(c2R, c2H, 14),
      new THREE.MeshStandardMaterial({ color: 0x047857, roughness: 0.78 })
    );
    foliage2.position.set(tx, yGround + trunkH + c1H * 0.68, zBall + tz);

    const c3H = treeHeightM * 0.30, c3R = treeHeightM * 0.10;
    const foliage3 = new THREE.Mesh(
      new THREE.ConeGeometry(c3R, c3H, 12),
      new THREE.MeshStandardMaterial({ color: 0x059669, roughness: 0.75 })
    );
    foliage3.position.set(tx, yGround + trunkH + c1H * 0.94, zBall + tz);
    courseGroup.add(trunk, foliage1, foliage2, foliage3);
  });

  // 7. True-Scale USGA Hole Cup (R = 0.054m) & Regulation 2.13m (7 ft) Flagstick at CUP_X_3D = 3.048m (10.0 ft)
  const cupMesh = new THREE.Mesh(
    new THREE.CircleGeometry(0.054, 32),
    new THREE.MeshBasicMaterial({ color: 0x05080f })
  );
  cupMesh.rotation.x = -Math.PI / 2;
  cupMesh.position.set(CUP_X_3D, yGround + 0.0025, zBall);
  const cupRim = new THREE.Mesh(
    new THREE.RingGeometry(0.054, 0.064, 32),
    new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide })
  );
  cupRim.rotation.x = -Math.PI / 2;
  cupRim.position.set(CUP_X_3D, yGround + 0.003, zBall);

  const pinHeightM = 2.13; // Regulation 7-ft USGA tournament flagstick
  const pinPole = new THREE.Mesh(
    new THREE.CylinderGeometry(0.0065, 0.0085, pinHeightM, 14),
    new THREE.MeshStandardMaterial({ color: 0xfef08a, metalness: 0.55, roughness: 0.25 })
  );
  pinPole.position.set(CUP_X_3D, yGround + pinHeightM * 0.5, zBall);
  const flagGeo = new THREE.PlaneGeometry(0.46, 0.32);
  const flagMat = new THREE.MeshStandardMaterial({ color: 0xef4444, roughness: 0.4, side: THREE.DoubleSide });
  const flagMesh = new THREE.Mesh(flagGeo, flagMat);
  flagMesh.position.set(CUP_X_3D + 0.23, yGround + pinHeightM - 0.18, zBall);
  courseGroup.add(cupMesh, cupRim, pinPole, flagMesh);
}

// 3D Dimpled Urethane Golf Ball on the Green (R = 0.02135 m, touching clubface at theta=0, t_imp=0.8168s!)
const golfBallMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.02135, 32, 32),
  new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: hdDimpleTex,
    bumpMap: hdDimpleTex,
    bumpScale: 0.0016,
    roughness: 0.18,
    metalness: 0.06
  })
);
scene.add(golfBallMesh);

// 3D Elastoplastic Impact Rebound Shock Ring at Clubface (flashes at t = 0.8168s!)
const impactFlashRing = new THREE.Mesh(
  new THREE.RingGeometry(0.022, 0.036, 32),
  new THREE.MeshBasicMaterial({ color: 0x38bdf8, side: THREE.DoubleSide, transparent: true, opacity: 0.0 })
);
impactFlashRing.rotation.y = Math.PI / 2;
scene.add(impactFlashRing);

// 3D Multi-Trial Monte Carlo Ball Roll Traces Group on the 3D Green
const ballTraces3DGroup = new THREE.Group();
scene.add(ballTraces3DGroup);

// 3D Impact Delivery Vectors Group (Face Angle Normal, Clubhead Path, & Ball Launch Direction at Impact)
const impactVecsGroup = new THREE.Group();
scene.add(impactVecsGroup);

// Inclined Stroke Plane translucent grid + Moving Butt Arc Curve + Clubhead Arc Curve
const planeGroup = new THREE.Group();
const buttArcGroup = new THREE.Group();
scene.add(planeGroup, buttArcGroup);

function rebuildPlaneMesh(psiDeg) {
  while (planeGroup.children.length) planeGroup.remove(planeGroup.children[0]);
  while (buttArcGroup.children.length) buttArcGroup.remove(buttArcGroup.children[0]);

  const beta = (90.0 - psiDeg) * Math.PI / 180.0;
  const geom = new THREE.PlaneGeometry(0.32, 0.92, 10, 14);
  const mat = new THREE.MeshBasicMaterial({ color: 0x0284c7, side: THREE.DoubleSide, transparent: true, opacity: 0.09 });
  const wire = new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true, transparent: true, opacity: 0.18 });
  const mesh = new THREE.Mesh(geom, mat);
  const wMesh = new THREE.Mesh(geom, wire);
  mesh.rotation.x = beta;
  wMesh.rotation.x = beta;
  mesh.position.set(0, -0.44 * Math.cos(beta), 0.44 * Math.sin(beta));
  wMesh.position.copy(mesh.position);
  planeGroup.add(mesh, wMesh);

  const buttPts = [];
  const headPts = [];
  const minDeg = -THETA_MAX_DEG;
  const maxDeg = THETA_MAX_DEG * 1.15 * POST_IMPACT_DECEL_RATIO;
  for (let deg = minDeg; deg <= maxDeg + 0.001; deg += (maxDeg - minDeg) / 36.0) {
    const rad = deg * Math.PI / 180.0;
    const cB = clubToThree(0.0, rad, 0, psiDeg);
    const cH = clubToThree(L_CLUB, rad, 0, psiDeg);
    buttPts.push(cB.pos);
    headPts.push(cH.pos);
  }
  const buttCurve = new THREE.CatmullRomCurve3(buttPts);
  const buttTube = new THREE.Mesh(
    new THREE.TubeGeometry(buttCurve, 40, 0.0042, 10, false),
    new THREE.MeshStandardMaterial({ color: 0xfacc15, emissive: 0xf59e0b, emissiveIntensity: 0.5 })
  );
  buttArcGroup.add(buttTube);

  const headCurve = new THREE.CatmullRomCurve3(headPts);
  const headTube = new THREE.Mesh(
    new THREE.TubeGeometry(headCurve, 50, 0.0042, 10, false),
    new THREE.MeshStandardMaterial({ color: 0x22d3ee, emissive: 0x0284c7, emissiveIntensity: 0.45 })
  );
  planeGroup.add(headTube);

  rebuildGolfCourseEnvironment(psiDeg);
}

// Shoulder Pivot Hub, Forearm Triangle Links, Moving Butt Sphere, Shaft, Grip, & Putter Head
const shoulderSphere = new THREE.Mesh(
  new THREE.SphereGeometry(0.014, 18, 18),
  new THREE.MeshStandardMaterial({ color: 0xfacc15, metalness: 0.5, roughness: 0.3 })
);
const leftArmCylinder = new THREE.Mesh(
  new THREE.CylinderGeometry(0.0045, 0.0045, 1.0, 10),
  new THREE.MeshStandardMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.55 })
);
const rightArmCylinder = new THREE.Mesh(
  new THREE.CylinderGeometry(0.0045, 0.0045, 1.0, 10),
  new THREE.MeshStandardMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.55 })
);
buttArcGroup.add(shoulderSphere, leftArmCylinder, rightArmCylinder);

const pivotSphere = new THREE.Mesh(new THREE.SphereGeometry(0.013, 20, 20), new THREE.MeshStandardMaterial({ color: 0xfacc15, emissive: 0xd97706, emissiveIntensity: 0.35 }));
scene.add(pivotSphere);

const shaftMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc, metalness: 0.7, roughness: 0.2 });
const gripMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.65 });
const headMat = new THREE.MeshStandardMaterial({ color: 0x22d3ee, metalness: 0.45, roughness: 0.2 });
const ghostMat = new THREE.MeshBasicMaterial({ color: 0xf97316, wireframe: true, transparent: true, opacity: 0.55 });

const gripCylinder = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.0095, 0.26, 20), gripMat);
const shaftCylinder = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.005, L_CLUB, 16), shaftMat);
const putterHead = new THREE.Mesh(new THREE.BoxGeometry(0.034, 0.024, 0.115), headMat);
const ghostShaft = new THREE.Mesh(new THREE.CylinderGeometry(0.0045, 0.0045, L_CLUB, 10), ghostMat);
const ghostHead = new THREE.Mesh(new THREE.BoxGeometry(0.034, 0.024, 0.115), ghostMat);
scene.add(gripCylinder, shaftCylinder, putterHead, ghostShaft, ghostHead);

const contactColors = [0x38bdf8, 0x2563eb, 0xfbbf24, 0xea580c];
const contactSpheres = contactColors.map(c => {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.0095, 16, 16), new THREE.MeshStandardMaterial({ color: c, emissive: c, emissiveIntensity: 0.45 }));
  scene.add(m);
  return m;
});

const midColors = [0x22d3ee, 0xf97316];
const midDiamonds = midColors.map(c => {
  const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.013, 0), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: c, emissiveIntensity: 0.7 }));
  scene.add(m);
  return m;
});

function createThickForceArrow(colorHex) {
  const group = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: colorHex, emissive: colorHex, emissiveIntensity: 0.40, roughness: 0.25 });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.0042, 0.0042, 1.0, 12), mat);
  shaft.position.y = 0.5;
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.0115, 0.026, 14), mat);
  head.position.y = 1.0;
  group.add(shaft, head);
  group.userData = { shaft, head };
  scene.add(group);
  return group;
}

function setThickForceArrow(arrowGroup, origin, dirVec, totalLen) {
  arrowGroup.position.copy(origin);
  arrowGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dirVec.clone().normalize());
  const headLen = 0.026;
  const shaftLen = Math.max(0.01, totalLen - headLen);
  arrowGroup.userData.shaft.scale.set(1, shaftLen, 1);
  arrowGroup.userData.shaft.position.y = shaftLen * 0.5;
  arrowGroup.userData.head.position.y = shaftLen + headLen * 0.5;
  return origin.clone().addScaledVector(dirVec.clone().normalize(), totalLen + 0.015);
}

const forceArrows = contactColors.map(c => createThickForceArrow(c));

// Impact Face Angle (Gold) & Clubhead Path (Violet) 3D Arrows at the Ball Impact Zone
const impactPathArrow = createThickForceArrow(0xa855f7);
const impactFaceArrow = createThickForceArrow(0xfacc15);
const impactLaunchArrow = createThickForceArrow(0x10b981);
impactVecsGroup.add(impactPathArrow, impactFaceArrow, impactLaunchArrow);

function createCurvedTorqueArrow(colorHex, baseRadius = 0.036) {
  const group = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: colorHex, emissive: colorHex, emissiveIntensity: 0.55, roughness: 0.2 });

  const arcAngle = 1.25 * Math.PI;
  const curvePts = [];
  const nSeg = 28;
  for (let i = 0; i <= nSeg; i++) {
    const ang = -0.35 * Math.PI + (i / nSeg) * arcAngle;
    curvePts.push(new THREE.Vector3(baseRadius * Math.cos(ang), baseRadius * Math.sin(ang), 0));
  }
  const arcCurve = new THREE.CatmullRomCurve3(curvePts);
  const tubeGeo = new THREE.TubeGeometry(arcCurve, 28, 0.0046, 10, false);
  const tubeMesh = new THREE.Mesh(tubeGeo, mat);

  const endAng = -0.35 * Math.PI + arcAngle;
  const endPos = new THREE.Vector3(baseRadius * Math.cos(endAng), baseRadius * Math.sin(endAng), 0);
  const tangentDir = new THREE.Vector3(-Math.sin(endAng), Math.cos(endAng), 0).normalize();
  const coneMesh = new THREE.Mesh(new THREE.ConeGeometry(0.012, 0.028, 14), mat);
  coneMesh.position.copy(endPos);
  coneMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangentDir);

  const hubRing = new THREE.Mesh(new THREE.TorusGeometry(0.012, 0.0022, 8, 16), mat);
  group.add(tubeMesh, coneMesh, hubRing);
  group.userData = { baseRadius, endPos };
  scene.add(group);
  return group;
}

const torqueArrows = [
  createCurvedTorqueArrow(0x22d3ee, 0.035),
  createCurvedTorqueArrow(0xf97316, 0.038),
];
const netTorqueCurvedArrow = createCurvedTorqueArrow(0x10b981, 0.048);

rebuildPlaneMesh(state.psiLieDeg);

function alignCylinder(mesh, pStart, pEnd, customLen = false) {
  const dir = new THREE.Vector3().subVectors(pEnd, pStart);
  const len = dir.length();
  mesh.position.copy(pStart).addScaledVector(dir, 0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
  if (customLen) mesh.scale.set(1, len, 1);
  return len;
}

// Per user request: "Don't show the values of the torque on the screen with live readouts"
// Symbolic labels are hidden by default and, when enabled via checkbox, show ONLY the mathematical symbol!
function project3DLabel(elId, worldPos, symbolHtml, visible = true) {
  const el = document.getElementById(elId);
  if (!el) return;
  if (!visible || !state.showVecLabels) {
    el.style.display = 'none';
    return;
  }
  const v = worldPos.clone().project(camera);
  if (v.z > 1.0 || v.z < -1.0) {
    el.style.display = 'none';
    return;
  }
  const xPx = (v.x * 0.5 + 0.5) * container.clientWidth;
  const yPx = (-v.y * 0.5 + 0.5) * container.clientHeight;
  el.style.display = 'block';
  el.style.left = `${xPx.toFixed(1)}px`;
  el.style.top = `${yPx.toFixed(1)}px`;
  el.innerHTML = symbolHtml;
}

