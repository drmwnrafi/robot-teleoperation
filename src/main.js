import * as THREE from 'three';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

// ─── Landmark metadata ────────────────────────────────────────────────────────
const LANDMARK_NAMES = [
  'WRIST',
  'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP',  'THUMB_TIP',
  'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
  'MIDDLE_MCP','MIDDLE_PIP','MIDDLE_DIP','MIDDLE_TIP',
  'RING_MCP',  'RING_PIP',  'RING_DIP',  'RING_TIP',
  'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',
];

// Bone pairs (indices into the 21-landmark array)
const CONNECTIONS = [
  // palm
  [0, 1], [0, 5], [0, 9], [0, 13], [0, 17], [5, 9], [9, 13], [13, 17],
  // thumb
  [1, 2], [2, 3], [3, 4],
  // index
  [5, 6], [6, 7], [7, 8],
  // middle
  [9, 10], [10, 11], [11, 12],
  // ring
  [13, 14], [14, 15], [15, 16],
  // pinky
  [17, 18], [18, 19], [19, 20],
];

// Fingertip landmark indices (shown larger / highlighted)
const TIPS = new Set([4, 8, 12, 16, 20]);
// Key joints shown in the coord panel with a highlight
const KEY_JOINTS = new Set([0, 4, 8, 12, 16, 20]);

// ─── Persistent settings (saved to localStorage) ─────────────────────────────
function lsGet(key, fallback) {
  try { const v = localStorage.getItem(key); return v !== null ? JSON.parse(v) : fallback; }
  catch { return fallback; }
}
function lsSet(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* quota exceeded / private mode */ }
}

// Physical distance wrist (lm[0]) → middle-MCP (lm[9]) for an average adult hand
let HAND_SIZE_CM   = lsGet('ht_handSize', 9.0);
// Webcam horizontal field-of-view (degrees). Typical USB webcam ≈ 60-70°
let CAMERA_FOV_DEG = lsGet('ht_camFov', 62.0);
// Three.js scene scale: 1 cm = CM_SCALE scene units (10 cm ≈ 1 unit, hand ~2 units tall)
const CM_SCALE = 0.1;

// ─── DOM refs ────────────────────────────────────────────────────────────────
const video         = document.getElementById('video');
const overlay       = document.getElementById('overlay');
const ctx           = overlay.getContext('2d');
const threePanel    = document.getElementById('three-panel');
const coordsEl      = document.getElementById('coords-panel');
const statusEl      = document.getElementById('status-badge');
const gestureEl     = document.getElementById('gesture-badge');
const handSizeInput = document.getElementById('hand-size-input');
const fovInput      = document.getElementById('fov-input');
const depthDisplay  = document.getElementById('depth-display');
const robotOut      = document.getElementById('robot-out');
const btnCamera     = document.getElementById('btn-camera');
const btnVideo      = document.getElementById('btn-video');
const fileInput     = document.getElementById('file-input');
const videoControls = document.getElementById('video-controls');
const btnPlayPause  = document.getElementById('btn-playpause');
const seekEl        = document.getElementById('seek');
const timeDisplay   = document.getElementById('time-display');
const speedSelect   = document.getElementById('speed-select');
const wsUrlInput    = document.getElementById('ws-url-input');
const btnWs         = document.getElementById('btn-ws');
const wsStatusDot   = document.getElementById('ws-status');

// Mirror flag: true for webcam (feels natural), false for video files
let isMirrored = true;

// ─── WebSocket manager ────────────────────────────────────────────────────────
// Sends robot data frames to a WebSocket server (e.g. Python bridge or ROS).
// Auto-reconnects with exponential back-off up to MAX_RETRY attempts.
class RobotSocket {
  static THROTTLE_MS = 33; // ~30 fps cap — prevents flooding slow robot bridges
  static MAX_RETRY   = 10;

  constructor() {
    this._ws         = null;
    this._enabled    = false;
    this._url        = '';
    this._attempts   = 0;
    this._retryTimer = null;
    this._lastSend   = 0;
  }

  connect(url) {
    this._url      = url.trim();
    this._enabled  = true;
    this._attempts = 0;
    clearTimeout(this._retryTimer);
    this._tryConnect();
  }

  disconnect() {
    this._enabled = false;
    clearTimeout(this._retryTimer);
    if (this._ws) {
      this._ws.onclose = null; // prevent auto-retry on intentional close
      this._ws.close();
      this._ws = null;
    }
    this._updateStatus('disconnected');
  }

  /** Send data; silently dropped if not connected or throttle interval not elapsed. */
  send(data) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    const now = performance.now();
    if (now - this._lastSend < RobotSocket.THROTTLE_MS) return;
    this._lastSend = now;
    try { this._ws.send(JSON.stringify(data)); } catch { /* ws closed mid-send */ }
  }

  _tryConnect() {
    if (!this._enabled) return;
    this._updateStatus('connecting');
    let ws;
    try {
      ws = new WebSocket(this._url);
    } catch {
      this._scheduleRetry();
      return;
    }
    this._ws = ws;
    ws.onopen  = () => { this._attempts = 0; this._updateStatus('connected'); };
    ws.onerror = () => { this._updateStatus('error'); };
    ws.onclose = () => {
      this._ws = null;
      if (this._enabled) this._scheduleRetry();
      else this._updateStatus('disconnected');
    };
  }

  _scheduleRetry() {
    if (this._attempts >= RobotSocket.MAX_RETRY) {
      this._updateStatus('error');
      this._enabled = false;
      if (btnWs) { btnWs.textContent = 'Connect'; btnWs.classList.remove('active'); }
      return;
    }
    const delay = Math.min(1000 * (1 << this._attempts), 10000); // 1s, 2s, 4s … 10s
    this._attempts++;
    this._retryTimer = setTimeout(() => this._tryConnect(), delay);
    this._updateStatus('connecting');
  }

  _updateStatus(state) {
    if (wsStatusDot) {
      wsStatusDot.className = `ws-dot ${state}`;
      const labels = { connecting: 'Connecting…', connected: 'Connected', disconnected: 'Disconnected', error: 'Error — retrying' };
      wsStatusDot.title = labels[state] ?? state;
    }
    if (btnWs) {
      btnWs.textContent = this._enabled ? 'Disconnect' : 'Connect';
      btnWs.classList.toggle('active', state === 'connected');
    }
  }
}

const robotSocket = new RobotSocket();

// ─── Gesture detection ────────────────────────────────────────────────────────
// Returns one of: 'GRAB' | 'OPEN' | 'POINT' | 'PEACE' | null (unknown/transition)
// All comparisons are wrist-relative → fully scale- and distance-invariant.
function detectGesture(lm) {
  const wrist = lm[0];
  const dist2  = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;

  // A finger is "curled" when its tip is closer to the wrist than its MCP joint.
  // 10% hysteresis (0.9 factor) reduces flicker at the boundary.
  const curled = (mcpIdx, tipIdx) =>
    dist2(wrist, lm[tipIdx]) < dist2(wrist, lm[mcpIdx]) * 0.81; // 0.9² = 0.81

  const ci = curled(5,  8);   // index
  const cm = curled(9,  12);  // middle
  const cr = curled(13, 16);  // ring
  const cp = curled(17, 20);  // pinky

  // Specific gestures FIRST — before GRAB, otherwise "index out + 3 curled"
  // would match GRAB (>=3) before we ever reach POINT.
  if (!ci &&  cm &&  cr &&  cp) return 'POINT';  // only index extended
  if (!ci && !cm &&  cr &&  cp) return 'PEACE';  // index + middle extended (V sign)

  const curledCount = [ci, cm, cr, cp].filter(Boolean).length;
  // GRAB: original used >= 3 — tolerates one stiff/partially-curled finger.
  if (curledCount >= 3) return 'GRAB';
  // OPEN: allow one slightly-bent finger (curledCount <= 1)
  if (curledCount <= 1) return 'OPEN';

  return null; // transitional / unrecognised pose
}

// ─── Three.js setup ──────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setClearColor(0x0d1117);
renderer.setPixelRatio(window.devicePixelRatio);
threePanel.appendChild(renderer.domElement);

// CSS2D label renderer (overlaid on top of the WebGL canvas for distance text)
const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0';
labelRenderer.domElement.style.left = '0';
labelRenderer.domElement.style.pointerEvents = 'none';
threePanel.appendChild(labelRenderer.domElement);

const scene  = new THREE.Scene();
// Camera sits at origin (= real camera position), looks toward -Z.
// FOV is set dynamically in resizeAll() to match the real camera.
const cam3d  = new THREE.PerspectiveCamera(50, 1, 0.001, 50);
cam3d.position.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dirLight = new THREE.DirectionalLight(0x00ff88, 2.5);
dirLight.position.set(0.2, 0.3, -0.4);
scene.add(dirLight);

// Axis helper placed at a typical hand depth (0.6 m = 60 cm)
const axesHelper = new THREE.AxesHelper(0.05);
axesHelper.position.set(0, 0, -0.6);
scene.add(axesHelper);

// ── Per-hand colour palettes ──
const HAND_PALETTE = [
  { normal: 0x00ff88, tip: 0xffff00, grab: 0xff4444, open: 0x00ff88, point: 0x58a6ff, peace: 0xf0b429, bone: 0x00aa55, boneActive: 0xaa2222 },
  { normal: 0x58a6ff, tip: 0xffa500, grab: 0xff8800, open: 0x58a6ff, point: 0xff88ff, peace: 0xffbb44, bone: 0x2255aa, boneActive: 0xaa6600 },
];

// ── 21 joint spheres × 2 hands ──
const GEO_JOINT = new THREE.SphereGeometry(0.03, 10, 10);
const GEO_TIP   = new THREE.SphereGeometry(0.048, 10, 10);
const handJointMeshes = HAND_PALETTE.map(palette =>
  Array.from({ length: 21 }, (_, i) => {
    const m = new THREE.Mesh(
      TIPS.has(i) ? GEO_TIP : GEO_JOINT,
      new THREE.MeshStandardMaterial({ color: palette.normal })
    );
    m.visible = false;
    scene.add(m);
    return m;
  })
);

// ── Bone lines × 2 hands ──
const handBoneLines = HAND_PALETTE.map(palette =>
  CONNECTIONS.map(([a, b]) => {
    const geo  = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(), new THREE.Vector3(),
    ]);
    const mat  = new THREE.LineBasicMaterial({ color: palette.bone });
    const line = new THREE.Line(geo, mat);
    line.visible = false;
    scene.add(line);
    return { line, a, b, mat };
  })
);

// ── Distance labels in 3D ──
// Per hand: wrist→tip for all 5 fingers; inter-hand: wrist-to-wrist.
const DIST_FINGER_PAIRS = [[0,4],[0,8],[0,12],[0,16],[0,20]];
const DIST_FINGER_NAMES = ['Thumb','Index','Middle','Ring','Pinky'];

function makeDistLabel(color = '#ffffff') {
  const div = document.createElement('div');
  div.style.cssText = `font-size:10px;font-family:Consolas,monospace;color:${color};
    background:rgba(13,17,23,0.75);padding:1px 5px;border-radius:3px;
    white-space:nowrap;pointer-events:none;letter-spacing:0.5px;`;
  const obj = new CSS2DObject(div);
  obj.visible = false;
  scene.add(obj);
  return { obj, div };
}

// [hand0: 5 finger labels], [hand1: 5 finger labels], [inter-hand label]
const handDistLabels = HAND_PALETTE.map((p, h) =>
  DIST_FINGER_NAMES.map(() => makeDistLabel(h === 0 ? '#00ff88' : '#58a6ff'))
);
const interHandLabel = makeDistLabel('#f0b429');
// Returns the actual displayed content rect within the video CSS box,
// accounting for object-fit: cover (the video is cropped, not letterboxed).
function getVideoContentRect() {
  const W  = overlay.width  || 640;
  const H  = overlay.height || 480;
  const vw = video.videoWidth  || W;
  const vh = video.videoHeight || H;
  // cover: scale so the smaller dimension fits, then crop the larger
  const scale = Math.max(W / vw, H / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const ox = (W - dw) / 2;  // negative means content exceeds canvas
  const oy = (H - dh) / 2;
  return { ox, oy, dw, dh, W, H };
}

function resizeAll() {
  // 2-D overlay matches the video element's rendered size
  const vRect = video.getBoundingClientRect();
  overlay.width  = vRect.width  || 640;
  overlay.height = vRect.height || 480;

  // Three.js renderer fills its panel
  const tw = threePanel.clientWidth;
  const th = threePanel.clientHeight;
  if (tw > 0 && th > 0) {
    renderer.setSize(tw, th, false);
    labelRenderer.setSize(tw, th);
    cam3d.aspect = tw / th;
    // Convert real camera horizontal FOV → Three.js vertical FOV
    const hfovRad = CAMERA_FOV_DEG * Math.PI / 180;
    cam3d.fov = 2 * Math.atan(Math.tan(hfovRad / 2) / cam3d.aspect) * 180 / Math.PI;
    cam3d.updateProjectionMatrix();
  }
}
window.addEventListener('resize', resizeAll);
setTimeout(resizeAll, 200); // run after layout paints

// ── Render loop ──
(function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, cam3d);
  labelRenderer.render(scene, cam3d);
})();

// ─── Colors ──────────────────────────────────────────────────────────────────
const GESTURE_COLOR = { GRAB: '#ff4444', OPEN: '#00ff88', POINT: '#58a6ff', PEACE: '#f0b429' };

// ─── Hand results callback ────────────────────────────────────────────────────
function onHandResults(results) {
  resizeAll();
  const W = overlay.width;
  const H = overlay.height;

  ctx.clearRect(0, 0, W, H);

  const numHands = results.multiHandLandmarks?.length ?? 0;

  if (numHands === 0) {
    handJointMeshes.forEach(hm => hm.forEach(m => { m.visible = false; }));
    handBoneLines.forEach(hl => hl.forEach(({ line }) => { line.visible = false; }));
    handDistLabels.forEach(hl => hl.forEach(l => { l.obj.visible = false; }));
    interHandLabel.obj.visible = false;
    gestureEl.classList.remove('active');
    gestureEl.textContent = '';
    coordsEl.innerHTML = '<span class="no-hand">No hand detected</span>';
    statusEl.textContent = 'Waiting for hand…';
    return;
  }

  statusEl.textContent = numHands === 1 ? 'Tracking 1 hand ✓' : 'Tracking 2 hands ✓';

  const AR     = overlay.width / (overlay.height || 1);
  const f_norm = 0.5 / Math.tan(CAMERA_FOV_DEG * Math.PI / 360);
  const { ox, oy, dw, dh } = getVideoContentRect();
  const lmX = (nx) => isMirrored ? ox + (1 - nx) * dw : ox + nx * dw;
  const lmY = (ny) => oy + ny * dh;

  const allHandData  = [];
  const gestureTexts = [];

  for (let h = 0; h < numHands; h++) {
    const lm      = results.multiHandLandmarks[h];
    const wLm     = results.multiHandWorldLandmarks?.[h];
    const palette = HAND_PALETTE[h] ?? HAND_PALETTE[0];
    const rawLabel  = results.multiHandedness?.[h]?.label ?? `Hand ${h + 1}`;
    // MediaPipe labels from the person's own perspective (mirror of camera view).
    // Swap Left↔Right so the label matches what the user sees on screen.
    const handLabel = rawLabel === 'Left' ? 'Right' : rawLabel === 'Right' ? 'Left' : rawLabel;

    // ── Depth estimation ────────────────────────────────────────────────────
    const d2d_norm = Math.hypot(lm[0].x - lm[9].x, (lm[0].y - lm[9].y) / AR);
    const depthCm  = (HAND_SIZE_CM * f_norm) / (d2d_norm + 1e-6);
    if (h === 0 && depthDisplay) depthDisplay.textContent = depthCm.toFixed(1);

    const wristXcm =  (lm[0].x - 0.5) / f_norm * depthCm;
    const wristYcm = -(lm[0].y - 0.5) / f_norm * depthCm;
    const wristZcm =  depthCm;

    // ── Gesture ─────────────────────────────────────────────────────────────
    const gesture = detectGesture(lm);
    const isGrab  = gesture === 'GRAB';
    if (gesture) gestureTexts.push({ label: handLabel, gesture });

    // ── 2-D skeleton overlay ────────────────────────────────────────────────
    const baseColor = h === 0 ? '#00ff88' : '#58a6ff';
    const boneColor = GESTURE_COLOR[gesture] ?? baseColor;
    ctx.lineWidth   = 2;
    ctx.strokeStyle = boneColor;

    CONNECTIONS.forEach(([a, b]) => {
      ctx.beginPath();
      ctx.moveTo(lmX(lm[a].x), lmY(lm[a].y));
      ctx.lineTo(lmX(lm[b].x), lmY(lm[b].y));
      ctx.stroke();
    });

    lm.forEach((pt, i) => {
      const r = TIPS.has(i) ? 7 : 4;
      ctx.fillStyle = TIPS.has(i) ? boneColor : (isGrab ? '#ff8888' : (h === 0 ? '#00cc66' : '#4488cc'));
      ctx.beginPath();
      ctx.arc(lmX(pt.x), lmY(pt.y), r, 0, Math.PI * 2);
      ctx.fill();
    });

    // Hand label drawn next to wrist
    ctx.font      = 'bold 13px Consolas, monospace';
    ctx.fillStyle = baseColor;
    ctx.fillText(handLabel, lmX(lm[0].x) + 10, lmY(lm[0].y) - 8);

    // ── 3-D positions in cm ─────────────────────────────────────────────────
    let pos_cm;
    if (wLm && wLm.length === 21) {
      pos_cm = wLm.map(wpt => ({
        x: wristXcm + wpt.x * 100,
        y: wristYcm - wpt.y * 100,  // world y is DOWN (image-space) → flip to y-up
        z: wristZcm - wpt.z * 100,
      }));
    } else {
      pos_cm = lm.map(pt => ({
        x:  (pt.x - 0.5) / f_norm * depthCm,
        y: -(pt.y - 0.5) / f_norm * depthCm,
        z:  depthCm - pt.z * 100,
      }));
    }

    allHandData.push({ handLabel, depthCm, wristXcm, wristYcm, wristZcm, pos_cm, pos3d: null, gesture, isGrab });
  }

  // ── Absolute 3D positions (camera frame → Three.js) ─────────────────────────
  // Camera is at origin. Real camera: x=right, y=up, z=depth(+away).
  // Three.js: x=right, y=up, z=toward viewer (+) → depth maps to -Z.
  const allPos3d = [];
  for (let h = 0; h < numHands; h++) {
    const { pos_cm, gesture, isGrab } = allHandData[h];
    const palette = HAND_PALETTE[h] ?? HAND_PALETTE[0];

    const pos3d = pos_cm.map(p => ({
      x:  p.x * CM_SCALE,
      y:  p.y * CM_SCALE,
      z: -p.z * CM_SCALE,   // depth → -Z
    }));
    allPos3d.push(pos3d);
    allHandData[h].pos3d = pos3d;

    // ── Update Three.js spheres ─────────────────────────────────────────────
    const jointColor = (i) => {
      if (!TIPS.has(i)) return isGrab ? palette.grab : palette.normal;
      const gc = { GRAB: palette.grab, OPEN: palette.open, POINT: palette.point, PEACE: palette.peace };
      return gc[gesture] ?? palette.tip;
    };
    handJointMeshes[h].forEach((mesh, i) => {
      mesh.visible = true;
      mesh.position.set(pos3d[i].x, pos3d[i].y, pos3d[i].z);
      mesh.material.color.setHex(jointColor(i));
    });

    // ── Update bone lines ───────────────────────────────────────────────────
    handBoneLines[h].forEach(({ line, a, b, mat }) => {
      line.visible = true;
      const attr = line.geometry.attributes.position;
      attr.setXYZ(0, pos3d[a].x, pos3d[a].y, pos3d[a].z);
      attr.setXYZ(1, pos3d[b].x, pos3d[b].y, pos3d[b].z);
      attr.needsUpdate = true;
      mat.color.setHex(isGrab ? palette.boneActive : palette.bone);
    });

    // ── Finger distance labels (wrist → each fingertip) ───────────────────
    DIST_FINGER_PAIRS.forEach(([a, b], fi) => {
      const pa = pos_cm[a], pb = pos_cm[b];
      const distCm = Math.hypot(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z);
      const mid3d  = {
        x: (pos3d[a].x + pos3d[b].x) / 2,
        y: (pos3d[a].y + pos3d[b].y) / 2,
        z: (pos3d[a].z + pos3d[b].z) / 2,
      };
      const lbl = handDistLabels[h][fi];
      lbl.div.textContent = `${DIST_FINGER_NAMES[fi]} ${distCm.toFixed(1)}cm`;
      lbl.obj.position.set(mid3d.x, mid3d.y, mid3d.z);
      lbl.obj.visible = true;
    });
  }

  // Hide unused finger labels
  for (let h = numHands; h < HAND_PALETTE.length; h++) {
    handJointMeshes[h].forEach(m => { m.visible = false; });
    handBoneLines[h].forEach(({ line }) => { line.visible = false; });
    handDistLabels[h].forEach(l => { l.obj.visible = false; });
  }

  // ── Inter-hand distance label ──────────────────────────────────────────────
  if (numHands === 2) {
    const w0 = allHandData[0], w1 = allHandData[1];
    const distCm = Math.hypot(
      w1.wristXcm - w0.wristXcm,
      w1.wristYcm - w0.wristYcm,
      w1.wristZcm - w0.wristZcm
    );
    const p0 = allPos3d[0][0], p1 = allPos3d[1][0];
    interHandLabel.div.textContent = `↔ ${distCm.toFixed(1)} cm`;
    interHandLabel.obj.position.set(
      (p0.x + p1.x) / 2,
      (p0.y + p1.y) / 2 + 0.15,
      (p0.z + p1.z) / 2
    );
    interHandLabel.obj.visible = true;
  } else {
    interHandLabel.obj.visible = false;
  }
  if (gestureTexts.length > 0) {
    const first = gestureTexts[0];
    gestureEl.textContent    = numHands > 1
      ? gestureTexts.map(g => `${g.label}: ${g.gesture}`).join('  |  ')
      : first.gesture;
    gestureEl.style.background = GESTURE_COLOR[first.gesture] ?? '#888';
    gestureEl.classList.add('active');
  } else {
    gestureEl.classList.remove('active');
    gestureEl.textContent = '';
  }

  // ── Coordinates panel ───────────────────────────────────────────────────────
  const coordsHtml = allHandData.map(({ handLabel, pos_cm }, h) => {
    const headerColor = h === 0 ? '#00ff88' : '#58a6ff';
    const cols = pos_cm.map((p, i) => {
      const isKey = KEY_JOINTS.has(i);
      return `<div class="coord-col${isKey ? ' key-joint' : ''}">
        <span class="jname">${LANDMARK_NAMES[i]}</span>
        <span class="axis">x:<b>${p.x.toFixed(1)}</b></span>
        <span class="axis">y:<b>${p.y.toFixed(1)}</b></span>
        <span class="axis">z:<b>${p.z.toFixed(1)}</b></span>
      </div>`;
    });
    return `<div class="coord-hand-label" style="color:${headerColor};writing-mode:vertical-rl;padding:4px 6px;font-size:10px;flex-shrink:0;border-right:1px solid #30363d;letter-spacing:1px;">${handLabel}</div>${cols.join('')}`;
  }).join('');
  coordsEl.innerHTML = coordsHtml || '<span class="no-hand">No hand detected</span>';

  // ── Robot data output ───────────────────────────────────────────────────────
  const robotData = {
    t:     Date.now(),
    hands: allHandData.map(({ handLabel, depthCm, wristXcm, wristYcm, wristZcm, pos_cm, gesture, isGrab }) => ({
      label:     handLabel,
      depth_cm:  +depthCm.toFixed(1),
      wrist_cm:  { x: +wristXcm.toFixed(2), y: +wristYcm.toFixed(2), z: +wristZcm.toFixed(2) },
      joints_cm: pos_cm.map(p => [+p.x.toFixed(2), +p.y.toFixed(2), +p.z.toFixed(2)]),
      gesture,
      is_grab: isGrab,
    })),
  };
  window.__handRobotData = robotData;
  window.dispatchEvent(new CustomEvent('hand-robot-data', { detail: robotData }));
  robotSocket.send(robotData);

  if (robotOut) {
    if (numHands > 1) {
      robotOut.textContent = allHandData.map(d =>
        `${d.handLabel}: wrist(${d.wristXcm.toFixed(1)}, ${d.wristYcm.toFixed(1)}, ${d.wristZcm.toFixed(1)}) gesture:${d.gesture ?? '—'}`
      ).join('  ||  ');
    } else {
      const { wristXcm, wristYcm, wristZcm, pos_cm, gesture } = allHandData[0];
      robotOut.textContent = `wrist(${wristXcm.toFixed(1)}, ${wristYcm.toFixed(1)}, ${wristZcm.toFixed(1)}) cm  |  tip[8]=(${pos_cm[8].x.toFixed(1)}, ${pos_cm[8].y.toFixed(1)}, ${pos_cm[8].z.toFixed(1)})  gesture:${gesture ?? '—'}`;
    }
  }
}

// ─── MediaPipe Hands init ─────────────────────────────────────────────────────
const hands = new window.Hands({
  locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${f}`,
});
hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});
hands.onResults(onHandResults);

// ─── Source management ────────────────────────────────────────────────────────
let webcamStream  = null;
let mpCamInstance = null;
let videoFileLoop = false;

function fmt(s) {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}

// ── Webcam mode ──
async function startWebcam() {
  stopVideoFile();
  isMirrored = true;
  video.classList.remove('no-mirror');
  videoControls.classList.remove('visible');
  btnCamera.classList.add('active');
  btnVideo.classList.remove('active');

  statusEl.textContent = 'Requesting camera…';
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: 'user' },
    });
    video.srcObject = webcamStream;
    video.src = '';
    await new Promise(r => video.addEventListener('loadeddata', r, { once: true }));
    resizeAll();
    statusEl.textContent = 'Loading MediaPipe…';
    mpCamInstance = new window.Camera(video, {
      async onFrame() { await hands.send({ image: video }); },
      width: 1280,
      height: 720,
    });
    mpCamInstance.start();
    statusEl.textContent = 'Waiting for hand…';
  } catch (err) {
    statusEl.textContent = `Camera error: ${err.message}`;
    coordsEl.innerHTML = `<span class="no-hand">⚠ ${err.message}</span>`;
  }
}

// ── Video file mode ──
function stopVideoFile() {
  videoFileLoop = false;
}

function stopWebcam() {
  if (mpCamInstance) { try { mpCamInstance.stop(); } catch (_) {} mpCamInstance = null; }
  if (webcamStream)  { webcamStream.getTracks().forEach(t => t.stop()); webcamStream = null; }
  video.srcObject = null;
}

async function startVideoFile(file) {
  console.log('[startVideoFile] called with:', file.name);
  try {
  stopWebcam();
  stopVideoFile();

  isMirrored = false;
  video.classList.add('no-mirror');
  videoControls.classList.add('visible');
  btnVideo.classList.add('active');
  btnCamera.classList.remove('active');

  const url = URL.createObjectURL(file);
  video.src = url;
  video.loop = true;
  video.muted = true;
  video.playbackRate = parseFloat(speedSelect.value);

  await new Promise((resolve, reject) => {
    video.addEventListener('loadedmetadata', resolve, { once: true });
    video.addEventListener('error', () => reject(new Error(`Cannot load video: ${video.error?.message || 'unsupported format'}`)), { once: true });
  });
  resizeAll();
  seekEl.max = video.duration;
  timeDisplay.textContent = `0:00 / ${fmt(video.duration)}`;
  video.play();
  btnPlayPause.textContent = '⏸';
  statusEl.textContent = `Video: ${file.name}`;

  videoFileLoop = true;
  (async function loop() {
    if (!videoFileLoop) return;
    if (!video.paused && !video.ended && video.readyState >= 2) {
      await hands.send({ image: video });
    }
    requestAnimationFrame(loop);
  })();
  } catch (err) {
    console.error('[startVideoFile] ERROR:', err);
    statusEl.textContent = `Video error: ${err.message}`;
  }
}

// ─── Video controls events ────────────────────────────────────────────────────
btnPlayPause.addEventListener('click', () => {
  if (video.paused) { video.play(); btnPlayPause.textContent = '⏸'; }
  else              { video.pause(); btnPlayPause.textContent = '▶'; }
});

speedSelect.addEventListener('change', () => {
  video.playbackRate = parseFloat(speedSelect.value);
});

video.addEventListener('timeupdate', () => {
  if (!isNaN(video.duration)) {
    seekEl.value = video.currentTime;
    timeDisplay.textContent = `${fmt(video.currentTime)} / ${fmt(video.duration)}`;
  }
});

let seeking = false;
seekEl.addEventListener('mousedown', () => { seeking = true; video.pause(); });
seekEl.addEventListener('input',     () => { video.currentTime = parseFloat(seekEl.value); });
seekEl.addEventListener('mouseup',   () => {
  seeking = false;
  video.play();
  btnPlayPause.textContent = '⏸';
});

// ─── Source toggle buttons ────────────────────────────────────────────────────
btnCamera.addEventListener('click', () => {
  stopVideoFile();
  startWebcam();
});

// btnVideo is now a <label for="file-input"> — browser opens file picker natively.

fileInput.addEventListener('change', () => {
  console.log('[change] files:', fileInput.files.length, fileInput.files[0]?.name);
  if (fileInput.files[0]) startVideoFile(fileInput.files[0]);
  fileInput.value = '';
});

// ─── Calibration controls ─────────────────────────────────────────────────────
if (handSizeInput) {
  handSizeInput.value = HAND_SIZE_CM;
  handSizeInput.addEventListener('input', () => {
    const v = parseFloat(handSizeInput.value);
    if (v > 0 && v < 40) { HAND_SIZE_CM = v; lsSet('ht_handSize', v); }
  });
}
if (fovInput) {
  fovInput.value = CAMERA_FOV_DEG;
  fovInput.addEventListener('input', () => {
    const v = parseFloat(fovInput.value);
    if (v > 10 && v < 170) { CAMERA_FOV_DEG = v; lsSet('ht_camFov', v); resizeAll(); }
  });
}
if (wsUrlInput) {
  wsUrlInput.value = lsGet('ht_wsUrl', 'ws://localhost:8765');
}
if (btnWs) {
  btnWs.addEventListener('click', () => {
    if (robotSocket._enabled) {
      robotSocket.disconnect();
    } else {
      const url = wsUrlInput?.value.trim() || 'ws://localhost:8765';
      lsSet('ht_wsUrl', url);
      robotSocket.connect(url);
    }
  });
}

// ─── Start with webcam by default ────────────────────────────────────────────
startWebcam();
