import * as THREE from 'three';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

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

// Pose upper-body bone pairs (MediaPipe Pose landmark indices)
const POSE_UPPER = [
  [11, 12],           // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // torso sides
  [23, 24],           // hips
];

// Fingertip landmark indices (shown larger / highlighted)
const TIPS = new Set([4, 8, 12, 16, 20]);

// ── Vector math helpers (for wrist orientation & robot metrics) ─────────────────
const vsub   = (a, b) => ({ x: a.x-b.x, y: a.y-b.y, z: a.z-b.z });
const vnorm  = v => { const m = Math.hypot(v.x, v.y, v.z) + 1e-9; return { x: v.x/m, y: v.y/m, z: v.z/m }; };
const vcross = (a, b) => ({ x: a.y*b.z - a.z*b.y, y: a.z*b.x - a.x*b.z, z: a.x*b.y - a.y*b.x });
const vdot   = (a, b) => a.x*b.x + a.y*b.y + a.z*b.z;
const vfmt   = (v, d = 3) => [+v.x.toFixed(d), +v.y.toFixed(d), +v.z.toFixed(d)];

// Per-finger joint chains (world landmark indices): [base, pip, dip, tip]
const FINGER_JOINTS = [
  [1,  2,  3,  4],  // thumb
  [5,  6,  7,  8],  // index
  [9,  10, 11, 12], // middle
  [13, 14, 15, 16], // ring
  [17, 18, 19, 20], // pinky
];
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
let HAND_SIZE_CM       = lsGet('ht_handSize',     9.0);
// Webcam horizontal field-of-view (degrees). Typical USB webcam ≈ 60-70°, phone ≈ 75-100°
let CAMERA_FOV_DEG     = lsGet('ht_camFov',       62.0);
// Shoulder-to-shoulder physical width (cm). Used to estimate body depth from pose.
let SHOULDER_WIDTH_CM  = lsGet('ht_shoulderWidth', 40.0);
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
const handSizeInput      = document.getElementById('hand-size-input');
const fovInput           = document.getElementById('fov-input');
const shoulderWidthInput = document.getElementById('shoulder-width-input');
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
const btnModeRos    = document.getElementById('btn-mode-ros');

// Mirror flag: true for webcam (feels natural), false for video files
let isMirrored = true;

// ─── WebSocket manager ────────────────────────────────────────────────────────
// Sends robot data frames to a WebSocket server (e.g. Python bridge or ROS).
// Auto-reconnects with exponential back-off up to MAX_RETRY attempts.
class RobotSocket {
  static THROTTLE_MS = 33; 
  static MAX_RETRY   = 10;

  constructor() {
    this._ws         = null;
    this._enabled    = false;
    this._url        = '';
    this._attempts   = 0;
    this._retryTimer = null;
    this._lastSend   = 0;
    
    // ─── DUAL MODE SUPPORT ──────────────────────────────────────────────
    // 'raw' -> Plain JSON (for Python websockets, Node.js, etc.)
    // 'ros' -> rosbridge protocol (for ROS 2 rosbridge_server)
    this._mode       = 'raw'; 
    
    // ROS specific config (only used if mode is 'ros')
    this._rosTopic   = '/raw_landmarks';
    this._rosType    = 'std_msgs/String';
    this._advertised = false;
    // ────────────────────────────────────────────────────────────────────
  }

  /**
   * Set the connection mode.
   * @param {string} mode - 'raw' or 'ros'
   */
  setMode(mode) {
    this._mode = mode;
    console.log(`[RobotSocket] Mode set to: ${mode}`);
    
    if (mode === 'ros' && this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._advertiseTopic();
    } else if (mode === 'raw') {
      this._advertised = false; 
    }
  }

  connect(url) {
    this._url        = url.trim();
    this._enabled    = true;
    this._attempts   = 0;
    this._advertised = false;
    clearTimeout(this._retryTimer);
    this._tryConnect();
  }

  disconnect() {
    this._enabled    = false;
    this._advertised = false;
    clearTimeout(this._retryTimer);
    if (this._ws) {
      this._ws.onclose = null;
      this._ws.close();
      this._ws = null;
    }
    this._updateStatus('disconnected');
  }

  send(data) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    // In ROS mode, wait until the topic is advertised before sending data
    if (this._mode === 'ros' && !this._advertised) return; 
    
    const now = performance.now();
    if (now - this._lastSend < RobotSocket.THROTTLE_MS) return;
    this._lastSend = now;
    
    let payload;
    if (this._mode === 'ros') {
      // Wrap in rosbridge publish message
      payload = {
        op: 'publish',
        topic: this._rosTopic,
        msg: { data: JSON.stringify(data) } // Sends JSON as a string inside std_msgs/String
      };
    } else {
      // Send raw JSON for plain Python/Node servers
      payload = data;
    }
    
    try { 
      this._ws.send(JSON.stringify(payload)); 
    } catch { /* ws closed mid-send */ }
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
    ws.onopen  = () => { 
      this._attempts = 0; 
      this._updateStatus('connected'); 
      // If in ROS mode, immediately advertise the topic upon connection
      if (this._mode === 'ros') {
        this._advertiseTopic();
      }
    };
    ws.onerror = () => { this._updateStatus('error'); };
    ws.onclose = () => {
      this._ws = null;
      this._advertised = false;
      if (this._enabled) this._scheduleRetry();
      else this._updateStatus('disconnected');
    };
  }

  _advertiseTopic() {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    const advMsg = {
      op: 'advertise',
      topic: this._rosTopic,
      type: this._rosType
    };
    try {
      this._ws.send(JSON.stringify(advMsg));
      this._advertised = true;
      console.log(`[RobotSocket] Advertised ROS topic '${this._rosTopic}'`);
    } catch (e) {
      console.error('[RobotSocket] Failed to advertise', e);
    }
  }

  _scheduleRetry() {
    if (this._attempts >= RobotSocket.MAX_RETRY) {
      this._updateStatus('error');
      this._enabled = false;
      if (btnWs) { btnWs.textContent = 'Connect'; btnWs.classList.remove('active'); }
      return;
    }
    const delay = Math.min(1000 * (1 << this._attempts), 10000);
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
      const isConnected = this._enabled && state === 'connected';
      btnWs.textContent = isConnected ? 'Disconnect' : 'Connect';
      btnWs.classList.toggle('active', isConnected);
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
labelRenderer.domElement.style.pointerEvents = 'auto'; // needed for OrbitControls on front view
threePanel.appendChild(labelRenderer.domElement);

// CSS2D label renderer (overlaid on top of the WebGL canvas for distance text)
const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0';
labelRenderer.domElement.style.left = '0';
labelRenderer.domElement.style.pointerEvents = 'auto'; // needed for OrbitControls on front view
threePanel.appendChild(labelRenderer.domElement);

const scene  = new THREE.Scene();
// Front view camera — interactive (OrbitControls). Starts near real-camera viewpoint.
// Far plane = 200 scene units = 2000 cm = 20 m, well beyond any realistic body depth.
// Near plane stays tiny so hands very close to the camera still render.
const cam3d  = new THREE.PerspectiveCamera(60, 1, 0.01, 200);
cam3d.position.set(0, 1.0, -2.0);

// OrbitControls bound to the left (FRONT) half of the panel via labelRenderer.domElement
const controls3d = new OrbitControls(cam3d, labelRenderer.domElement);
controls3d.dampingFactor = 0.08;
controls3d.minDistance   = 0.1;
controls3d.maxDistance   = 100.0;  
controls3d.target.set(0, 0, -4.5);
controls3d.update();
// Double-click to reset camera to default position
labelRenderer.domElement.addEventListener('dblclick', () => {
  cam3d.position.set(0, 1.0, -2.0);
  controls3d.target.copy(_orbitTarget);
  controls3d.update();
});

// Scene-space center of detected hands — orbit target smoothly follows this
const _orbitTarget = new THREE.Vector3(0, 0, -4.5);

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dirLight = new THREE.DirectionalLight(0x00ff88, 2.5);
dirLight.position.set(0.2, 0.3, -0.4);
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
// ── 21 joint spheres × 2 hands ──
const GEO_JOINT = new THREE.SphereGeometry(0.03, 10, 10);
const GEO_TIP   = new THREE.SphereGeometry(0.048, 10, 10);
const jointMats = Array.from({ length: 21 }, () =>
  new THREE.MeshStandardMaterial({ color: 0x00ff88 })
);
const jointMeshes = Array.from({ length: 21 }, (_, i) => {
  const m = new THREE.Mesh(TIPS.has(i) ? GEO_TIP : GEO_JOINT, jointMats[i]);
  m.visible = false;
  scene.add(m);
  return m;
});

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

// ── Body skeleton (MediaPipe Pose upper body) ──
const bodyBoneLines = POSE_UPPER.map(() => {
  const geo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(), new THREE.Vector3(),
  ]);
  const mat = new THREE.LineBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.7 });
  const line = new THREE.Line(geo, mat);
  line.visible = false;
  scene.add(line);
  return { line };
});

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

// ── Body skeleton (MediaPipe Pose upper body) ──
const bodyBoneLines = POSE_UPPER.map(() => {
  const geo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(), new THREE.Vector3(),
  ]);
  const mat = new THREE.LineBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.7 });
  const line = new THREE.Line(geo, mat);
  line.visible = false;
  scene.add(line);
  return { line };
});

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

  const tw = threePanel.clientWidth;
  const th = threePanel.clientHeight;
  if (tw > 0 && th > 0) {
    renderer.setSize(tw, th, false);
    labelRenderer.setSize(tw, th);
  }
}
window.addEventListener('resize', resizeAll);
setTimeout(resizeAll, 200); // run after layout paints

// ── Render loop ──
(function animate() {
  requestAnimationFrame(animate);

  const tw = threePanel.clientWidth || 100;
  const th = threePanel.clientHeight || 100;

  // Orbit center slowly drifts to where the hands are
  controls3d.target.lerp(_orbitTarget, 0.02);

  cam3d.aspect = tw / th;
  cam3d.updateProjectionMatrix();
  controls3d.update();

  renderer.setViewport(0, 0, tw, th);
  renderer.render(scene, cam3d);
  labelRenderer.render(scene, cam3d);
})();

// ─── Colors ──────────────────────────────────────────────────────────────────
const GESTURE_COLOR  = { GRAB: '#ff4444', OPEN: '#00ff88', POINT: '#58a6ff', PEACE: '#f0b429' };
const GESTURE_ID     = { GRAB: 1, OPEN: 2, POINT: 3, PEACE: 4 };  // numeric, in addition to string

// MediaPipe Pose body landmark names (for self-describing body output)
const BODY_LANDMARK_NAMES = {
  0:'nose', 11:'shoulder_L', 12:'shoulder_R', 13:'elbow_L', 14:'elbow_R',
  15:'wrist_L', 16:'wrist_R', 23:'hip_L', 24:'hip_R',
};

// Latest pose image landmarks — written by onPoseResults, read by onHandResults for 2-D overlay
let currentPoseLms    = null;
let currentBodyMetrics = null;          // body joint cm positions for robot WS
const smoothDepth     = [null, null];   // per-hand EMA depth smoothing
let   lastHandDepthCm = null;           // most recent hand depth — used to anchor body skeleton
let   lastBodyDepthCm = null;           // most recent body depth — used to clamp hand depth (occlusion)
const lastWrist3d     = {};             // { 'Left': {x,y,z}, 'Right': {x,y,z} } — real 3D hand wrist positions

// WebSocket frame counter (monotonic, used by receiver to detect dropped packets)
let _frameSeq = 0;
// FPS measurement (sliding window of last 30 frames)
const _frameTimes = [];

// ─── Hand results callback ────────────────────────────────────────────────────
function determineHandLabel(handLm) {
  if (!currentPoseLms) return null;

  const wrist = handLm[0];

  const leftWrist = currentPoseLms[15];
  const rightWrist = currentPoseLms[16];

  if (!leftWrist || !rightWrist) return null;

  const dLeft = Math.hypot(
    wrist.x - leftWrist.x,
    wrist.y - leftWrist.y
  );

  const dRight = Math.hypot(
    wrist.x - rightWrist.x,
    wrist.y - rightWrist.y
  );

  return dLeft < dRight ? "Left" : "Right";
}

function onHandResults(results) {
  resizeAll();
  const W = overlay.width;
  const H = overlay.height;

  ctx.clearRect(0, 0, W, H);

  // ── 2‑D body skeleton overlay ─────────────────────────────────────────────
  if (currentPoseLms) {
    const { ox, oy, dw, dh } = getVideoContentRect();
    const plmX = (nx) => isMirrored ? ox + (1 - nx) * dw : ox + nx * dw;
    const plmY = (ny) => oy + ny * dh;
    ctx.strokeStyle = 'rgba(160,160,160,0.55)';
    ctx.lineWidth = 2;
    POSE_UPPER.forEach(([a, b]) => {
      const pa = currentPoseLms[a], pb = currentPoseLms[b];
      if (!pa || !pb) return;
      ctx.beginPath();
      ctx.moveTo(plmX(pa.x), plmY(pa.y));
      ctx.lineTo(plmX(pb.x), plmY(pb.y));
      ctx.stroke();
    });
  }

  const numHands = results.multiHandLandmarks?.length ?? 0;
  const maxDisplayHands = Math.min(numHands, HAND_PALETTE.length);

  // Hide hand visuals if no hands, but continue to body data publishing
  if (numHands === 0) {
    handJointMeshes.forEach(hm => hm.forEach(m => { m.visible = false; }));
    handBoneLines.forEach(hl => hl.forEach(({ line }) => { line.visible = false; }));
    handDistLabels.forEach(hl => hl.forEach(l => { l.obj.visible = false; }));
    interHandLabel.obj.visible = false;
    gestureEl.classList.remove('active');
    gestureEl.textContent = '';
    coordsEl.innerHTML = '<span class="no-hand">No hand detected</span>';
    statusEl.textContent = 'Waiting for hand…';
  } else {
    statusEl.textContent = numHands === 1 ? 'Tracking 1 hand ✓' : 'Tracking 2 hands ✓';
  }

  const AR = (video.videoWidth || overlay.width) / (video.videoHeight || overlay.height || 1);
  const f_norm = 0.5 / Math.tan(CAMERA_FOV_DEG * Math.PI / 360);
  const { ox, oy, dw, dh } = getVideoContentRect();
  const lmX = (nx) => isMirrored ? ox + (1 - nx) * dw : ox + nx * dw;
  const lmY = (ny) => oy + ny * dh;

  const allHandData = [];
  const gestureTexts = [];

  // ── Process hands (only up to available resources) ────────────────────────
  for (let h = 0; h < maxDisplayHands; h++) {
    const lm = results.multiHandLandmarks[h];
    const wLm = results.multiHandWorldLandmarks?.[h];
    const palette = HAND_PALETTE[h];
    const rawLabel = results.multiHandedness?.[h]?.label ?? `Hand ${h + 1}`;
    const handLabel = isMirrored
      ? (rawLabel === 'Left' ? 'Right' : rawLabel === 'Right' ? 'Left' : rawLabel)
      : rawLabel;

    // Depth estimation
    const SEG_PAIRS = [
      [0, 9, 1.00], [0, 5, 0.95], [0, 17, 0.85], [5, 17, 0.78]
    ];
    const DEPTH_MIN = 10, DEPTH_MAX = 500;
    const validDepths = SEG_PAIRS.map(([a, b, ratio]) => {
      const d2d = Math.hypot(lm[a].x - lm[b].x, (lm[a].y - lm[b].y) / AR);
      const lenCm = HAND_SIZE_CM * ratio;
      const d = (lenCm * f_norm) / (d2d + 1e-9);
      return (d >= DEPTH_MIN && d <= DEPTH_MAX) ? d : null;
    }).filter(d => d !== null).sort((x, y) => x - y);
    let rawDepth;
    if (validDepths.length >= 2) {
      rawDepth = (validDepths[Math.floor(validDepths.length / 2) - 1] +
                  validDepths[Math.floor(validDepths.length / 2)]) / 2;
    } else if (validDepths.length === 1) {
      rawDepth = validDepths[0];
    } else {
      rawDepth = smoothDepth[h] ?? 100;
    }
    if (!smoothDepth[h]) smoothDepth[h] = rawDepth;
    smoothDepth[h] = smoothDepth[h] * 0.90 + rawDepth * 0.10;
    let depthCm = smoothDepth[h];
    if (lastBodyDepthCm !== null && lastBodyDepthCm > 20 && lastBodyDepthCm < 500) {
      depthCm = Math.min(depthCm, lastBodyDepthCm);
    }
    smoothDepth[h] = depthCm;

    lastHandDepthCm = smoothDepth.filter(Boolean).reduce((s, v) => s + v, 0) /
                      smoothDepth.filter(Boolean).length;
    if (h === 0 && depthDisplay) depthDisplay.textContent = depthCm.toFixed(1);

    const wristXcm = (lm[0].x - 0.5) / f_norm * depthCm;
    const wristYcm = -(lm[0].y - 0.5) / f_norm * depthCm;
    const wristZcm = depthCm;

    const gesture = detectGesture(lm);
    const isGrab = gesture === 'GRAB';
    if (gesture) gestureTexts.push({ label: handLabel, gesture });

    // 2‑D overlay drawing
    const baseColor = h === 0 ? '#00ff88' : '#58a6ff';
    const boneColor = GESTURE_COLOR[gesture] ?? baseColor;
    ctx.lineWidth = 2;
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
    ctx.font = 'bold 13px Consolas, monospace';
    ctx.fillStyle = baseColor;
    ctx.fillText(handLabel, lmX(lm[0].x) + 10, lmY(lm[0].y) - 8);

    // 3‑D positions (cm)
    const pos_cm = lm.map((pt, i) => {
      const relZcm = (wLm && wLm[i]) ? wLm[i].z * 100 : (pt.z * depthCm);
      const jd = depthCm + relZcm;
      return {
        x: (pt.x - 0.5) / f_norm * jd,
        y: -(pt.y - 0.5) / f_norm * jd,
        z: jd,
      };
    });

    allHandData.push({
      handLabel, depthCm, wristXcm, wristYcm, wristZcm, pos_cm, wLm,
      palmNormal: null, fingerDir: null, fingerCurl: null, pinchCm: null,
      pos3d: null, gesture, isGrab
    });

    // Orientation metrics (optional)
    if (wLm && wLm.length === 21) {
      const toYup = v => ({ x: v.x, y: -v.y, z: -v.z });
      const fwd = vnorm(vsub(wLm[9], wLm[0]));
      const side = vnorm(vsub(wLm[17], wLm[5]));
      allHandData[h].palmNormal = vfmt(toYup(vnorm(vcross(fwd, side))));
      allHandData[h].fingerDir = vfmt(toYup(fwd));
      allHandData[h].fingerCurl = FINGER_JOINTS.map(([base, pip, , tip]) => {
        const cosA = vdot(vnorm(vsub(wLm[pip], wLm[base])), vnorm(vsub(wLm[tip], wLm[pip])));
        return +((1 - cosA) / 2).toFixed(3);
      });
      allHandData[h].pinchCm = [8, 12, 16, 20].map(i => +(Math.hypot(
        (wLm[i].x - wLm[4].x) * 100,
        (wLm[i].y - wLm[4].y) * 100,
        (wLm[i].z - wLm[4].z) * 100
      )).toFixed(1));
    }
  }

  // Update orbit target (use body center if no hands)
  if (allHandData.length > 0) {
    const n = allHandData.length;
    const avgX = allHandData.reduce((s, d) => s + d.wristXcm, 0) / n;
    const avgY = allHandData.reduce((s, d) => s + d.wristYcm, 0) / n;
    const avgZ = allHandData.reduce((s, d) => s + d.wristZcm, 0) / n;
    _orbitTarget.set(avgX * CM_SCALE, avgY * CM_SCALE, -avgZ * CM_SCALE);
  } else if (currentBodyMetrics && currentBodyMetrics.shoulder_L && currentBodyMetrics.shoulder_R) {
    const [lx, ly, lz] = currentBodyMetrics.shoulder_L;
    const [rx, ry, rz] = currentBodyMetrics.shoulder_R;
    const midX = (lx + rx) / 2;
    const midY = (ly + ry) / 2;
    const midZ = (lz + rz) / 2;
    _orbitTarget.set(midX * CM_SCALE, midY * CM_SCALE, -midZ * CM_SCALE);
  }

  // ── 3D updates for hands (only for those we actually processed) ───────────
  const allPos3d = [];
  for (let h = 0; h < maxDisplayHands; h++) {
    const handData = allHandData[h];
    if (!handData) continue; // safety

    const { pos_cm, wLm: hWLm, gesture, isGrab } = handData;
    const palette = HAND_PALETTE[h];
    const pos3d = pos_cm.map(p => ({
      x: p.x * CM_SCALE,
      y: p.y * CM_SCALE,
      z: -p.z * CM_SCALE,
    }));
    allPos3d.push(pos3d);
    handData.pos3d = pos3d;
    lastWrist3d[handData.handLabel] = pos3d[0];

    // Joint spheres
    const meshes = handJointMeshes[h];
    if (meshes) {
      meshes.forEach((mesh, i) => {
        if (pos3d[i]) {
          mesh.visible = true;
          mesh.position.set(pos3d[i].x, pos3d[i].y, pos3d[i].z);
          const jointColor = (i) => {
            if (!TIPS.has(i)) return isGrab ? palette.grab : palette.normal;
            const gc = { GRAB: palette.grab, OPEN: palette.open, POINT: palette.point, PEACE: palette.peace };
            return gc[gesture] ?? palette.tip;
          };
          mesh.material.color.setHex(jointColor(i));
        } else {
          mesh.visible = false;
        }
      });
    }

    // Bone lines
    const bones = handBoneLines[h];
    if (bones) {
      bones.forEach(({ line, a, b, mat }) => {
        line.visible = true;
        const attr = line.geometry.attributes.position;
        attr.setXYZ(0, pos3d[a].x, pos3d[a].y, pos3d[a].z);
        attr.setXYZ(1, pos3d[b].x, pos3d[b].y, pos3d[b].z);
        attr.needsUpdate = true;
        mat.color.setHex(isGrab ? palette.boneActive : palette.bone);
      });
    }

    // Distance labels
    const labels = handDistLabels[h];
    if (labels) {
      DIST_FINGER_PAIRS.forEach(([a, b], fi) => {
        const distCm = hWLm && hWLm.length === 21
          ? Math.hypot((hWLm[b].x - hWLm[a].x) * 100, (hWLm[b].y - hWLm[a].y) * 100, (hWLm[b].z - hWLm[a].z) * 100)
          : Math.hypot(pos_cm[b].x - pos_cm[a].x, pos_cm[b].y - pos_cm[a].y, pos_cm[b].z - pos_cm[a].z);
        const mid3d = {
          x: (pos3d[a].x + pos3d[b].x) / 2,
          y: (pos3d[a].y + pos3d[b].y) / 2,
          z: (pos3d[a].z + pos3d[b].z) / 2,
        };
        const lbl = labels[fi];
        if (lbl) {
          lbl.div.textContent = `${DIST_FINGER_NAMES[fi]} ${distCm.toFixed(1)}cm`;
          lbl.obj.position.set(mid3d.x, mid3d.y, mid3d.z);
          lbl.obj.visible = true;
        }
      });
    }
  }

  // Hide any leftover visual resources (for indices beyond maxDisplayHands)
  for (let h = maxDisplayHands; h < HAND_PALETTE.length; h++) {
    handJointMeshes[h]?.forEach(m => { m.visible = false; });
    handBoneLines[h]?.forEach(({ line }) => { line.visible = false; });
    handDistLabels[h]?.forEach(l => { l.obj.visible = false; });
  }

  // Inter‑hand distance label
  if (allHandData.length === 2 && allPos3d.length === 2) {
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

  // Update gesture badge
  if (gestureTexts.length > 0) {
    const first = gestureTexts[0];
    gestureEl.textContent = gestureTexts.length > 1
      ? gestureTexts.map(g => `${g.label}: ${g.gesture}`).join('  |  ')
      : first.gesture;
    gestureEl.style.background = GESTURE_COLOR[first.gesture] ?? '#888';
    gestureEl.classList.add('active');
  } else {
    gestureEl.classList.remove('active');
    gestureEl.textContent = '';
  }

  // Coordinates panel
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
    });
    return `<div class="coord-hand-label" style="color:${headerColor};writing-mode:vertical-rl;padding:4px 6px;font-size:10px;flex-shrink:0;border-right:1px solid #30363d;letter-spacing:1px;">${handLabel}</div>${cols.join('')}`;
  }).join('');
  coordsEl.innerHTML = coordsHtml || '<span class="no-hand">No hand detected</span>';

  // ── ROBOT DATA (always built and sent) ─────────────────────────────────────
  const now = Date.now();
  _frameSeq++;
  _frameTimes.push(now);
  if (_frameTimes.length > 30) _frameTimes.shift();
  const fpsMeasured = _frameTimes.length > 1
    ? +((_frameTimes.length - 1) * 1000 / (_frameTimes[_frameTimes.length - 1] - _frameTimes[0])).toFixed(1)
    : 0;

  const robotData = {
    t: now,
    seq: _frameSeq,
    fps: fpsMeasured,
    frame: { origin: 'camera', x: 'right', y: 'up', z: 'away_from_camera', units: 'cm' },
    calibration: {
      hand_size_cm: +HAND_SIZE_CM.toFixed(2),
      fov_deg: +CAMERA_FOV_DEG.toFixed(1),
      shoulder_width_cm: +SHOULDER_WIDTH_CM.toFixed(1),
    },
    hands: ['Right', 'Left'].map((slotLabel) => {
      const flipLabel = isMirrored ? slotLabel : (slotLabel === 'Left' ? 'Right' : 'Left');
      const hi = allHandData.findIndex(d => d.handLabel === flipLabel);
      if (hi < 0) {
        return { label: flipLabel, present: false };
      }
      const { handLabel, depthCm, wristXcm, wristYcm, wristZcm, pos_cm,
              palmNormal, fingerDir, fingerCurl, pinchCm, gesture, isGrab } = allHandData[hi];
      return {
        label: flipLabel,
        present: true,
        index: hi,
        confidence: results.multiHandedness?.[hi]?.score ?? null,
        depth_cm: +depthCm.toFixed(1),
        wrist_cm: { x: +wristXcm.toFixed(2), y: +wristYcm.toFixed(2), z: +wristZcm.toFixed(2) },
        palm_normal: palmNormal,
        finger_dir: fingerDir,
        finger_curl: fingerCurl,
        pinch_cm: pinchCm,
        grip_aperture_cm: pinchCm?.[0] ?? null,
        gesture: gesture,
        gesture_id: gesture ? GESTURE_ID[gesture] ?? 0 : 0,
        is_grab: isGrab,
        joints_cm: pos_cm.map(p => ({ x: +p.x.toFixed(2), y: +p.y.toFixed(2), z: +p.z.toFixed(2) })),
      };
    }),
    body: currentBodyMetrics,
    landmark_names: {
      hand_joints: LANDMARK_NAMES,
      body_joints: BODY_LANDMARK_NAMES,
    },
  };

  if (allHandData.length === 2) {
    const a = allHandData[0], b = allHandData[1];
    robotData.interhand_cm = +Math.hypot(
      b.wristXcm - a.wristXcm, b.wristYcm - a.wristYcm, b.wristZcm - a.wristZcm
    ).toFixed(1);
  }

  window.__handRobotData = robotData;
  window.dispatchEvent(new CustomEvent('hand-robot-data', { detail: robotData }));
  robotSocket.send(robotData);
  if (robotOut) {
    const fmt = (p) => `(${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)})`;
    const handLines = allHandData.map(d => {
      const p3 = d.pos3d;
      if (!p3) return '';
      return [
        ''
      ].join('\n');
    });
    robotOut.style.whiteSpace = 'pre';
    robotOut.style.fontSize   = '10px';
  }
}

// ─── MediaPipe Pose results callback ─────────────────────────────────────────
function onPoseResults(results) {
  currentPoseLms = results.poseLandmarks ?? null;
  if (!currentPoseLms) {
    bodyBoneLines.forEach(({ line }) => { line.visible = false; });
    return;
  }

  const f_norm = 0.5 / Math.tan(CAMERA_FOV_DEG * Math.PI / 360);

  // Anchor depth: body uses ITS OWN depth estimate (from shoulder width), not the hand's.
  // Using hand depth for the body would squish the whole skeleton to the hand's distance
  // and wreck the arm geometry whenever a hand is extended forward.
  const sL = currentPoseLms[11], sR = currentPoseLms[12];
  const shoulderNorm = Math.abs(sR.x - sL.x);
  const shoulderDepthCm = (SHOULDER_WIDTH_CM * f_norm) / (shoulderNorm + 1e-9);
  const bodyDepthCm = shoulderDepthCm;
  lastBodyDepthCm = bodyDepthCm;  // expose to onHandResults for occlusion clamp
  const mirX = isMirrored ? -1 : 1;

  // Body 3D positions using the SAME projection as hands: image-landmark x,y projected
  // through the camera ray at per-joint depth = bodyDepthCm + world relative z.
  // Hand and body now live in one consistent cm frame, so the wrist override below
  // is a tiny snap (just depth-estimator disagreement), not a coordinate-system jump.
  const wPose = results.poseWorldLandmarks;
  const bodyPos3d = currentPoseLms.map((lm2d, idx) => {
    const relZcm = (wPose && wPose[idx]) ? wPose[idx].z * 100 : 0;
    const jd     = bodyDepthCm + relZcm;
    return {
      x: (lm2d.x - 0.5) / f_norm * jd * CM_SCALE,
      y: -(lm2d.y - 0.5) / f_norm * jd * CM_SCALE,
      z: -jd * CM_SCALE,
    };
  });

  // Override pose wrists with hand-tracker wrists (hand has its own, more accurate, depth).
  // handLabel reflects the person's actual hand (MediaPipe gives reversed labels for
  // unflipped input; the swap at top of onHandResults corrects that). So person-left → idx 15.
  if (lastWrist3d['Left'])  bodyPos3d[15] = lastWrist3d['Left'];
  if (lastWrist3d['Right']) bodyPos3d[16] = lastWrist3d['Right'];

  // Reconstruct elbow on the shoulder→wrist line. MediaPipe Pose can't reliably localize
  // the elbow in 3D when the arm points toward the camera (heavy foreshortening), and any
  // off-line elbow position produces the "bent forearm pointing up" artifact you saw.
  // Upper-arm:forearm ratio ≈ 28:26 → elbow at ~52% from shoulder to wrist.
  // (If the user genuinely bends an arm sideways, this loses the bend; acceptable trade-off
  // for the user's robotic-arm use case where arms are mostly extended.)
  const ELBOW_FRACTION = 0.52;
  const interpJoint = (s, w, f) => ({
    x: s.x + f * (w.x - s.x),
    y: s.y + f * (w.y - s.y),
    z: s.z + f * (w.z - s.z),
  });
  if (lastWrist3d['Right']) bodyPos3d[14] = interpJoint(bodyPos3d[12], bodyPos3d[16], ELBOW_FRACTION);
  if (lastWrist3d['Left'])  bodyPos3d[13] = interpJoint(bodyPos3d[11], bodyPos3d[15], ELBOW_FRACTION);

  // Debug: store body coords globally (read by onHandResults to avoid flicker)
  const f3 = (p) => `(${p.x.toFixed(2)},${p.y.toFixed(2)},${p.z.toFixed(2)})`;
  const fw = (w) => w ? `(${w.x.toFixed(3)},${w.y.toFixed(3)},${w.z.toFixed(3)})` : 'n/a';
  const bnames = { 11:'shldr_L', 12:'shldr_R', 13:'elbow_L', 14:'elbow_R', 15:'wrist_L', 16:'wrist_R' };
  window.__bodyDbg = `=== BODY  depth=${bodyDepthCm.toFixed(1)}cm mirX=${isMirrored} ===\n`
    + Object.entries(bnames).map(([i,n]) =>
        `  ${n.padEnd(8)} ${f3(bodyPos3d[+i])}  raw:${fw(wPose?.[+i])}`
    ).join('\n');

  POSE_UPPER.forEach(([a, b], i) => {
    const { line } = bodyBoneLines[i];
    const pa = bodyPos3d[a], pb = bodyPos3d[b];
    const attr = line.geometry.attributes.position;
    attr.setXYZ(0, pa.x, pa.y, pa.z);
    attr.setXYZ(1, pb.x, pb.y, pb.z);
    attr.needsUpdate = true;
    line.visible = true;
  });

  // Robot telemetry uses the CORRECTED bodyPos3d (wrist overrides + straightened elbow),
  // not the raw pose landmarks — that way the robot gets the same geometry the user sees.
  const scene2cm = (p) => p ? [
    +(p.x / CM_SCALE).toFixed(1),
    +(p.y / CM_SCALE).toFixed(1),
    +(-p.z / CM_SCALE).toFixed(1),
  ] : null;
  currentBodyMetrics = {
    shoulder_L: scene2cm(bodyPos3d[11]),
    shoulder_R: scene2cm(bodyPos3d[12]),
    elbow_L:    scene2cm(bodyPos3d[13]),
    elbow_R:    scene2cm(bodyPos3d[14]),
    wrist_L:    scene2cm(bodyPos3d[15]),
    wrist_R:    scene2cm(bodyPos3d[16]),
  };
}

// ─── MediaPipe Hands init ─────────────────────────────────────────────────────
const hands = new window.Hands({
  locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${f}`,
});
hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 0,
  minDetectionConfidence: 0.1,
  minTrackingConfidence: 0.1,
});
hands.onResults(onHandResults);

// ─── MediaPipe Pose init ──────────────────────────────────────────────────────
const pose = new window.Pose({
  locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`,
});
pose.setOptions({
  modelComplexity: 2,
  smoothLandmarks: true,
  enableSegmentation: false,
  minDetectionConfidence: 0.1,
  minTrackingConfidence: 0.1,
});
pose.onResults(onPoseResults);

// ─── Source management ────────────────────────────────────────────────────────
let webcamStream  = null;
let mpCamInstance = null;
let videoFileLoop = false;
let _poseFrame    = 0; // run pose every 3rd frame for performance

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
      video: { width: 1920, height: 1080, facingMode: 'user' },
    });
    video.srcObject = webcamStream;
    video.src = '';
    await new Promise(r => video.addEventListener('loadeddata', r, { once: true }));
    resizeAll();
    statusEl.textContent = 'Loading MediaPipe…';
    mpCamInstance = new window.Camera(video, {
      async onFrame() {
        await hands.send({ image: video });
        if (++_poseFrame % 3 === 0) await pose.send({ image: video });
      },
      width: 1920,
      height: 1080,
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
      if (++_poseFrame % 3 === 0) await pose.send({ image: video });
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
if (shoulderWidthInput) {
  shoulderWidthInput.value = SHOULDER_WIDTH_CM;
  shoulderWidthInput.addEventListener('input', () => {
    const v = parseFloat(shoulderWidthInput.value);
    if (v > 10 && v < 100) { SHOULDER_WIDTH_CM = v; lsSet('ht_shoulderWidth', v); }
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
      // If URL is empty, default to port 9090 for ROS, or 8765 for plain
      const defaultPort = robotSocket._mode === 'ros' ? 'ws://localhost:9090' : 'ws://localhost:8765';
      const url = wsUrlInput?.value.trim() || defaultPort;
      lsSet('ht_wsUrl', url);
      robotSocket.connect(url);
    }
  });
}
// --- 2. The "ROS" button acts purely as a format toggle ---
if (btnModeRos) {
  btnModeRos.addEventListener('click', () => {
    const isCurrentlyRos = robotSocket._mode === 'ros';
    const newMode = isCurrentlyRos ? 'raw' : 'ros';
    
    // Update the socket mode
    robotSocket.setMode(newMode);
    
    // Update the button UI to show it's active
    btnModeRos.classList.toggle('active', !isCurrentlyRos);
    btnModeRos.textContent = !isCurrentlyRos ? 'ROS ✓' : 'ROS';
  });
}
// ─── Start with webcam by default ────────────────────────────────────────────
startWebcam();