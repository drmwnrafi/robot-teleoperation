# 3D Hand Tracker — Robot Teleoperation

Real-time hand tracking in the browser that streams 3D joint coordinates to a robot via WebSocket. Uses [MediaPipe Hands](https://mediapipe.dev) for landmark detection, [Three.js](https://threejs.org) for 3D visualization, and a built-in WebSocket client for robot integration.

```
Camera → MediaPipe (21 landmarks) → Gesture detection → WebSocket → Your robot
                                   ↘ Three.js 3D view
```

---

## Features

| Feature | Description |
|---------|-------------|
| 21 joint 3D tracking | Full hand skeleton in centimetres (camera frame) |
| Depth estimation | Pinhole-model depth from apparent hand size |
| Gesture detection | **GRAB**, **OPEN**, **POINT**, **PEACE** |
| WebSocket output | ~30 fps JSON frames to any server |
| Video file mode | Test with pre-recorded `.mp4` / `.webm` files |
| Calibration | Hand size + camera FOV, saved to `localStorage` |

---

## Requirements

- **Node.js 18+** ([download for Windows](https://nodejs.org/en/download))  
- A modern browser: Chrome 90+, Edge 90+, Firefox 90+  
- A webcam (or a video file for offline testing)  
- **WSL is NOT required** — runs 100% on native Windows

---

## Installation (Windows, no WSL)

### 1. Install Node.js

Download and install from https://nodejs.org (LTS version).  
After installing, open **PowerShell** or **Command Prompt** and verify:

```powershell
node -v   # should print v18.x.x or higher
npm -v    # should print 9.x.x or higher
```

### 2. Clone and install dependencies

```powershell
git clone https://github.com/obotx/robot-teleoperation.git
cd robot-teleoperation
npm install
```

### 3. Start the development server

```powershell
npm run dev
```

You will see output like:

```
  VITE v5.x  ready in xxx ms
  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.x.x:5173/
```

### 4. Open the app

Open **http://localhost:5173** in your browser.  
Allow camera access when prompted.

---

## Running without a dev server (production build)

```powershell
npm run build    # creates dist/ folder
npm run preview  # serves dist/ locally
```

Or deploy the `dist/` folder to any static hosting (GitHub Pages, Netlify, etc.).

---

## Usage Guide

### Camera mode (default)
- App starts with your webcam automatically
- Hold your hand in front of the camera, palm facing the lens
- The left panel shows the 2D overlay; the right panel shows the 3D skeleton

### Video file mode
- Click **📁 Video** to load a `.mp4` or `.webm` file
- Use the playback bar to pause / seek / change speed

### Calibration
| Field | What it does | Default |
|-------|-------------|---------|
| Hand size (cm) | Physical wrist→middle-MCP distance | 9.0 cm (adult average) |
| Cam FOV (°) | Camera horizontal field of view | 62° (typical USB webcam) |

Settings are saved automatically to `localStorage`.

### Gestures
| Gesture | Description | Colour |
|---------|-------------|--------|
| **GRAB** | Fist (all 4 fingers curled) | Red |
| **OPEN** | All 4 fingers extended | Green |
| **POINT** | Only index extended | Blue |
| **PEACE** | Index + middle extended (V sign) | Yellow |

### WebSocket connection
1. Enter your server URL in the **WS:** field (default `ws://localhost:8765`)
2. Click **Connect**
3. The dot turns **green** when connected, **yellow** while connecting, **red** on error
4. Auto-reconnects with exponential back-off (up to 10 attempts)

---

## WebSocket Data Format

Every ~33 ms (≈30 fps) the app sends a JSON frame:

```json
{
  "t": 1748448000000,
  "depth_cm": 45.2,
  "wrist_cm": { "x": 1.23, "y": -0.50, "z": 45.20 },
  "joints_cm": [
    [x0, y0, z0],
    [x1, y1, z1],
    "...21 joints total..."
  ],
  "gesture": "GRAB",
  "is_grab": true
}
```

**Coordinate system** (camera frame):
- `x` — right is positive  
- `y` — up is positive  
- `z` — depth away from lens is positive  
- Joint order follows [MediaPipe landmark indices](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models)

---

## Example: Python WebSocket server

Minimal server that receives hand data (no WSL needed — runs on Windows Python):

```python
# pip install websockets
import asyncio, json
import websockets

async def handler(websocket):
    print("Client connected")
    async for message in websocket:
        data = json.loads(message)
        wrist = data["wrist_cm"]
        gesture = data.get("gesture")
        print(f"wrist=({wrist['x']:.1f}, {wrist['y']:.1f}, {wrist['z']:.1f}) cm  gesture={gesture}")

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("WebSocket server on ws://localhost:8765")
        await asyncio.Future()  # run forever

asyncio.run(main())
```

Run it with:
```powershell
pip install websockets
python server.py
```

---

## Example: ROS 2 bridge

If you use ROS 2 on a separate machine, use `rosbridge_server`:

```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

Then set the WS URL in the browser to `ws://<robot-ip>:9090`.

The JSON frame can be republished as a custom ROS message or decoded inline in a `rosbridge` subscriber.

---

## Possible Problems & Solutions

### Camera not starting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Camera error: Permission denied" | Browser blocked camera access | Click the camera icon in the address bar → Allow |
| "Camera error: Requested device not found" | No webcam connected | Plug in a webcam or use Video File mode |
| Black video, no tracking | Camera used by another app | Close Zoom, Teams, OBS, etc. |
| Camera works in Chrome but not Edge | Old Edge version | Update Edge / use Chrome |

### MediaPipe not loading

| Symptom | Cause | Fix |
|---------|-------|-----|
| Status stuck at "Loading MediaPipe…" | CDN blocked / no internet | Check firewall; try VPN; or self-host the `.wasm` files |
| Console: "Hands is not a constructor" | Script load order issue | Reload the page; check CDN availability |
| Very slow first load | WASM files downloading (~10 MB) | Normal on first visit; cached afterwards |

### Vite / npm issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `npm install` fails with EACCES | Permission error | Run PowerShell as Administrator |
| `'vite' is not recognized` | node_modules not installed | Run `npm install` first |
| Port 5173 already in use | Another Vite instance running | Run `npm run dev -- --port 5174` |
| `npm run dev` hangs | Antivirus blocking Node.js | Add project folder to AV exclusions |
| Module not found: three | dependencies not installed | Delete `node_modules/` and run `npm install` again |

### WebSocket

| Symptom | Cause | Fix |
|---------|-------|-----|
| Dot stays yellow forever | Server not running | Start your Python/ROS server first |
| "Error — retrying" after 10 attempts | Server URL wrong or server crashed | Check URL, restart server, click Connect again |
| HTTPS site → WS blocked | Browser blocks `ws://` on HTTPS | Use `wss://` with a TLS server, or run app on `http://` |

### Tracking quality

| Symptom | Cause | Fix |
|---------|-------|-----|
| Depth value jumps | Incorrect hand size calibration | Measure your wrist→middle-MCP distance and enter it |
| Skeleton flickers | Poor lighting | Add a light source in front of you |
| Gestures not recognized | Ambiguous hand pose | Move to a clear GRAB / OPEN / POINT / PEACE position |
| Tracking lost frequently | Background clutter | Use a plain background or better lighting |

---

## Project Structure

```
robot-teleoperation/
├── index.html        # UI layout and styles
├── src/
│   └── main.js       # Three.js, MediaPipe, gestures, WebSocket
├── package.json      # Vite + three.js dependencies
└── README.md
```

---

## Browser API (no server needed)

If you prefer to consume data within the same browser tab:

```js
window.addEventListener('hand-robot-data', (e) => {
  const { wrist_cm, gesture, joints_cm } = e.detail;
  // drive your in-page robot simulation here
});

// Or poll the latest frame:
const latest = window.__handRobotData;
```

---

## License

MIT
