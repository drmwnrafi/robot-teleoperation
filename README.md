

# Hand & Body Tracker

Web app that takes a camera feed (or a video file), detects hands + upper body via
MediaPipe, and pushes a JSON frame over WebSocket every ~33 ms. Intended for
teleoperating a robot arm in real time.

## Demo

<video src="https://github.com/user-attachments/assets/14bf5437-a007-4f30-be92-aae054092e56" controls width="100%"></video>

## ROS Demo

<video src="https://github.com/user-attachments/assets/69ce3621-14c9-4a60-9063-0dc903a5dc13" controls width="100%"></video>

<video src="https://github.com/user-attachments/assets/dd417593-5118-4d7c-b3a1-64e6295da1a3" controls width="100%"></video>

## Requirements

- Node.js 18+
- A modern browser: Chrome / Edge / Firefox (latest)
- A webcam or a video file

## Running

```bash
npm install
npm run dev        # → http://localhost:5173
```

For a production build: `npm run build` (output in `dist/`).

## Calibration

Three fields at the top of the page. They persist in `localStorage`.

| Field | What it is | Typical values |
|-------|-----------|----------------|
| Hand size | Wrist-crease to base-of-middle-finger distance (cm) | 7–12 cm (adult ≈ 9) |
| Shoulder | Across-the-shoulder width (cm) | 35–50 cm |
| Cam FOV | Camera horizontal field-of-view (°) | USB webcam 60–70, laptop 65–75, phone back 70–80, phone front 85–100, ultra-wide 100–120 |

Quick FOV dial-in: hold a ruler exactly **30 cm** from the camera, flat to the
image plane. Watch the `Depth` reading. If it says 30, FOV is correct. If it says
60, double FOV. If it says 15, halve FOV.

## WebSocket

Enter the server URL in the `WS` field (default `ws://localhost:8765`), click
`Connect`. The dot turns green when connected. Auto-reconnect with exponential
backoff, up to 10 attempts.

### Message format

Text WebSocket, JSON, UTF-8. One frame every ~33 ms (≈30 fps). Browser sends only,
doesn't receive.

```jsonc
{
  "t":   1717361234567,   // unix-ms
  "seq": 12345,            // monotonic frame counter
  "fps": 29.4,             // current frame rate

  "frame": {               // coordinate system description — receiver needs this
    "origin": "camera",
    "x": "right",
    "y": "up",
    "z": "away_from_camera",
    "units": "cm"
  },

  "calibration": {         // what values are currently set
    "hand_size_cm": 9.0,
    "fov_deg": 88.0,
    "shoulder_width_cm": 42.0
  },

  "hands": [               // ALWAYS 2 slots: [Left, Right]
    {
      "label": "Left",
      "present": true,     // if false — no other fields
      "index": 0,
      "confidence": 0.94,  // 0..1 from MediaPipe

      "depth_cm": 87.4,    // wrist → camera distance

      "wrist_cm": { "x": 1.23, "y": -0.50, "z": 87.41 },

      "palm_normal": [0.01, -0.04, 0.99],  // unit vec ⊥ palm, points OUT of palm
      "finger_dir":  [0.05,  0.81, 0.58],  // unit vec wrist → middle-MCP

      "finger_curl":     [0.05, 0.10, 0.12, 0.60, 0.72],  // [thumb, idx, mid, ring, pinky], 0=straight 1=fully curled
      "pinch_cm":        [4.2, 7.1, 6.9, 6.5],             // tip→thumb-tip per finger
      "grip_aperture_cm": 4.2,                             // thumb↔index

      "gesture":     "GRAB",    // "GRAB" | "OPEN" | "POINT" | "PEACE" | null
      "gesture_id":  1,          // 0=none, 1=GRAB, 2=OPEN, 3=POINT, 4=PEACE
      "is_grab":     true
    },
    { "label": "Right", "present": false }
  ],

  "interhand_cm": 41.2,    // only when both hands are visible

  "body": {                // always 6 points when MediaPipe Pose sees the body
    "shoulder_L": [-12.4, 3.1, -120.5],
    "shoulder_R": [11.8, 2.9, -120.1],
    "elbow_L":    [-18.7, -9.2, -110.0],
    "elbow_R":    [16.4, -8.8, -109.5],
    "wrist_L":    [1.23, -0.50, -87.41],  // == hands[Left].wrist_cm (always equal)
    "wrist_R":    [-1.32, -1.46, -88.10]
  },

  "landmark_names": {
    "hand_joints": [
      "WRIST",
      "THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
      "INDEX_MCP","INDEX_PIP","INDEX_DIP","INDEX_TIP",
      "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
      "RING_MCP","RING_PIP","RING_DIP","RING_TIP",
      "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP"
    ],
    "body_joints": {
      "0": "nose",
      "11": "shoulder_L", "12": "shoulder_R",
      "13": "elbow_L",    "14": "elbow_R",
      "15": "wrist_L",    "16": "wrist_R",
      "23": "hip_L",      "24": "hip_R"
    }
  }
}
```

### Things worth knowing

1. Everything is in **centimetres** in camera frame. Multiply by 0.01 to get metres.
2. `body.wrist_L` and `hands[Left].wrist_cm` are the **same point**. The body wrist
   is overwritten with the hand wrist every frame so the arm bone meets the hand.
3. `label` is from the person's perspective, not the screen. In mirrored webcam
   mode, the "Left" hand appears on the right side of the screen — that's normal.
4. `hands` always has 2 elements in fixed order [Left, Right]. Missing hand =
   `{"label": "Right", "present": false}` with no other fields.
5. `seq` is monotonically increasing. Receiver detects drops: `received - expected`.
   `t` is wall-clock, not for ordering.
6. For the end-effector: `wrist_cm` (position) + `palm_normal` & `finger_dir`
   (orientation). Their cross = third axis → full gripper orientation.
7. For the gripper: `grip_aperture_cm` (0 = closed, 8–10 = open) or `is_grab` boolean.
8. For high-level commands: `gesture` (string) or `gesture_id` (0..4).

### If the robot is in a different coordinate frame

```python
# p_cam in camera frame, cm
p_cam = [hand.wrist_cm.x, hand.wrist_cm.y, hand.wrist_cm.z]

# T_cam_to_base — 4x4 matrix camera → robot base.
# Measure once (hand-eye calibration).
import numpy as np
T = np.array([...])  # fill in
p_base = T @ np.append(p_cam, 1.0)
```

For orientation: build a matrix from `palm_normal` (Z), `finger_dir` (Y), and their
cross (X), then multiply by T.

## Example: Python server

```bash
pip install websockets
```

```python
# server.py
import asyncio, json
import websockets

async def handler(ws):
    print(f"Connected: {ws.remote_address}")
    last_seq = -1
    async for raw in ws:
        data = json.loads(raw)
        if last_seq >= 0 and data["seq"] != last_seq + 1:
            print(f"!! dropped {last_seq+1}..{data['seq']-1}")
        last_seq = data["seq"]

        print(f"\n[seq={data['seq']} fps={data['fps']}]")
        for hand in data["hands"]:
            if not hand["present"]:
                print(f"  {hand['label']}: ABSENT")
                continue
            w = hand["wrist_cm"]
            print(f"  {hand['label']:5} conf={hand['confidence']:.2f}  "
                  f"wrist=({w['x']:6.2f}, {w['y']:6.2f}, {w['z']:6.2f})cm  "
                  f"grip={hand['grip_aperture_cm']}cm  gesture={hand.get('gesture')}")

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("ws://0.0.0.0:8765")
        await asyncio.Future()

asyncio.run(main())
```

Run: `python server.py`. In the browser, set `ws://localhost:8765` and click
`Connect`.

## Example: ROS 2

```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

In the browser: `ws://<robot-ip>:9090`. On the ROS side, subscribe to the WebSocket
and republish into a ROS topic.

## In-page API (no server needed)

```js
// Push event
window.addEventListener('hand-robot-data', (e) => {
  const { hands, body, frame, calibration } = e.detail;
});

// Or just the latest frame
const latest = window.__handRobotData;
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Depth 60 at real 30 cm | FOV too narrow | Double FOV |
| Depth 15 at real 30 cm | FOV too wide | Halve FOV |
| Body 130, hand 80 | Normal — hand is in front of body | Re-verify FOV with the 30 cm ruler |
| Hand "goes behind" body in 3D | MediaPipe briefly glitches during occlusion | Already handled (clamp to body depth) |
| Gesture flickers | Pose is borderline | Hold a clearer GRAB/OPEN/POINT/PEACE position |
| Depth jumps around | FOV or hand size wrong | Re-check 30 cm ruler + measure your hand |

## Structure

```
robot-teleoperation/
├── index.html
├── src/main.js
├── package.json
└── README.md
```
