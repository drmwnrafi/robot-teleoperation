#!/usr/bin/env python3
import json
import csv
import math
import argparse
import asyncio
import sys
import time
import datetime

try:
    import websockets
except ImportError:
    websockets = None

class LandmarkProcessor:
    def __init__(self, fix_x=True, fix_y=True, fix_z=False):
        self.fix_x = fix_x
        self.fix_y = fix_y
        self.fix_z = fix_z
        self.alpha = 0.3 
        self.prev_wrist_pos = {
            'Left': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'Right': {'x': 0.0, 'y': 0.0, 'z': 0.0}
        }
        self.prev_joints = {
            'Left': [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(21)],
            'Right': [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(21)]
        }
        self.is_initialized = {'Left': False, 'Right': False}
        
        self.prev_body = {}
        self.is_body_initialized = False

    @staticmethod
    def _js_to_ros(x, y, z):
        """Transforms MediaPipe/JS coordinates to ROS coordinates."""
        return float(z), -float(x), float(y)
    
    def get_distance(self, p1, p2):
        return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)

    def _smooth_and_format_hand(self, hand, side_label, shoulder_mid):
        wx, wy, wz = hand["wrist_cm"]["x"], hand["wrist_cm"]["y"], hand["wrist_cm"]["z"]
        ros_wx, ros_wy, ros_wz = self._js_to_ros(wx, wy, wz)

        ros_wx -= shoulder_mid[0]
        ros_wy -= shoulder_mid[1]
        ros_wz -= shoulder_mid[2]

        if not self.is_initialized[side_label]:
            self.prev_wrist_pos[side_label] = {'x': ros_wx, 'y': ros_wy, 'z': ros_wz}
            
            joints = hand.get("joints_cm", [])
            if len(joints) == 21:
                for i in range(21):
                    jx, jy, jz = joints[i]['x'], joints[i]['y'], joints[i]['z']
                    rjx, rjy, rjz = self._js_to_ros(jx, jy, jz)
                    
                    rjx -= shoulder_mid[0]
                    rjy -= shoulder_mid[1]
                    rjz -= shoulder_mid[2]
                    
                    self.prev_joints[side_label][i] = {'x': rjx, 'y': rjy, 'z': rjz}
            self.is_initialized[side_label] = True
        
        self.prev_wrist_pos[side_label]['x'] = self.alpha * ros_wx + (1 - self.alpha) * self.prev_wrist_pos[side_label]['x']
        self.prev_wrist_pos[side_label]['y'] = self.alpha * ros_wy + (1 - self.alpha) * self.prev_wrist_pos[side_label]['y']
        self.prev_wrist_pos[side_label]['z'] = self.alpha * ros_wz + (1 - self.alpha) * self.prev_wrist_pos[side_label]['z']

        smoothed_wrist_m = {
            'x': round(self.prev_wrist_pos[side_label]['x'] / 100.0, 4),
            'y': round(self.prev_wrist_pos[side_label]['y'] / 100.0, 4),
            'z': round(self.prev_wrist_pos[side_label]['z'] / 100.0, 4)
        }

        # Apply EMA to joints
        joints = hand.get("joints_cm", [])
        smoothed_joints_m = []
        if len(joints) == 21:
            for i in range(21):
                jx, jy, jz = joints[i]['x'], joints[i]['y'], joints[i]['z']
                rjx, rjy, rjz = self._js_to_ros(jx, jy, jz)
                
                rjx -= shoulder_mid[0]
                rjy -= shoulder_mid[1]
                rjz -= shoulder_mid[2]
                
                pj = self.prev_joints[side_label][i]
                pj['x'] = self.alpha * rjx + (1 - self.alpha) * pj['x']
                pj['y'] = self.alpha * rjy + (1 - self.alpha) * pj['y']
                pj['z'] = self.alpha * rjz + (1 - self.alpha) * pj['z']
                
                smoothed_joints_m.append({
                    'x': round(pj['x'] / 100.0, 4),
                    'y': round(pj['y'] / 100.0, 4),
                    'z': round(pj['z'] / 100.0, 4)
                })

        msg = {
            'present': True,
            'confidence': round(hand.get("confidence", 0.0), 3),
            'depth_m': round(hand.get("depth_cm", 0.0) / 100.0, 4),
            'wrist_m': smoothed_wrist_m,
            'joints_m': smoothed_joints_m
        }
        
        if "palm_normal" in hand and isinstance(hand["palm_normal"], list) and len(hand["palm_normal"]) == 3:
            pn = hand["palm_normal"]
            msg['palm_normal'] = list(self._js_to_ros(pn[0], pn[1], pn[2]))
        else:
            msg['palm_normal'] = hand.get("palm_normal", [])
            
        if "finger_dir" in hand and isinstance(hand["finger_dir"], list) and len(hand["finger_dir"]) == 3:
            fd = hand["finger_dir"]
            msg['finger_dir'] = list(self._js_to_ros(fd[0], fd[1], fd[2]))
        else:
            msg['finger_dir'] = hand.get("finger_dir", [])
            
        msg['finger_curl'] = hand.get("finger_curl", [])
        msg['pinch_m'] = [round(p / 100.0, 4) for p in hand.get("pinch_cm", [0.0, 0.0, 0.0, 0.0])]
        msg['grip_aperture_m'] = round(hand.get("grip_aperture_cm", 0.0) / 100.0, 4)
        msg['gesture'] = str(hand.get("gesture") or "UNKNOWN")
        msg['gesture_id'] = int(hand.get("gesture_id", 0))
        msg['is_grab'] = bool(hand.get("is_grab", False))
        
        return msg

    def process(self, raw_json_str):
        """Processes a single JSON string and returns the processed dictionary."""
        try:
            raw_data = json.loads(raw_json_str)
        except json.JSONDecodeError:
            return None

        present_hands = [h for h in raw_data.get("hands", []) if h.get("present")]
        assigned_left = None
        assigned_right = None

        shoulder_mid = [0.0, 0.0, 0.0]
        raw_body = raw_data.get("body", {})
        
        ls_key = rs_key = None
        for k in raw_body.keys():
            k_lower = k.lower()
            if "shoulder" in k_lower:
                if "left" in k_lower or k_lower.endswith("_l"): ls_key = k
                elif "right" in k_lower or k_lower.endswith("_r"): rs_key = k
        
        if ls_key and rs_key:
            ls = raw_body[ls_key]
            rs = raw_body[rs_key]
            if isinstance(ls, list) and len(ls) == 3 and isinstance(rs, list) and len(rs) == 3:
                rls_x, rls_y, rls_z = self._js_to_ros(ls[0], ls[1], ls[2])
                rrs_x, rrs_y, rrs_z = self._js_to_ros(rs[0], rs[1], rs[2])
                
                shoulder_mid = [
                    (rls_x + rrs_x) / 2.0 if self.fix_x else 0.0,
                    (rls_y + rrs_y) / 2.0 if self.fix_y else 0.0,
                    (rls_z + rrs_z) / 2.0 if self.fix_z else 0.0
                ]

        def get_ros_pos(h):
            wx, wy, wz = h["wrist_cm"]["x"], h["wrist_cm"]["y"], h["wrist_cm"]["z"]
            rx, ry, rz = self._js_to_ros(wx, wy, wz)
            rx -= shoulder_mid[0]; ry -= shoulder_mid[1]; rz -= shoulder_mid[2]
            return {'x': rx, 'y': ry, 'z': rz}, wx

        if len(present_hands) == 1:
            hand = present_hands[0]
            pos_ros, js_x = get_ros_pos(hand)
            
            dist_left = self.get_distance(pos_ros, self.prev_wrist_pos["Left"]) if self.is_initialized["Left"] else float('inf')
            dist_right = self.get_distance(pos_ros, self.prev_wrist_pos["Right"]) if self.is_initialized["Right"] else float('inf')
            
            if not self.is_initialized["Left"] and not self.is_initialized["Right"]:
                if js_x < 0:
                    assigned_left = hand
                    self.is_initialized["Left"] = True
                else:
                    assigned_right = hand
                    self.is_initialized["Right"] = True
            elif dist_left < dist_right:
                assigned_left = hand
            else:
                assigned_right = hand

        elif len(present_hands) == 2:
            h1, h2 = present_hands[0], present_hands[1]
            p1_ros, js_x1 = get_ros_pos(h1)
            p2_ros, js_x2 = get_ros_pos(h2)
            
            if not self.is_initialized["Left"] and not self.is_initialized["Right"]:
                if js_x1 < js_x2:
                    assigned_left, assigned_right = h1, h2
                else:
                    assigned_left, assigned_right = h2, h1
                self.is_initialized["Left"] = True
                self.is_initialized["Right"] = True
            else:
                d1L = self.get_distance(p1_ros, self.prev_wrist_pos["Left"]) if self.is_initialized["Left"] else float('inf')
                d1R = self.get_distance(p1_ros, self.prev_wrist_pos["Right"]) if self.is_initialized["Right"] else float('inf')
                d2L = self.get_distance(p2_ros, self.prev_wrist_pos["Left"]) if self.is_initialized["Left"] else float('inf')
                d2R = self.get_distance(p2_ros, self.prev_wrist_pos["Right"]) if self.is_initialized["Right"] else float('inf')
                
                if (d1L + d2R) < (d1R + d2L):
                    assigned_left, assigned_right = h1, h2
                else:
                    assigned_left, assigned_right = h2, h1

        smoothed_body_msgs = []
        if not self.is_body_initialized and raw_body:
            for joint, coords in raw_body.items():
                if isinstance(coords, list) and len(coords) == 3:
                    rx, ry, rz = self._js_to_ros(coords[0], coords[1], coords[2])
                    self.prev_body[joint] = [rx - shoulder_mid[0], ry - shoulder_mid[1], rz - shoulder_mid[2]]
            self.is_body_initialized = True
            
        for joint, coords in raw_body.items():
            if isinstance(coords, list) and len(coords) == 3:
                if joint not in self.prev_body:
                    rx, ry, rz = self._js_to_ros(coords[0], coords[1], coords[2])
                    self.prev_body[joint] = [rx - shoulder_mid[0], ry - shoulder_mid[1], rz - shoulder_mid[2]]
                
                nx, ny, nz = self._js_to_ros(coords[0], coords[1], coords[2])
                nx -= shoulder_mid[0]; ny -= shoulder_mid[1]; nz -= shoulder_mid[2]
                
                px, py, pz = self.prev_body[joint]
                sx = self.alpha * nx + (1 - self.alpha) * px
                sy = self.alpha * ny + (1 - self.alpha) * py
                sz = self.alpha * nz + (1 - self.alpha) * pz
                
                self.prev_body[joint] = [sx, sy, sz]
                smoothed_body_msgs.append({
                    'joint_name': joint,
                    'x': round(sx / 100.0, 4),
                    'y': round(sy / 100.0, 4),
                    'z': round(sz / 100.0, 4)
                })

        out_msg = {
            't': float(raw_data.get("t", 0.0)),
            'seq': int(raw_data.get("seq", 0)),
            'fps': float(raw_data.get("fps", 0.0)),
            'frame_info': raw_data.get("frame", {}),
            'calibration_info': raw_data.get("calibration", {}),
            'landmark_names': list(raw_data.get("landmark_names", {}).keys()),
            'body_landmarks': smoothed_body_msgs
        }

        if assigned_left:
            out_msg['left_hand'] = self._smooth_and_format_hand(assigned_left, "Left", shoulder_mid)
        else:
            out_msg['left_hand'] = {'present': False}
            
        if assigned_right:
            out_msg['right_hand'] = self._smooth_and_format_hand(assigned_right, "Right", shoulder_mid)
        else:
            out_msg['right_hand'] = {'present': False}

        return out_msg

connected_clients = set()

async def ws_handler(ws, processor, path=None, raw_file=None, proc_file=None, raw_writer=None, proc_writer=None):
    print(f"[WS] Connected: {ws.remote_address}")
    connected_clients.add(ws)
    last_seq = -1
    
    try:
        async for raw in ws:
            try:
                ts = time.time()
                if raw_writer:
                    raw_writer.writerow([ts, raw])
                    raw_file.flush()
                processed_data = processor.process(raw)
                if processed_data:
                    if proc_writer:
                        left_present = processed_data['left_hand'].get('present', False)
                        right_present = processed_data['right_hand'].get('present', False)
                        proc_writer.writerow([ts, left_present, right_present, json.dumps(processed_data)])
                        proc_file.flush()
                    seq = processed_data.get("seq", -1)
                    if last_seq >= 0 and seq != last_seq + 1:
                        print(f"!! dropped {last_seq+1}..{seq-1}", file=sys.stderr)
                    last_seq = seq
                    if connected_clients:
                        msg = json.dumps(processed_data)
                        await asyncio.gather(
                            *[client.send(msg) for client in connected_clients],
                            return_exceptions=True 
                        )
            except Exception as e:
                print(f"[WS] Error processing message: {e}", file=sys.stderr)
    finally:
        connected_clients.remove(ws)
        print(f"[WS] Disconnected: {ws.remote_address}")

async def run_websocket(processor, host, port, record=False):
    if websockets is None:
        print("Error: 'websockets' library is not installed. Please run: pip install websockets")
        return
        
    raw_file = proc_file = None
    raw_writer = proc_writer = None
    
    if record:
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = f"raw_landmarks_{ts_str}.csv"
        proc_path = f"processed_landmarks_{ts_str}.csv"
        
        raw_file = open(raw_path, 'w', newline='')
        proc_file = open(proc_path, 'w', newline='')
        
        raw_writer = csv.writer(raw_file)
        proc_writer = csv.writer(proc_file)
        
        raw_writer.writerow(['timestamp_sec', 'raw_json'])
        proc_writer.writerow(['timestamp_sec', 'left_hand_present', 'right_hand_present', 'processed_json'])
        
        print(f"[REC] Recording to {raw_path} and {proc_path}")
        
    print(f"[WS] Starting broadcast server on ws://{host}:{port}")
    
    async def handler(*args):
        ws = args[0]
        await ws_handler(ws, processor, raw_file=raw_file, proc_file=proc_file, 
                         raw_writer=raw_writer, proc_writer=proc_writer)
        
    try:
        async with websockets.serve(handler, host, port):
            await asyncio.Future()  # run forever
    finally:
        if raw_file: raw_file.close()
        if proc_file: proc_file.close()
        if record:
            print("[REC] Recording files closed.")

def run_csv(processor, input_csv, output_csv):
    print(f"[CSV] Reading from {input_csv} and saving to {output_csv}...")
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['timestamp_sec', 'left_hand_present', 'right_hand_present', 'processed_json'])
        
        processed_count = 0
        for row in reader:
            if len(row) < 2: continue
            try:
                ts = float(row[0])
            except ValueError:
                continue
                
            json_str = row[1]
            processed_data = processor.process(json_str)
            
            if processed_data:
                left_present = processed_data['left_hand'].get('present', False)
                right_present = processed_data['right_hand'].get('present', False)
                writer.writerow([ts, left_present, right_present, json.dumps(processed_data)])
                processed_count += 1
                
    print(f"[CSV] Done! Successfully processed {processed_count} frames.")

def main():
    parser = argparse.ArgumentParser(description="Landmark Processor (CSV or WebSocket)")
    parser.add_argument("--mode", choices=["csv", "ws"], default="csv", 
                        help="Operation mode: 'csv' to process a file, 'ws' to run a WebSocket server")
    
    parser.add_argument("--input", default="raw_landmarks.csv", help="Input CSV file (for csv mode)")
    parser.add_argument("--output", default="processed_landmarks.csv", help="Output CSV file (for csv mode)")
    
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket host (for ws mode)")
    parser.add_argument("--port", type=int, default=9090, help="WebSocket port (for ws mode)")
    
    parser.add_argument("--record", action="store_true", help="Record raw and processed data to CSV files while running in ws mode")
    
    args = parser.parse_args()
    
    processor = LandmarkProcessor(fix_x=True, fix_y=True, fix_z=False)
    
    if args.mode == "csv":
        run_csv(processor, args.input, args.output)
    elif args.mode == "ws":
        try:
            asyncio.run(run_websocket(processor, args.host, args.port, record=args.record))
        except KeyboardInterrupt:
            print("\n[WS] Shutting down server.")

if __name__ == '__main__':
    main()