from fmpose3d import FMPose3DInference, FMPose3DConfig
import cv2
import numpy as np
import torch
import pyvista as pv
import base64
import threading
import time
import babyros 

# --- 1. Skeleton Definitions ---
SKELETON_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
    [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
    [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]
]

def show2Dpose(kps, img):
    lcolor = (255, 0, 0)  # Blue (OpenCV uses BGR)
    rcolor = (0, 0, 255)  # Red
    thickness = 3
    # Left/Right mask for the 16 connections
    LR = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0] 

    for j, c in enumerate(SKELETON_CONNECTIONS):
        start = tuple(map(int, kps[c[0]]))
        end = tuple(map(int, kps[c[1]]))
        color = lcolor if LR[j] else rcolor
        cv2.line(img, start, end, color, thickness)
    return img

# --- 2. Camera Access (BabyROS) ---
latest_image = None
image_lock = threading.Lock()

def image_callback(msg: dict):
    global latest_image
    try:
        img_bytes = base64.b64decode(msg['data'])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        with image_lock:
            latest_image = img
    except Exception as e:
        print(f"Error decoding image: {e}")

# --- 3. Inference Setup ---
config = FMPose3DConfig(model_type="fmpose3d_humans")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FMPose3DInference(model_cfg=config, device=device)

IMAGE_TOPIC = "camera_image" # Change if your topic name is different
print(f"Subscribing to {IMAGE_TOPIC}...")
sub_image = babyros.node.Subscriber(topic=IMAGE_TOPIC, callback=image_callback)

# --- 4. PyVista 3D Setup (BackgroundPlotter) ---
# Build the lines array for PyVista: [num_points, p1, p2]
body_lines = []
for conn in SKELETON_CONNECTIONS:
    body_lines.extend([2, conn[0], conn[1]])

# 👇 USE BackgroundPlotter to prevent OpenCV from blocking the 3D render loop
plotter = pv.BackgroundPlotter(window_size=(800, 800), title="3D FMPose3D")
plotter.set_background('#1e1e1e')

dummy_pose_points = np.zeros((17, 3))
pose_mesh = pv.PolyData(dummy_pose_points.copy(), lines=body_lines)
plotter.add_mesh(
    pose_mesh, 
    color='#00FFFF',       # Cyan
    line_width=5, 
    point_size=12, 
    render_points_as_spheres=True, 
    name='pose_body'
)

plotter.add_axes()
plotter.show_grid(color='#444444')
# Note: No plotter.show() needed! BackgroundPlotter opens the window automatically.

# --- 5. OpenCV 2D Setup ---
cv2.namedWindow("2D Pose", cv2.WINDOW_NORMAL)
print("Starting live FMPose3D tracking... Press 'q' in the 2D window to quit.")

try:
    while True:
        with image_lock:
            frame = latest_image
        
        if frame is None:
            time.sleep(0.01)
            continue
            
        # Run Inference
        result_2d = model.prepare_2d(frame)
        
        # Fallback if no person is detected
        if result_2d.keypoints.size == 0:
            cv2.imshow("2D Pose", frame)
            pose_mesh.points = dummy_pose_points
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue
            
        result_3d = model.pose_3d(result_2d.keypoints, result_2d.image_size)
        
        # === 2D Plot (OpenCV) ===
        kps_2d = np.asarray(result_2d.keypoints[0, 0])
        if kps_2d.shape[-1] == 3:
            kps_2d = kps_2d[:, :2]
            
        img_2d = show2Dpose(kps_2d, frame.copy())
        cv2.imshow("2D Pose", img_2d)
        
        # === 3D Plot (PyVista) ===
        kps_3d = np.asarray(result_3d.poses_3d[0, 0])
        
        # Update the mesh points directly (BackgroundPlotter handles the redraw)
        pose_mesh.points = kps_3d
        
        # Quit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    sub_image.delete()
    cv2.destroyAllWindows()
    plotter.close()