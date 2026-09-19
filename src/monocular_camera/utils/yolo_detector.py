import cv2
import numpy as np
from ultralytics import YOLO

class YoloOnnxPoseDetector:
    def __init__(self, model_path='models/yolo26n-pose.onnx', conf_thres=0.5, device="0"):
        self.conf_thres = conf_thres
        self.device = device
        
        self.model = YOLO(model_path)
        
        self.last_xy = None
        self.last_box = None
        self.last_score = None
        self.last_kpts_conf = None

    def __call__(self, image):
        return self.detect(image)

    def detect(self, image):
        results = self.model(image, device=self.device, verbose=False)
        
        boxes = results[0].boxes
        keypoints = results[0].keypoints
        
        if len(boxes) == 0:
            self.last_xy = None
            self.last_box = None
            self.last_score = None
            self.last_kpts_conf = None
            return None, None, None

        scores = boxes.conf.cpu().numpy()
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score < self.conf_thres:
            self.last_xy = None
            self.last_box = None
            self.last_score = None
            self.last_kpts_conf = None
            return None, None, None

        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        
        xy = keypoints.xy[best_idx].cpu().numpy() 
        kpts_conf = keypoints.conf[best_idx].cpu().numpy()
        
        self.last_xy = xy
        self.last_box = (x1, y1, x2, y2)
        self.last_score = best_score
        self.last_kpts_conf = kpts_conf

        boxes_arr = np.array([[x1, y1, x2, y2]], dtype=np.int32)
        scores_arr = np.array([best_score], dtype=np.float32)

        nose = xy[0]
        l_shoulder = xy[5]
        r_shoulder = xy[6]
        l_elbow = xy[7]
        r_elbow = xy[8]
        l_wrist = xy[9]
        r_wrist = xy[10]
        l_hip = xy[11]
        r_hip = xy[12]
        l_knee = xy[13]
        r_knee = xy[14]
        l_ankle = xy[15]
        r_ankle = xy[16]
        
        pelvis = (l_hip + r_hip) * 0.5
        thorax = (l_shoulder + r_shoulder) * 0.5
        spine = (pelvis + thorax) * 0.5
        head = nose + np.array([0.0, -0.25 * np.linalg.norm(l_shoulder - r_shoulder)])

        kp17 = np.stack([
            pelvis,        # 0
            r_hip,         # 1
            r_knee,        # 2
            r_ankle,       # 3
            l_hip,         # 4
            l_knee,        # 5
            l_ankle,       # 6
            spine,         # 7
            thorax,        # 8
            nose,          # 9
            head,          # 10
            l_shoulder,    # 11
            l_elbow,       # 12
            l_wrist,       # 13
            r_shoulder,    # 14
            r_elbow,       # 15
            r_wrist        # 16
        ])

        return boxes_arr, scores_arr, kp17.astype(np.float32)

    def draw_pose(self, image):
        img = image.copy()
        if self.last_xy is None:
            return img
            
        if self.last_box is not None:
            x1, y1, x2, y2 = self.last_box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Pose: {self.last_score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw COCO skeleton
        COCO_SKELETON = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], 
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], 
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        COCO_SKELETON = [[i-1, j-1] for i, j in COCO_SKELETON]

        for i, pt in enumerate(self.last_xy):
            if self.last_kpts_conf is not None and self.last_kpts_conf[i] > 0.3:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
            
        # Draw skeleton lines
        for i, j in COCO_SKELETON:
            if self.last_kpts_conf is not None and self.last_kpts_conf[i] > 0.3 and self.last_kpts_conf[j] > 0.3:
                pt1 = (int(self.last_xy[i][0]), int(self.last_xy[i][1]))
                pt2 = (int(self.last_xy[j][0]), int(self.last_xy[j][1]))
                cv2.line(img, pt1, pt2, (255, 255, 255), 2)
            
        return img