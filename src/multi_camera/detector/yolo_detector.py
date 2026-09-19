import cv2
from ultralytics import YOLO

class YOLOPoseDetector:
    def __init__(self, model_path="yolo26n-pose.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def __call__(self, image):
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        return results

    def close(self):
        pass