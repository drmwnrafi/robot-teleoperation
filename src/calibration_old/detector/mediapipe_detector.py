import cv2
import mediapipe as mp

class MediaPipePoseDetector:
    def __init__(self, 
                 score_threshold=0.5, 
                 model_complexity=1, 
                 smooth_landmarks=True, 
                 enable_segmentation=False
                ):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=score_threshold,
            min_tracking_confidence=score_threshold,
        )

    def __call__(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb)

    def close(self):
        self.pose.close()


class MediaPipeHandDetector:
    def __init__(self, 
                 score_threshold=0.5, 
                 max_num_hands=2, 
                 model_complexity=1
                ):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=score_threshold,
            min_tracking_confidence=score_threshold,
        )

    def __call__(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def close(self):
        self.hands.close()


class MediaPipeHolisticDetector:
    def __init__(self, 
                 score_threshold=0.8, 
                 model_complexity=2, 
                 smooth_landmarks=True, 
                 refine_face_landmarks=False
                ):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=score_threshold,
            min_tracking_confidence=score_threshold,
        )

    def __call__(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.holistic.process(rgb)

    def close(self):
        self.holistic.close()