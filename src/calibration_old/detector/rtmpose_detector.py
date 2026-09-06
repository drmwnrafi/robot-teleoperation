import cv2
import numpy as np
import onnxruntime as ort


class RTMPoseWholebodyDetector:
    COCO_SKELETON = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
        (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4), (3, 5), (4, 6)
    ]

    def __init__(
        self,
        model_path="rtmpose_wholebody_m_1x3x256x192_16_with_post.onnx",
        device="cpu",
        score_threshold=0.3,
        input_size=None,
        return_body17=False,
        bgr_to_rgb=False,
    ):
        self.score_threshold = score_threshold
        self.return_body17 = return_body17
        self.bgr_to_rgb = bgr_to_rgb
        self.sess = self._build_session(model_path, device)

        if input_size is not None:
            self.input_size = tuple(input_size)
        else:
            self.input_size = self._get_model_input_size(fallback=(192, 256))

        self.model_has_post = len(self.sess.get_inputs()) > 1
        self.last_xy = None
        self.last_keypoints = None
        self.last_box = None
        self.last_score = None
        self.last_kpts_conf = None

    def __call__(self, image):
        return self.detect(image)

    def detect(self, image):
        if image is None or image.size == 0:
            self._reset()
            return None, None, None

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w = image.shape[:2]
        resized_img, warp_mat = self._preprocess(image)
        outputs = self._inference(resized_img, image)
        keypoints = self._parse_outputs(outputs)

        if keypoints is None or keypoints.size == 0 or keypoints.shape[0] == 0:
            self._reset()
            return None, None, None

        person_scores = self._get_person_scores(keypoints)
        best_idx = int(np.argmax(person_scores))
        best_score = float(person_scores[best_idx])

        if best_score < self.score_threshold:
            self._reset()
            return None, None, None

        kpt = keypoints[best_idx].astype(np.float32)

        xy = kpt[:, :2]
        conf = kpt[:, 2]

        if not self.model_has_post:
            xy = self._inverse_transform_keypoints(xy, warp_mat)

        xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)

        self.last_xy = xy.astype(np.float32)
        self.last_kpts_conf = conf.astype(np.float32)
        self.last_keypoints = np.concatenate(
            [self.last_xy, self.last_kpts_conf[:, None]],
            axis=1
        ).astype(np.float32)

        self.last_score = best_score
        self.last_box = self._keypoints_to_box(
            xy=self.last_xy,
            conf=self.last_kpts_conf,
            img_w=w,
            img_h=h
        )

        boxes_arr = np.array([self.last_box], dtype=np.int32)
        scores_arr = np.array([best_score], dtype=np.float32)

        if self.return_body17:
            kp17 = self.get_last_kp17()
            return boxes_arr, scores_arr, kp17

        return boxes_arr, scores_arr, self.last_keypoints

    def draw_pose(self, image):
        img = image.copy()

        if self.last_xy is None:
            return img

        if self.last_box is not None:
            x1, y1, x2, y2 = self.last_box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if self.last_score is not None:
                cv2.putText(
                    img,
                    f"Pose: {self.last_score:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        xy = self.last_xy

        if xy.size == 0:
            return img

        if (
            self.last_kpts_conf is not None
            and len(self.last_kpts_conf) == len(xy)
        ):
            conf = self.last_kpts_conf
        else:
            conf = np.ones(len(xy), dtype=np.float32)

        skeleton = globals().get("skeleton", None)
        palette = globals().get("palette", None)
        link_color = globals().get("link_color", None)
        point_color = globals().get("point_color", None)

        if skeleton is None or palette is None or link_color is None or point_color is None:
            skeleton = self.COCO_SKELETON if xy.shape[0] >= 17 else []
            palette = [
                (0, 255, 255),
                (0, 255, 0),
                (255, 128, 0),
                (255, 255, 255),
                (255, 153, 255),
                (102, 178, 255),
                (255, 51, 51),
            ]
            link_color = [0] * len(skeleton)
            point_color = [0] * xy.shape[0]

        if palette is None or len(palette) == 0:
            palette = [(0, 255, 0)]

        if len(point_color) != xy.shape[0]:
            point_color = [0] * xy.shape[0]

        if len(link_color) != len(skeleton):
            link_color = [0] * len(skeleton)

        for (u, v), color_idx in zip(skeleton, link_color):
            if u >= xy.shape[0] or v >= xy.shape[0]:
                continue

            if (
                conf[u] > self.score_threshold
                and conf[v] > self.score_threshold
            ):
                color = palette[int(color_idx) % len(palette)]
                pt1 = (int(xy[u][0]), int(xy[u][1]))
                pt2 = (int(xy[v][0]), int(xy[v][1]))

                cv2.line(
                    img,
                    pt1,
                    pt2,
                    color,
                    2,
                    cv2.LINE_AA
                )

        for i, (pt, color_idx) in enumerate(zip(xy, point_color)):
            if conf[i] > self.score_threshold:
                color = palette[int(color_idx) % len(palette)]

                cv2.circle(
                    img,
                    (int(pt[0]), int(pt[1])),
                    3,
                    color,
                    -1,
                    cv2.LINE_AA
                )

        return img

    def get_last_kp17(self):
        if self.last_xy is None:
            return None

        body17 = self.last_xy[:17]

        if body17.shape[0] < 17:
            return body17.astype(np.float32)

        return self._convert_coco17_to_custom_kp17(body17)

    def _reset(self):
        self.last_xy = None
        self.last_keypoints = None
        self.last_box = None
        self.last_score = None
        self.last_kpts_conf = None

    def _build_session(self, model_path, device):
        device = str(device).lower()

        if device in ["cpu", "cpus"]:
            providers = ["CPUExecutionProvider"]
        elif device.lower() == 'cuda':
            providers = ["CUDAExecutionProvider"]
        else:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

        sess = ort.InferenceSession(
            path_or_bytes=model_path,
            providers=providers
        )

        return sess

    def _get_model_input_size(self, fallback=(192, 256)):
        input_shape = self.sess.get_inputs()[0].shape

        if len(input_shape) == 4:
            try:
                h = int(input_shape[2])
                w = int(input_shape[3])

                if h > 0 and w > 0:
                    return w, h
            except Exception:
                pass

        return fallback

    def _preprocess(self, image):
        img = image

        if self.bgr_to_rgb and img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        img_wh = np.asarray([w, h], dtype=np.float32)

        center = img_wh * 0.5
        scale = img_wh * 1.25

        resized_img, warp_mat = self._top_down_affine(
            input_size=self.input_size,
            bbox_scale=scale,
            bbox_center=center,
            img=img
        )

        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        resized_img = (resized_img - mean) / std
        resized_img = resized_img.astype(np.float32)

        return resized_img, warp_mat

    def _top_down_affine(self, input_size, bbox_scale, bbox_center, img):
        w, h = input_size
        warp_size = (int(w), int(h))

        bbox_w = float(bbox_scale[0])
        bbox_h = float(bbox_scale[1])

        aspect = float(w) / float(h)

        w_scaled = max(bbox_h * aspect, bbox_w)
        h_scaled = max(bbox_w / aspect, bbox_h)

        bbox_scale = np.array([w_scaled, h_scaled], dtype=np.float32)

        warp_mat = self._get_warp_matrix(
            center=bbox_center,
            scale=bbox_scale,
            rot=0,
            output_size=(w, h),
            inv=False
        )

        resized_img = cv2.warpAffine(
            img,
            warp_mat,
            warp_size,
            flags=cv2.INTER_LINEAR
        )

        return resized_img, warp_mat

    def _get_warp_matrix(
        self,
        center,
        scale,
        rot,
        output_size,
        shift=(0.0, 0.0),
        inv=False
    ):
        shift = np.array(shift, dtype=np.float32)

        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.deg2rad(rot)

        src_dir = self._rotate_point(
            np.array([0.0, src_w * -0.5], dtype=np.float32),
            rot_rad
        )

        dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)

        src_dim0 = center + scale * shift
        src_dim1 = center + src_dir + scale * shift

        src = np.concatenate(
            [
                [src_dim0],
                [src_dim1],
                [self._get_3rd_point(src_dim0, src_dim1)],
            ],
            axis=0,
            dtype=np.float32
        )

        dst_dim0 = np.asarray(
            [dst_w * 0.5, dst_h * 0.5],
            dtype=np.float32
        )

        dst_dim1 = dst_dim0 + dst_dir

        dst = np.concatenate(
            [
                [dst_dim0],
                [dst_dim1],
                [self._get_3rd_point(dst_dim0, dst_dim1)],
            ],
            axis=0,
            dtype=np.float32
        )

        if inv:
            warp_mat = cv2.getAffineTransform(dst, src)
        else:
            warp_mat = cv2.getAffineTransform(src, dst)

        return warp_mat.astype(np.float32)

    def _rotate_point(self, pt, angle_rad):
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)

        rot_mat = np.array(
            [[cs, -sn],
             [sn, cs]],
            dtype=np.float32
        )

        return rot_mat @ pt

    def _get_3rd_point(self, a, b):
        direction = a - b
        c = b + np.r_[-direction[1], direction[0]]
        return c

    def _inference(self, resized_img, image):
        input_tensor = np.ascontiguousarray(
            resized_img.transpose(2, 0, 1)[np.newaxis, ...],
            dtype=np.float32
        )

        feed = {
            self.sess.get_inputs()[0].name: input_tensor
        }

        if self.model_has_post:
            size_input = self.sess.get_inputs()[1]

            dtype = np.int64
            if "int32" in size_input.type:
                dtype = np.int32
            elif "float" in size_input.type:
                dtype = np.float32

            img_size = np.array(
                [[image.shape[1], image.shape[0]]],
                dtype=dtype
            )

            if len(size_input.shape) == 1:
                img_size = img_size.reshape(-1)

            feed[size_input.name] = img_size

        outputs = self.sess.run(None, feed)

        return outputs

    def _parse_outputs(self, outputs):
        if outputs is None or len(outputs) == 0:
            return None

        for out in outputs:
            arr = np.asarray(out, dtype=np.float32)

            if arr.ndim == 2 and arr.shape[-1] == 3:
                return arr[None]

            if arr.ndim == 3 and arr.shape[-1] == 3:
                return arr

        if len(outputs) >= 2:
            kpts = np.asarray(outputs[0], dtype=np.float32)
            scores = np.asarray(outputs[1], dtype=np.float32)

            if kpts.ndim == 2:
                kpts = kpts[None]

            if scores.ndim == 2:
                scores = scores[..., None]

            if (
                kpts.ndim == 3
                and kpts.shape[-1] == 2
                and scores.ndim == 3
                and scores.shape[-1] == 1
            ):
                return np.concatenate([kpts, scores], axis=-1).astype(np.float32)

        arr = np.asarray(outputs[0], dtype=np.float32)

        if arr.ndim == 2:
            arr = arr[None]

        if arr.ndim == 3:
            if arr.shape[-1] == 2:
                score = np.ones(arr.shape[:-1] + (1,), dtype=np.float32)
                arr = np.concatenate([arr, score], axis=-1)

            elif arr.shape[-1] == 3:
                return arr

        return arr

    def _get_person_scores(self, keypoints):
        if keypoints.ndim != 3 or keypoints.shape[0] == 0:
            return np.array([0.0], dtype=np.float32)

        scores = keypoints[..., 2]
        person_scores = []

        for s in scores:
            if s.size == 0:
                person_scores.append(0.0)
                continue

            valid = s > 0.1

            if np.any(valid):
                person_scores.append(float(np.mean(s[valid])))
            else:
                person_scores.append(float(np.max(s)))

        return np.array(person_scores, dtype=np.float32)

    def _inverse_transform_keypoints(self, xy, warp_mat):
        inv_mat = cv2.invertAffineTransform(warp_mat)

        pts = xy.astype(np.float32).reshape(1, -1, 2)
        pts = cv2.transform(pts, inv_mat)
        pts = pts.reshape(-1, 2)

        return pts

    def _keypoints_to_box(self, xy, conf, img_w, img_h):
        visible = conf > self.score_threshold

        if not np.any(visible):
            return 0, 0, img_w - 1, img_h - 1

        pts = xy[visible]

        x1, y1 = np.min(pts, axis=0)
        x2, y2 = np.max(pts, axis=0)

        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))

        pad_w = int(0.05 * bw)
        pad_h = int(0.05 * bh)

        x1 = max(0, int(x1 - pad_w))
        y1 = max(0, int(y1 - pad_h))
        x2 = min(img_w - 1, int(x2 + pad_w))
        y2 = min(img_h - 1, int(y2 + pad_h))

        if x2 <= x1:
            x2 = min(img_w - 1, x1 + 1)

        if y2 <= y1:
            y2 = min(img_h - 1, y1 + 1)

        return x1, y1, x2, y2

    def _convert_coco17_to_custom_kp17(self, body17):
        body17 = body17.astype(np.float32)

        nose = body17[0]
        l_shoulder = body17[5]
        r_shoulder = body17[6]
        l_elbow = body17[7]
        r_elbow = body17[8]
        l_wrist = body17[9]
        r_wrist = body17[10]
        l_hip = body17[11]
        r_hip = body17[12]
        l_knee = body17[13]
        r_knee = body17[14]
        l_ankle = body17[15]
        r_ankle = body17[16]

        pelvis = (l_hip + r_hip) * 0.5
        thorax = (l_shoulder + r_shoulder) * 0.5
        spine = (pelvis + thorax) * 0.5

        shoulder_dist = float(np.linalg.norm(l_shoulder - r_shoulder))
        head = nose + np.array(
            [0.0, -0.25 * shoulder_dist],
            dtype=np.float32
        )

        kp17 = np.stack([
            pelvis,
            r_hip,
            r_knee,
            r_ankle,
            l_hip,
            l_knee,
            l_ankle,
            spine,
            thorax,
            nose,
            head,
            l_shoulder,
            l_elbow,
            l_wrist,
            r_shoulder,
            r_elbow,
            r_wrist
        ])

        return kp17.astype(np.float32)