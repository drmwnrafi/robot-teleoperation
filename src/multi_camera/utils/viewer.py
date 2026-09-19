# utils/viewer.py
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from typing import List, Tuple, Optional, Dict
from rich.console import Console
from rich.panel import Panel
from utils.skeleton import SKELETON_FORMATS


class PoseViewer:
    def __init__(self, window_name: str, base_widths: List[int], base_heights: List[int],
                 view_3d_base_width: int, confidence_threshold: float = 0.3,
                 skeleton_format: str = "coco_wholebody_133",
                 debug: bool = False,
                 initial_view_state: Optional[Dict] = None,
                 landmark_scale_2d: float = 2.5,
                 landmark_scale_3d: float = 0.9):

        self.window_name = window_name
        self.confidence_threshold = confidence_threshold
        self.debug = debug

        if skeleton_format not in SKELETON_FORMATS:
            raise ValueError(f"Unknown skeleton format: {skeleton_format}. Choose from {list(SKELETON_FORMATS.keys())}")
        self.skeleton_config = SKELETON_FORMATS[skeleton_format]
        self.num_keypoints = self.skeleton_config["num_keypoints"]

        self.screen_width, self.screen_height = self._get_screen_resolution()
        self.scale, self.target_h, self.target_widths, self.target_heights = self._calculate_scaling(
            base_widths, base_heights, view_3d_base_width
        )

        default_rot = self._yaw_pitch_roll_matrix(
            np.deg2rad(0.0),      # yaw (ψ)
            np.deg2rad(0.0),      # pitch (θ)
            np.deg2rad(0.0)       # roll (φ)
        )
        default_zoom = 3.0
        default_pan = np.array([0.0, 0.0])

        if initial_view_state is not None:
            try:
                default_rot = np.array(initial_view_state["rotation_matrix"], dtype=float)
            except KeyError:
                pass
            default_zoom = float(initial_view_state.get("zoom", default_zoom))
            pan = initial_view_state.get("pan_offset", default_pan)
            default_pan = np.array(pan, dtype=float)

        self._default_rotation_matrix = default_rot.copy()
        self._default_zoom = default_zoom
        self._default_pan = default_pan.copy()

        self.rotation_matrix = default_rot.copy()
        self.zoom = default_zoom
        self.pan_offset = default_pan.copy()

        self.is_panning = False
        self.last_pan_pos = (0, 0)

        self.is_dragging = False
        self.is_zooming = False
        self.last_zoom_y = 0

        self._arcball_start_vec: Optional[np.ndarray] = None
        self._arcball_start_matrix: Optional[np.ndarray] = None

        self.depth_cue_enabled = True
        self.show_axis_gizmo = True
        self.show_ghost_hypotheses = False

        self.landmark_scale_2d = landmark_scale_2d
        self.landmark_scale_3d = landmark_scale_3d

        self._gizmo_hit_points: Dict[str, Tuple[int, int]] = {}
        self._gizmo_hit_radius = 12
        self._gizmo_view_presets = {
            '+X': self._yaw_pitch_matrix(np.pi / 2, 0.0),
            '-X': self._yaw_pitch_matrix(-np.pi / 2, 0.0),
            '+Y': self._yaw_pitch_matrix(0.0, np.pi / 2 - 1e-3),
            '-Y': self._yaw_pitch_matrix(0.0, -np.pi / 2 + 1e-3),
            '+Z': self._yaw_pitch_matrix(0.0, 0.0),
            '-Z': self._yaw_pitch_matrix(np.pi, 0.0),
        }

        self._active_gizmo_label: Optional[str] = None
        self._active_gizmo_axis: Optional[str] = None     # 'X' / 'Y' / 'Z'
        self._active_gizmo_start: Tuple[int, int] = (0, 0)
        self._active_gizmo_last: Tuple[int, int] = (0, 0)
        self._active_gizmo_dragged: bool = False
        self._gizmo_drag_threshold_px: int = 4
        self._gizmo_rotate_sensitivity: float = 0.015

        self.root_joint_id = self.skeleton_config.get("root_joint_id", 0)
        self.trail_max_len = 20
        self.root_trail: List[np.ndarray] = []  # list of raw (unrotated, normalized) 3D points

        sep_width = 5
        self.offset_x_3d = (len(base_widths) * self.target_widths[0]) + (len(base_widths) * sep_width)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self._print_hotkeys()

    def _print_hotkeys(self):
        console = Console()
        console.print()
        console.print(
            Panel(
                "[bold white]Launch viewer ...[/bold white]\n\n"
                " [bold]3D view controls:[/bold]\n"
                "  Left drag on 3D pane      - Free trackball rotate (follows cursor)\n"
                "  Middle drag on 3D pane    - Pan view (X/Y translation)\n"
                "  Right drag on 3D pane     - Zoom\n"
                "  Drag a gizmo axis dot     - Rotate around JUST that axis (X/Y/Z)\n"
                "  Click a gizmo axis dot    - Snap to that axis's canonical view\n"
                "  R - Reset view (incl. pan & zoom)\n"
                "  D - Toggle depth cueing\n"
                "  G - Toggle ghost TTA hypotheses\n"
                "  P - Print current view state (for reuse via initial_view_state)\n\n"
                " [dim]Press ESC to exit[/dim]",
                title="[bold cyan] VIEWER [/bold cyan]",
                border_style="cyan",
                expand=False,
            )
        )
        console.print()

    def _get_screen_resolution(self) -> Tuple[int, int]:
        try:
            root = tk.Tk()
            root.withdraw()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return w, h
        except Exception:
            return 1920, 1080

    def _calculate_scaling(self, base_widths: List[int], base_heights: List[int], view_3d_base_width: int) -> Tuple[float, int, List[int], List[int]]:
        sep_width = 5
        num_panels = len(base_widths) + 1  # cameras + 1 for 3D viewer

        total_sep_width = (num_panels - 1) * sep_width
        available_width = self.screen_width - total_sep_width

        grid_w = max(1, available_width // num_panels)
        grid_h = max(1, self.screen_height // 2)

        target_widths = [grid_w] * num_panels
        target_heights = [grid_h] * num_panels

        return 1.0, grid_h, target_widths, target_heights

    def _hit_test_gizmo(self, local_x: int, local_y: int) -> Optional[str]:
        for label, (gx, gy) in self._gizmo_hit_points.items():
            if (local_x - gx) ** 2 + (local_y - gy) ** 2 <= self._gizmo_hit_radius ** 2:
                return label
        return None

    def _print_debug_state(self):
        if not self.debug:
            return
        rows = ",\n        ".join(
            "[" + ", ".join(f"{v:.6f}" for v in row) + "]" for row in self.rotation_matrix
        )
        print("\n--- viewer debug state (paste into initial_view_state) ---")
        print("initial_view_state = {")
        print(f"    \"rotation_matrix\": [\n        {rows}\n    ],")
        print(f"    \"zoom\": {self.zoom:.4f},")
        print(f"    \"pan_offset\": [{self.pan_offset[0]:.2f}, {self.pan_offset[1]:.2f}],")
        print("}")
        print("-----------------------------------------------------------\n")

    def _mouse_callback(self, event, x, y, flags, param):
        view_w = self.target_widths[-1]
        in_pane = self.offset_x_3d <= x < self.offset_x_3d + view_w

        # --- MIDDLE CLICK PANNING ---
        if event == cv2.EVENT_MBUTTONDOWN and in_pane:
            self.is_panning = True
            self.last_pan_pos = (x, y)
            return

        if event == cv2.EVENT_MBUTTONUP:
            self.is_panning = False
            self._print_debug_state()
            return

        if event == cv2.EVENT_MOUSEMOVE and self.is_panning:
            dx = x - self.last_pan_pos[0]
            dy = y - self.last_pan_pos[1]
            self.pan_offset[0] += dx
            self.pan_offset[1] += dy
            self.last_pan_pos = (x, y)
            return

        if event == cv2.EVENT_LBUTTONDOWN and in_pane:
            local_x, local_y = x - self.offset_x_3d, y
            hit = self._hit_test_gizmo(local_x, local_y)
            if hit is not None:
                self._active_gizmo_label = hit
                self._active_gizmo_axis = hit[1]  # 'X' / 'Y' / 'Z' from '+X' / '-X' / etc.
                self._active_gizmo_start = (x, y)
                self._active_gizmo_last = (x, y)
                self._active_gizmo_dragged = False
                self.is_dragging = False
                return
            self.is_dragging = True
            self._arcball_start_vec = self._arcball_vector(local_x, local_y)
            self._arcball_start_matrix = self.rotation_matrix.copy()
            return

        if event == cv2.EVENT_MOUSEMOVE and self._active_gizmo_axis is not None:
            dx = x - self._active_gizmo_last[0]
            dy = y - self._active_gizmo_last[1]
            total_dist = np.hypot(x - self._active_gizmo_start[0], y - self._active_gizmo_start[1])
            if total_dist > self._gizmo_drag_threshold_px:
                self._active_gizmo_dragged = True
            axis = self._active_gizmo_axis
            local_axis = {'X': np.array([1., 0., 0.]),
                           'Y': np.array([0., 1., 0.]),
                           'Z': np.array([0., 0., 1.])}[axis]
            delta = (dx if axis in ('Y', 'Z') else dy) * self._gizmo_rotate_sensitivity
            self._rotate_local(local_axis, delta)
            self._active_gizmo_last = (x, y)
            return

        if event == cv2.EVENT_LBUTTONUP and self._active_gizmo_axis is not None:
            if not self._active_gizmo_dragged and self._active_gizmo_label in self._gizmo_view_presets:
                self.rotation_matrix = self._gizmo_view_presets[self._active_gizmo_label].copy()
            self._active_gizmo_label = None
            self._active_gizmo_axis = None
            self._active_gizmo_dragged = False
            self._print_debug_state()
            return

        if in_pane:
            if event == cv2.EVENT_MOUSEMOVE and self.is_dragging and self._arcball_start_vec is not None:
                local_x, local_y = x - self.offset_x_3d, y
                v1 = self._arcball_vector(local_x, local_y)
                v0 = self._arcball_start_vec
                dot = float(np.clip(np.dot(v0, v1), -1.0, 1.0))
                angle = np.arccos(dot)

                axis = np.cross(v1, v0)

                delta_rot = self._rotation_about_axis(axis, angle)
                self.rotation_matrix = delta_rot @ self._arcball_start_matrix
            elif event == cv2.EVENT_LBUTTONUP:
                self.is_dragging = False
                self._arcball_start_vec = None
                self._arcball_start_matrix = None
                self._print_debug_state()

            elif event == cv2.EVENT_RBUTTONDOWN:
                self.is_zooming = True
                self.last_zoom_y = y
            elif event == cv2.EVENT_MOUSEMOVE and self.is_zooming:
                delta_y = self.last_zoom_y - y
                self.zoom += delta_y * 0.01
                self.zoom = max(0.1, min(5.0, self.zoom))
                self.last_zoom_y = y
            elif event == cv2.EVENT_RBUTTONUP:
                self.is_zooming = False
                self._print_debug_state()

    def _rotate_local(self, axis: np.ndarray, angle: float):
        self.rotation_matrix = self.rotation_matrix @ self._rotation_about_axis(axis, angle)

    @staticmethod
    def _yaw_pitch_matrix(yaw: float, pitch: float) -> np.ndarray:
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        return Rx @ Ry

    @staticmethod
    def _yaw_pitch_roll_matrix(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])

        return Rz @ Rx @ Ry

    @staticmethod
    def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-9 or abs(angle) < 1e-9:
            return np.eye(3)
        x, y, z = axis / norm
        c, s, C = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
        return np.array([
            [x * x * C + c,     x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c,     y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ])

    def _arcball_vector(self, local_x: float, local_y: float) -> np.ndarray:
        w, h = self.target_widths[-1], self.target_h
        cx, cy = w / 2.0, h / 2.0
        radius = min(w, h) / 2.0 * 0.95
        dx = (local_x - cx) / radius
        dy = (local_y - cy) / radius
        d2 = dx * dx + dy * dy
        if d2 <= 1.0:
            dz = np.sqrt(1.0 - d2)
        else:
            n = np.sqrt(d2)
            dx, dy = dx / n, dy / n
            dz = 0.0
        v = np.array([dx, -dy, dz])
        return v / np.linalg.norm(v)

    def reset_view(self):
        self.rotation_matrix = self._default_rotation_matrix.copy()
        self.zoom = self._default_zoom
        self.pan_offset = self._default_pan.copy()

    def _get_rotation_matrix(self):
        return self.rotation_matrix

    def _get_euler_angles_deg(self):
        R = self.rotation_matrix
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0.0
        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    def _draw_unicode_text(self, canvas: np.ndarray, text: str, pos: Tuple[int, int],
                           font_size: int = 13, color: Tuple[int, int, int] = (220, 220, 220)):
        try:
            from PIL import Image, ImageDraw, ImageFont
            pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            font_paths = ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "Helvetica.ttf"]
            font = None
            for fp in font_paths:
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except IOError:
                    continue

            if font is None:
                font = ImageFont.load_default()

            draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except ImportError:
            ascii_text = text.replace("φ", "R").replace("θ", "P").replace("ψ", "Y").replace("°", "deg")
            cv2.putText(canvas, ascii_text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            return canvas

    def _project_points(self, pts_norm: np.ndarray, w: int, h: int, focal: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
        pts_rot = (self._get_rotation_matrix() @ pts_norm.T).T
        z = np.maximum(pts_rot[:, 2] + focal, 0.1)
        norm_x = pts_rot[:, 0] / z
        norm_y = pts_rot[:, 1] / z

        proj_x = (norm_x * self.zoom) * (w / 2) + w / 2 + self.pan_offset[0]
        proj_y = (norm_y * self.zoom) * (h / 2) + h / 2 + self.pan_offset[1]

        return np.stack([proj_x, proj_y], axis=1), z

    def _draw_pose_generic(self, output: np.ndarray, kp_map: dict, is_3d: bool = False,
                            depth_map: Optional[Dict[int, float]] = None,
                            conf_map: Optional[Dict[int, float]] = None,
                            alpha_override: Optional[float] = None,
                            color_override: Optional[Tuple[int, int, int]] = None):
        scale_factor = self.landmark_scale_3d if is_3d else self.landmark_scale_2d

        near, far = None, None
        if depth_map:
            zs = list(depth_map.values())
            if zs:
                near, far = min(zs), max(zs)

        for part_name, part_config in self.skeleton_config["parts"].items():
            base_color = color_override if color_override is not None else part_config["color"]
            base_radius = part_config["radius"] * scale_factor
            base_thickness = part_config["thickness"] * scale_factor

            for idx in part_config["indices"]:
                if idx not in kp_map:
                    continue
                radius, color, alpha = base_radius, base_color, 1.0

                if depth_map and idx in depth_map and near is not None and far is not None and far > near:
                    t = np.clip((depth_map[idx] - near) / (far - near), 0.0, 1.0)
                    radius = base_radius * (1.35 - 0.65 * t)
                    dim = 1.0 - 0.55 * t
                    color = tuple(int(c * dim) for c in base_color)

                if conf_map and idx in conf_map:
                    alpha *= np.clip(conf_map[idx], 0.15, 1.0)
                if alpha_override is not None:
                    alpha *= alpha_override

                self._draw_circle_alpha(output, kp_map[idx], int(max(1, radius)), color, alpha)

            for a, b in part_config["skeleton"]:
                if a in kp_map and b in kp_map:
                    thickness = base_thickness
                    color = base_color
                    alpha = 1.0
                    if depth_map and a in depth_map and b in depth_map and near is not None and far is not None and far > near:
                        t = np.clip(((depth_map[a] + depth_map[b]) / 2 - near) / (far - near), 0.0, 1.0)
                        thickness = base_thickness * (1.25 - 0.5 * t)
                        dim = 1.0 - 0.55 * t
                        color = tuple(int(c * dim) for c in base_color)
                    if conf_map:
                        alpha *= np.clip(min(conf_map.get(a, 1.0), conf_map.get(b, 1.0)), 0.15, 1.0)
                    if alpha_override is not None:
                        alpha *= alpha_override
                    self._draw_line_alpha(output, kp_map[a], kp_map[b], color, int(max(1, thickness)), alpha)

    @staticmethod
    def _draw_circle_alpha(img, center, radius, color, alpha):
        if alpha >= 0.98:
            cv2.circle(img, center, radius, color, -1)
            return
        overlay = img.copy()
        cv2.circle(overlay, center, radius, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

    @staticmethod
    def _draw_line_alpha(img, p1, p2, color, thickness, alpha):
        if alpha >= 0.98:
            cv2.line(img, p1, p2, color, thickness)
            return
        overlay = img.copy()
        cv2.line(overlay, p1, p2, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

    def _draw_2d_frame(self, frame, packet, cam_id):
        output = frame.copy()
        if not packet:
            cv2.putText(output, f"Cam {cam_id} (No Detection)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return output

        kp_map = {int(pid): (int(x), int(y)) for pid, (x, y), conf in zip(packet.keypoint_id, packet.img_loc, packet.confidence)
                  if float(conf) >= self.confidence_threshold and 0 <= pid < self.num_keypoints}

        self._draw_pose_generic(output, kp_map, is_3d=False)
        cv2.putText(output, f"Cam {cam_id}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return output

    def _draw_axis_gizmo(self, canvas, w, h, origin_px=(50, 50), length=28):
        axes = np.array([
            [0, 0, 0],
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ], dtype=float)
        rot = self._get_rotation_matrix()
        rotated = (rot @ axes.T).T
        ox, oy = origin_px
        pts_2d = [(ox + rotated[i, 0] * length, oy + rotated[i, 1] * length) for i in range(7)]
        depths = rotated[:, 2]

        labels = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
        base_colors = {
            '+X': (60, 60, 255), '-X': (60, 60, 255),
            '+Y': (60, 255, 60), '-Y': (60, 255, 60),
            '+Z': (255, 140, 60), '-Z': (255, 140, 60),
        }

        origin = (int(pts_2d[0][0]), int(pts_2d[0][1]))
        self._gizmo_hit_points = {}

        cv2.circle(canvas, origin, int(length * 1.55), (35, 35, 38), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, origin, int(length * 1.55), (70, 70, 75), 1, lineType=cv2.LINE_AA)

        for i, label in enumerate(labels, start=1):
            end = (int(pts_2d[i][0]), int(pts_2d[i][1]))
            color = base_colors[label]
            is_positive = label.startswith('+')
            depth_t = np.clip((depths[i] + 1) / 2, 0.0, 1.0)
            dim = 0.5 + 0.5 * depth_t
            line_color = tuple(int(c * dim) for c in color)

            cv2.line(canvas, origin, end, line_color, 2, lineType=cv2.LINE_AA)
            radius = 9 if is_positive else 6
            if is_positive:
                cv2.circle(canvas, end, radius, line_color, -1, lineType=cv2.LINE_AA)
                cv2.putText(canvas, label[1], (end[0] - 4, end[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (10, 10, 10), 1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, end, radius, line_color, 1, lineType=cv2.LINE_AA)

            self._gizmo_hit_points[label] = end

    def _update_root_trail(self, pts_scaled: np.ndarray, kp_ids: np.ndarray):
        if self.root_joint_id in kp_ids:
            idx = int(np.where(kp_ids == self.root_joint_id)[0][0])
            self.root_trail.append(pts_scaled[idx].copy())
            if len(self.root_trail) > self.trail_max_len:
                self.root_trail.pop(0)

    def _draw_root_trail(self, canvas, w, h):
        if len(self.root_trail) < 2:
            return
        proj, _ = self._project_points(np.array(self.root_trail), w, h)
        n = len(proj)
        for i in range(1, n):
            t = i / n
            p1 = tuple(np.round(proj[i - 1]).astype(int))
            p2 = tuple(np.round(proj[i]).astype(int))
            self._draw_line_alpha(canvas, p1, p2, (200, 200, 60), 2, alpha=0.15 + 0.55 * t)

    def _render_3d_view(self, xyz_df, w, h, hypotheses: Optional[np.ndarray] = None):
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 20

        if xyz_df is None or getattr(xyz_df, 'empty', True):
            if self.show_axis_gizmo:
                self._draw_axis_gizmo(canvas, w, h)
            cv2.putText(canvas, "No 3D Data", (w // 2 - 70, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return canvas

        pts = xyz_df[['x_coord', 'y_coord', 'z_coord']].values
        kp_ids = xyz_df['keypoint_id'].values.astype(int)
        conf_col = xyz_df['confidence'].values if 'confidence' in xyz_df.columns else None

        center = np.array([0.0, 0.0, 0.0])
        fixed_extent = 2.0
        pts_scaled = (pts - center) / fixed_extent * 1.5

        proj_xy, depth_z = self._project_points(pts_scaled, w, h)
        kp_map = {int(kp): (int(x), int(y)) for kp, (x, y) in zip(kp_ids, proj_xy)}
        depth_map = {int(kp): float(z) for kp, z in zip(kp_ids, depth_z)} if self.depth_cue_enabled else None
        conf_map = {int(kp): float(c) for kp, c in zip(kp_ids, conf_col)} if conf_col is not None else None

        if self.show_ghost_hypotheses and hypotheses is not None:
            ghost_colors = [(180, 180, 180), (120, 180, 255), (180, 255, 120)]
            for hi in range(hypotheses.shape[0]):
                h_pts = hypotheses[hi]
                h_scaled = (h_pts - center) / fixed_extent * 1.5
                h_proj, _ = self._project_points(h_scaled, w, h)
                h_kp_map = {i: (int(x), int(y)) for i, (x, y) in enumerate(h_proj)}
                self._draw_pose_generic(canvas, h_kp_map, is_3d=True,
                                         alpha_override=0.35,
                                         color_override=ghost_colors[hi % len(ghost_colors)])

        self._update_root_trail(pts_scaled, kp_ids)
        self._draw_root_trail(canvas, w, h)

        self._draw_pose_generic(canvas, kp_map, is_3d=True, depth_map=depth_map, conf_map=conf_map)

        if self.show_axis_gizmo:
            self._draw_axis_gizmo(canvas, w, h)

        # Bottom-left: Euler angles
        yaw, pitch, roll = self._get_euler_angles_deg()
        info_text = f"φ : {roll:5.1f}°   θ : {pitch:5.1f}°   ψ : {yaw:5.1f}°"
        canvas = self._draw_unicode_text(canvas, info_text, (15, h-20), font_size=13, color=(220, 220, 220))

        # Bottom-right: Zoom value
        zoom_text = f"Zoom: {self.zoom:.2f}x"
        (text_width, _), _ = cv2.getTextSize(zoom_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(canvas, zoom_text, (w - text_width - 15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

        return canvas

    def show(self, raw_frames: List[np.ndarray], packets: List, xyz_df: Optional[pd.DataFrame], frame_index: int,
              hypotheses: Optional[np.ndarray] = None) -> bool:
        drawn_2d_frames = [self._draw_2d_frame(f, p, i) for i, (f, p) in enumerate(zip(raw_frames, packets))]

        resized_2d = []
        for i, f in enumerate(drawn_2d_frames):
            if f.size == 0:
                padded = np.zeros((self.target_heights[i], self.target_widths[i], 3), dtype=np.uint8)
                resized_2d.append(padded)
                continue

            orig_h, orig_w = f.shape[:2]
            orig_aspect = orig_w / orig_h

            grid_w = self.target_widths[i]
            grid_h = self.target_heights[i]
            grid_aspect = grid_w / grid_h

            if orig_aspect > grid_aspect:
                target_w = grid_w
                target_h = max(1, int(grid_w / orig_aspect))
                resized = cv2.resize(f, (target_w, target_h))
                pad_top = (grid_h - target_h) // 2
                pad_bottom = grid_h - target_h - pad_top
                padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[20, 20, 20])
            else:
                target_h = grid_h
                target_w = max(1, int(grid_h * orig_aspect))
                resized = cv2.resize(f, (target_w, target_h))
                pad_left = (grid_w - target_w) // 2
                pad_right = grid_w - target_w - pad_left
                padded = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[20, 20, 20])

            resized_2d.append(padded)

        view_3d = self._render_3d_view(xyz_df, self.target_widths[-1], self.target_heights[-1], hypotheses=hypotheses)

        sep_h = self.target_heights[0] if self.target_heights else self.target_h
        sep = np.ones((sep_h, 5, 3), dtype=np.uint8) * 255

        combined_parts = []
        for frame in resized_2d:
            combined_parts.append(frame)
            combined_parts.append(sep)
        combined_parts.append(view_3d)

        combined = cv2.hconcat(combined_parts)

        pts_count = len(xyz_df) if xyz_df is not None and not getattr(xyz_df, 'empty', True) else 0
        cv2.putText(combined, f"Frame: {frame_index} | 3D Pts: {pts_count}",
                    (20, combined.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 2)

        cv2.resizeWindow(self.window_name, combined.shape[1], combined.shape[0])
        cv2.imshow(self.window_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return False
        elif key in (ord('r'), ord('R')):
            self.reset_view()
        elif key in (ord('d'), ord('D')):
            self.depth_cue_enabled = not self.depth_cue_enabled
        elif key in (ord('g'), ord('G')):
            self.show_ghost_hypotheses = not self.show_ghost_hypotheses
        elif key in (ord('p'), ord('P')):
            self._print_debug_state()

        return True

    def destroy(self):
        cv2.destroyAllWindows()
