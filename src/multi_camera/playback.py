import os
import cv2
import time
import queue
import threading
import argparse
import sys
import tomllib
import numpy as np
import pandas as pd
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.live import Live
from rich.progress import (
    Progress, BarColumn, TextColumn, MofNCompleteColumn,
    TimeElapsedColumn, TimeRemainingColumn
)

from caliscope.trackers.onnx_tracker import OnnxTracker
from caliscope.trackers.model_card import ModelCard
from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import ImagePoints

from utils.viewer import PoseViewer
from utils.babyros_publisher import BabyROSPublisher

console = Console()


class RecordedPacket:
    __slots__ = ("keypoint_id", "img_loc", "confidence")

    def __init__(self, keypoint_id, img_loc, confidence):
        self.keypoint_id = keypoint_id
        self.img_loc = img_loc
        self.confidence = confidence

    def __bool__(self):
        return len(self.keypoint_id) > 0


def main():
    parser = argparse.ArgumentParser(description="Replay videos with 3D tracking")
    parser.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="Session folder containing info.toml and cam_*.mp4"
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        default=None,
        help="Video files to replay (e.g., cam_0.mp4 cam_1.mp4)"
    )
    parser.add_argument(
        "--camera-array",
        type=Path,
        default=None,
        help="Path to camera_array.toml"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to RTMPose ONNX model"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0, e.g., 2.0 for 2x speed, 0.5 for half speed)"
    )
    parser.add_argument(
        "--xyz",
        action="store_true",
        help="If recorded 2D/3D landmarks already exist (saved alongside --folder, "
             "or in the parent folder of --videos), skip running the pose model "
             "entirely and just replay the recorded 2D keypoints (all cams) and "
             "3D landmarks over the videos."
    )
    args = parser.parse_args()

    if args.folder is None and args.videos is None:
        parser.error("one of --folder or --videos is required")

    if args.folder is not None and args.videos is not None:
        parser.error("--folder cannot be used together with --videos")

    if args.folder is not None and args.camera_array is not None:
        parser.error("--camera-array cannot be used together with --folder")

    if args.folder is not None and args.model is not None:
        parser.error("--model cannot be used together with --folder")

    speed_multiplier = max(0.1, args.speed)

    if args.folder is not None:
        folder = args.folder.resolve()
        info_path = folder / "info.toml"

        if not folder.is_dir():
            console.print(f"[bold red]ERROR:[/bold red] Folder not found: [cyan]{folder}[/cyan]")
            sys.exit(1)

        if not info_path.exists():
            console.print(f"[bold red]ERROR:[/bold red] File not found: [cyan]{info_path}[/cyan]")
            sys.exit(1)

        with open(info_path, "rb") as f:
            info = tomllib.load(f)

        arguments = info.get("arguments", {})

        videos = sorted(folder.glob("cam_*.mp4"), key=lambda p: p.name)
        if not videos:
            console.print(f"[bold red]ERROR:[/bold red] No cam_*.mp4 files found in [cyan]{folder}[/cyan]")
            sys.exit(1)

        camera_array_value = arguments.get("camera_array_path")
        model_value = arguments.get("model_path")

        if not camera_array_value:
            console.print("[bold red]ERROR:[/bold red] camera_array_path not found in info.toml")
            sys.exit(1)

        if not model_value:
            console.print("[bold red]ERROR:[/bold red] model_path not found in info.toml")
            sys.exit(1)

        camera_array_path = Path(camera_array_value)
        model_path = Path(model_value)

        confidence_threshold = float(arguments.get("confidence_threshold", 0.4))
        if args.conf is not None:
            confidence_threshold = args.conf

        mode = "Folder"

    else:
        videos = args.videos

        if args.camera_array is None:
            parser.error("--camera-array is required when using --videos")

        camera_array_path = args.camera_array.resolve()
        model_path = args.model.resolve() if args.model is not None else Path("models/rtmpose_l_coco_wholebody.onnx").resolve()
        confidence_threshold = args.conf if args.conf is not None else 0.4

        mode = "Explicit"

    if args.folder is not None:
        landmarks_dir = folder
    else:
        landmarks_dir = videos[0].resolve().parent

    landmarks_2d_path = landmarks_dir / "landmarks_2d.csv"
    landmarks_3d_path = landmarks_dir / "landmarks_3d.csv"

    xyz_playback_active = False
    landmarks_2d_df_all = None
    landmarks_3d_df_all = None

    if args.xyz:
        if landmarks_2d_path.exists() and landmarks_3d_path.exists():
            landmarks_2d_df_all = pd.read_csv(landmarks_2d_path)
            landmarks_3d_df_all = pd.read_csv(landmarks_3d_path)
            if not landmarks_2d_df_all.empty and not landmarks_3d_df_all.empty:
                xyz_playback_active = True

    paths_to_validate = list(videos)
    if not xyz_playback_active:
        paths_to_validate = [camera_array_path, model_path] + paths_to_validate

    for path in paths_to_validate:
        if not path.exists():
            console.print(f"[bold red]ERROR:[/bold red] File not found: [cyan]{path.resolve()}[/cyan]")
            sys.exit(1)

    console.print()
    console.print(Panel(
        "[bold white]VIDEO PLAYBACK WITH 3D TRACKING[/bold white]",
        style="bold blue",
        expand=False,
    ))

    if args.xyz and not xyz_playback_active:
        console.print()
        console.print(Panel(
            "[bold yellow]--xyz was requested, but no recorded landmarks were found[/bold yellow]\n"
            f"[dim]Looked for:[/dim] {landmarks_2d_path.name}, {landmarks_3d_path.name} "
            f"in [cyan]{landmarks_dir}[/cyan]\n"
            "[white]Falling back to running the pose model — landmarks will be recorded this run.[/white]",
            title="[bold yellow]NOTICE[/bold yellow]",
            border_style="yellow",
            expand=False,
        ))

    info_table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    info_table.add_column("Setting", style="bold cyan")
    info_table.add_column("Value", style="white")
    info_table.add_row("Mode", mode)

    if args.folder is not None:
        info_table.add_row("Session Folder", str(args.folder.resolve()))

    info_table.add_row(
        "Landmark Mode",
        "[bold green]Playback (recorded 2D/3D, model skipped)[/bold green]"
        if xyz_playback_active else
        "[bold yellow]Recording (running model)[/bold yellow]"
    )
    info_table.add_row("Landmarks Dir", str(landmarks_dir))

    if not xyz_playback_active:
        info_table.add_row("Camera Array", str(camera_array_path.resolve()))
        info_table.add_row("Model", str(model_path.resolve()))

    info_table.add_row("Confidence Threshold", f"[bold yellow]{confidence_threshold}[/bold yellow]")
    info_table.add_row("Playback Speed", f"[bold green]{speed_multiplier}x[/bold green]")
    info_table.add_row("Video Files", str(len(videos)))

    console.print()
    console.print(info_table)

    console.print()
    console.print(Rule("[bold cyan]Loading Components[/bold cyan]"))

    camera_array = None
    tracker = None

    if not xyz_playback_active:
        console.print(f"  [bold yellow]Camera Array:[/bold yellow] {camera_array_path.name}")
        camera_array = CameraArray.from_toml(camera_array_path)

        console.print(f"  [bold yellow]RTMPose Model:[/bold yellow] {model_path.name}")
        card = ModelCard(
            name="RTMPose-L-WholeBody",
            model_path=model_path,
            format="simcc",
            input_width=288,
            input_height=384,
            confidence_threshold=confidence_threshold,
            point_name_to_id={f"keypoint_{i}": i for i in range(133)},
            wireframe=None
        )
        tracker = OnnxTracker(card)
    else:
        console.print(
            f"  [dim]Skipping camera array & model load — replaying recorded landmarks from "
            f"[cyan]{landmarks_dir}[/cyan][/dim]"
        )

    console.print()
    console.print(Rule("[bold cyan]Opening Videos[/bold cyan]"))
    caps = [cv2.VideoCapture(str(f)) for f in videos]

    if not all(c.isOpened() for c in caps):
        console.print("[bold red]ERROR:[/bold red] Failed to open one or more videos")
        sys.exit(1)

    widths = [int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) for c in caps]
    heights = [int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) for c in caps]
    fps_list = [c.get(cv2.CAP_PROP_FPS) or 30.0 for c in caps]
    frame_counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]

    base_fps = fps_list[0]
    target_frame_time = (1.0 / base_fps) / speed_multiplier

    vid_table = Table(title="Video Files", show_header=True, header_style="bold magenta")
    vid_table.add_column("Camera", style="bold cyan", justify="center")
    vid_table.add_column("File", style="white")
    vid_table.add_column("Resolution", justify="center")
    vid_table.add_column("FPS", justify="center")
    vid_table.add_column("Frames", justify="right")
    vid_table.add_column("Duration", justify="right")

    for i, (f, w, h, fps, fc) in enumerate(zip(videos, widths, heights, fps_list, frame_counts)):
        duration_sec = fc / fps if fps > 0 else 0
        duration_str = f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s"
        vid_table.add_row(f"Cam {i}", f.name, f"{w}x{h}", f"{fps:.1f}", str(fc), duration_str)

    console.print()
    console.print(vid_table)

    viewer = PoseViewer(
        window_name="Playback 3D Viewer",
        base_widths=widths,
        base_heights=heights,
        view_3d_base_width=heights[0],
        confidence_threshold=confidence_threshold,
        skeleton_format="coco_wholebody_133",
        debug=False,
        initial_view_state = {
            "rotation_matrix": [
                [0.418491, 0.798565, 0.432619],
                [0.854092, -0.184047, -0.486470],
                [-0.308856, 0.573080, -0.759070]
            ],
            "zoom": 3.4100,
            "pan_offset": [165.00, 192.00],
        }
    )

    ros_publisher = BabyROSPublisher()

    num_cams = len(caps)

    landmarks_2d_groups = {}
    landmarks_3d_groups = {}

    if xyz_playback_active:
        for key, df in landmarks_2d_df_all.groupby(["sync_index", "cam_id"]):
            landmarks_2d_groups[key] = df
        for key, df in landmarks_3d_df_all.groupby("sync_index"):
            landmarks_3d_groups[key] = df.drop(columns=["time_sec"], errors="ignore")

    def packet_from_recorded(sync_index, cam_id):
        df = landmarks_2d_groups.get((sync_index, cam_id))
        if df is None or df.empty:
            return None
        keypoint_id = df["keypoint_id"].to_numpy()
        img_loc = list(zip(df["img_loc_x"].to_numpy(), df["img_loc_y"].to_numpy()))
        confidence = df["confidence"].to_numpy()
        return RecordedPacket(keypoint_id, img_loc, confidence)

    def xyz_from_recorded(sync_index):
        return landmarks_3d_groups.get(sync_index)

    recorded_2d_rows = []
    recorded_3d_frames = []

    def make_frame_image_points(frame_index, packets):
        rows = []
        for cam_id, packet in enumerate(packets):
            if not packet: continue
            for point_id, (x, y), confidence in zip(packet.keypoint_id, packet.img_loc, packet.confidence):
                if float(confidence) >= confidence_threshold:
                    rows.append({
                        "sync_index": frame_index,
                        "cam_id": cam_id,
                        "object_id": 0,
                        "keypoint_id": int(point_id),
                        "img_loc_x": float(x),
                        "img_loc_y": float(y)
                    })
        return ImagePoints(pd.DataFrame(rows)) if rows else None

    def triangulate_frame(frame_index, packets):
        image_points = make_frame_image_points(frame_index, packets)
        if image_points is None: return None
        world_points = image_points.triangulate(camera_array)
        return world_points.df if not world_points.df.empty else None

    def record_2d_frame(frame_index, packets):
        for cam_id, packet in enumerate(packets):
            if not packet: continue
            for point_id, (x, y), confidence in zip(packet.keypoint_id, packet.img_loc, packet.confidence):
                recorded_2d_rows.append({
                    "sync_index": frame_index,
                    "time_sec": frame_index / base_fps,
                    "cam_id": cam_id,
                    "object_id": 0,
                    "keypoint_id": int(point_id),
                    "img_loc_x": float(x),
                    "img_loc_y": float(y),
                    "confidence": float(confidence),
                })

    def record_3d_frame(frame_index, xyz_df):
        if xyz_df is None or xyz_df.empty:
            return
        df = xyz_df.copy()
        if "sync_index" not in df.columns:
            df["sync_index"] = frame_index
        df["time_sec"] = frame_index / base_fps
        recorded_3d_frames.append(df)

    input_q = queue.Queue(maxsize=2)
    output_q = queue.Queue(maxsize=2)
    worker = None

    if not xyz_playback_active:
        def worker_thread():
            while True:
                item = input_q.get()
                if item is None: break
                frame_idx, frames = item
                packets = [tracker._detect(f, cam_id=i) for i, f in enumerate(frames)]
                xyz_df = triangulate_frame(frame_idx, packets)
                record_2d_frame(frame_idx, packets)
                record_3d_frame(frame_idx, xyz_df)
                output_q.put((frame_idx, packets, xyz_df))

        worker = threading.Thread(target=worker_thread, daemon=True)
        worker.start()

    console.print()
    effective_fps = base_fps * speed_multiplier

    frame_index = 0
    last_result = (0, [None] * num_cams, None)
    last_frame_time = time.perf_counter()

    progress = Progress(
        TextColumn("[bold cyan]Playback"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )
    playback_task = progress.add_task("playback", total=frame_counts[0])

    def render_display(points_3d: int, actual_fps: float) -> Panel:
        mode_line = (
            "[bold green]Replaying recorded 2D/3D landmarks (model skipped)[/bold green]"
            if xyz_playback_active else
            "[bold white]Starting playback with 3D tracking...[/bold white]"
        )
        intro = (
            f"{mode_line}\n"
            f"Speed: [bold green]{speed_multiplier}x ({effective_fps:.1f} effective FPS)[/bold green]\n"
            "[dim]Press ESC to exit[/dim]"
        )

        stats = Table.grid(padding=(0, 2))
        stats.add_column(justify="right", style="dim")
        stats.add_column(style="bold white")
        stats.add_row("3D points tracked:", f"[bold green]{points_3d}[/bold green]")
        stats.add_row(
            "Actual FPS:",
            f"[bold yellow]{actual_fps:.1f}[/bold yellow] [dim]/ {effective_fps:.1f} target[/dim]",
        )

        body = Group(intro, "", progress, "", stats)

        return Panel(
            body,
            title="[bold cyan]PLAYBACK[/bold cyan]",
            border_style="cyan",
            expand=False,
        )

    live = Live(render_display(0, 0.0), console=console, refresh_per_second=12)
    live.start()

    prev_loop_time = time.perf_counter()
    fps_smooth = 0.0

    try:
        while True:
            elapsed = time.perf_counter() - last_frame_time
            sleep_time = target_frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_frame_time = time.perf_counter()

            frames = []
            all_ok = True

            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    all_ok = False
                    break
                frames.append(frame)

            if not all_ok or len(frames) != num_cams:
                break

            if xyz_playback_active:
                packets = [packet_from_recorded(frame_index, cam_id) for cam_id in range(num_cams)]
                xyz_df = xyz_from_recorded(frame_index)
                last_result = (frame_index, packets, xyz_df)
            else:
                try:
                    input_q.put_nowait((frame_index, frames))
                except queue.Full:
                    pass

                try:
                    last_result = output_q.get_nowait()
                except queue.Empty:
                    pass

            res_idx, packets, xyz_df = last_result

            ros_publisher.publish_frame(
                frame_index=frame_index,
                xyz_df=xyz_df
            )

            if not viewer.show(frames, packets, xyz_df, frame_index):
                break

            frame_index += 1

            now = time.perf_counter()
            inst_fps = 1.0 / (now - prev_loop_time) if now > prev_loop_time else 0.0
            prev_loop_time = now
            fps_smooth = inst_fps if frame_index == 1 else (0.9 * fps_smooth + 0.1 * inst_fps)

            points_3d = len(xyz_df) if xyz_df is not None else 0

            progress.update(playback_task, completed=frame_index)
            live.update(render_display(points_3d, fps_smooth))

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Keyboard interrupt detected.[/bold yellow]")

    finally:
        live.stop()

    console.print()
    console.print(Rule("[bold cyan]Cleaning Up[/bold cyan]"))

    saved_2d_path = None
    saved_3d_path = None
    saved_2d_rows = 0
    saved_3d_rows = 0

    if not xyz_playback_active:
        input_q.put(None)
        if worker is not None:
            while worker.is_alive():
                try:
                    output_q.get_nowait()
                except queue.Empty:
                    time.sleep(0.01)
            worker.join()

        landmarks_dir.mkdir(parents=True, exist_ok=True)

        landmarks_2d_out = pd.DataFrame(recorded_2d_rows)
        landmarks_3d_out = (
            pd.concat(recorded_3d_frames, ignore_index=True)
            if recorded_3d_frames else pd.DataFrame()
        )

        landmarks_2d_out.to_csv(landmarks_2d_path, index=False)
        landmarks_3d_out.to_csv(landmarks_3d_path, index=False)

        saved_2d_path, saved_3d_path = landmarks_2d_path, landmarks_3d_path
        saved_2d_rows, saved_3d_rows = len(landmarks_2d_out), len(landmarks_3d_out)

    for cap in caps:
        cap.release()

    viewer.destroy()
    ros_publisher.cleanup()
    console.print("  [dim]Videos released and viewer closed.[/dim]")

    console.print()

    done_lines = [
        "[bold green]Playback finished[/bold green]",
        "",
        f"[white]Total Frames Processed:[/white] [bold cyan]{frame_index}[/bold cyan]",
    ]

    if xyz_playback_active:
        done_lines.append(
            f"[white]Landmarks replayed from:[/white] [bold cyan]{landmarks_dir}[/bold cyan]"
        )
    elif saved_2d_path is not None:
        done_lines.append(
            f"[white]2D landmarks saved:[/white] [bold cyan]{saved_2d_path}[/bold cyan] "
            f"([green]{saved_2d_rows}[/green] rows)"
        )
        done_lines.append(
            f"[white]3D landmarks saved:[/white] [bold cyan]{saved_3d_path}[/bold cyan] "
            f"([green]{saved_3d_rows}[/green] rows)"
        )

    console.print(Panel(
        "\n".join(done_lines),
        title="[bold green]DONE[/bold green]",
        style="bold green",
        expand=False,
    ))


if __name__ == "__main__":
    main()
