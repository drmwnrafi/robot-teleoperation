import sys
import os
import cv2
import time
import queue
import threading
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

from caliscope.trackers.onnx_tracker import OnnxTracker
from caliscope.trackers.model_card import ModelCard
from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import ImagePoints

from utils.viewer import PoseViewer
from utils.babyros_publisher import BabyROSPublisher

console = Console()

class ThreadedVideoWriter:
    def __init__(self, filename, fourcc, fps, frame_size):
        self.writer = cv2.VideoWriter(str(filename), fourcc, fps, frame_size)
        self.queue = queue.Queue(maxsize=120)
        self.stopped = False
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def _write_loop(self):
        while not self.stopped or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                if frame is not None:
                    self.writer.write(frame)
                self.queue.task_done()
            except:
                continue

    def write(self, frame):
        if not self.stopped:
            self.queue.put(frame)

    def release(self):
        self.stopped = True
        self.writer.release()
        self.thread.join()

def write_info_toml(output_dir: Path, args, cal_sizes, actual_sizes, actual_fps, session_start: str):
    info_path = output_dir / "info.toml"

    def esc(p):
        return str(p).replace("\\", "\\\\")

    lines = []

    lines.append("[session]")
    lines.append(f'timestamp = "{session_start}"')
    lines.append(f'output_directory = "{esc(output_dir.resolve())}"')
    lines.append("")

    lines.append("[arguments]")
    lines.append(f"camera_indices = {args.cams}")
    lines.append(f'camera_array_path = "{esc(args.camera_array.resolve())}"')
    lines.append(f'model_path = "{esc(args.model.resolve())}"')
    lines.append(f'output_base_dir = "{esc(args.output.resolve())}"')
    lines.append(f"confidence_threshold = {args.conf}")
    lines.append(f"save_enabled = {'true' if args.save else 'false'}")
    lines.append("")

    lines.append("[model]")
    lines.append('name = "RTMPose-L-WholeBody"')
    lines.append('format = "simcc"')
    lines.append("input_width = 288")
    lines.append("input_height = 384")
    lines.append("num_keypoints = 133")
    lines.append('skeleton_format = "coco_wholebody_133"')
    lines.append("")

    lines.append("[cameras]")
    for i, idx in enumerate(args.cams):
        w_cal, h_cal = cal_sizes[i]
        w_act, h_act = actual_sizes[i]
        fps = actual_fps[i]
        match = (w_act, h_act) == (w_cal, h_cal)

        lines.append("")
        lines.append(f"[cameras.cam_{idx}]")
        lines.append(f"requested_width = {w_cal}")
        lines.append(f"requested_height = {h_cal}")
        lines.append(f"actual_width = {w_act}")
        lines.append(f"actual_height = {h_act}")
        lines.append(f"actual_fps = {fps:.2f}")
        lines.append(f"calibration_match = {'true' if match else 'false'}")

    lines.append("")

    lines.append("[output_files]")
    video_list = ", ".join(f'"cam_{idx}.mp4"' for idx in args.cams)
    lines.append(f"videos = [{video_list}]")
    lines.append('session_info = "info.toml"')
    lines.append("")

    with open(info_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return info_path


def append_summary_toml(output_dir: Path, frame_index: int):
    info_path = output_dir / "info.toml"

    lines = []
    lines.append("[summary]")
    lines.append(f"total_frames = {frame_index}")
    lines.append("")

    with open(info_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Live 3D RTMPose Multi-Camera Tracking")
    parser.add_argument("cams", nargs="+", type=int, help="Camera indices (e.g., 0 1)")
    parser.add_argument("--camera-array", type=Path, required=True, help="Path to camera_array.toml")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/dwpose_l_coco_wholebody_384x288.onnx"),
        help="Path to RTMPose ONNX model (default: models/rtmpose_l_coco_wholebody.onnx)"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Base output directory (default: outputs)")
    parser.add_argument("--save", action="store_true", help="Save camera video feeds as MP4 and info.toml")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")

    args = parser.parse_args()

    if len(args.cams) < 2:
        console.print("[bold red]ERROR:[/bold red] At least 2 camera indices are required for 3D triangulation.")
        sys.exit(1)

    session_start = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.save:
        session_dir = args.output / f"online_{session_start}"
        session_dir.mkdir(parents=True, exist_ok=True)
        args.output = session_dir

    for path in [args.camera_array, args.model]:
        if not path.exists():
            console.print(f"[bold red]ERROR:[/bold red] File not found: [cyan]{path.resolve()}[/cyan]")
            raise FileNotFoundError(f"File not found:\n{path.resolve()}")

    console.print()
    console.print(Panel(
        "[bold white]LIVE 3D RTMPOSE TRACKING[/bold white]",
        style="bold blue",
        expand=False,
    ))

    setup_table = Table(title="Session Configuration", show_header=True, header_style="bold magenta")
    setup_table.add_column("Setting", style="bold cyan")
    setup_table.add_column("Value", style="white")
    setup_table.add_row("Camera Indices", ", ".join(map(str, args.cams)))
    setup_table.add_row("Camera Array", str(args.camera_array.resolve()))
    setup_table.add_row("Model", str(args.model.resolve()))
    setup_table.add_row("Confidence Threshold", str(args.conf))
    setup_table.add_row("Save Videos + Info", "[bold green]YES[/bold green]" if args.save else "[dim]NO[/dim]")
    if args.save:
        setup_table.add_row("Output Directory", str(args.output.resolve()))
    console.print()
    console.print(setup_table)

    console.print()
    console.print(Rule("[bold cyan]Loading Components[/bold cyan]"))
    console.print(f"  [bold yellow]Camera Array:[/bold yellow] {args.camera_array.name}")
    camera_array = CameraArray.from_toml(args.camera_array)

    console.print(f"  [bold yellow]RTMPose Model:[/bold yellow] {args.model.name}")
    card = ModelCard(
        name="RTMPose-L-WholeBody",
        model_path=args.model,
        format="simcc",
        input_width=288,
        input_height=384,
        confidence_threshold=args.conf,
        point_name_to_id={f"keypoint_{i}": i for i in range(133)},
        wireframe=None
    )
    tracker = OnnxTracker(card)

    console.print()
    console.print(Rule("[bold cyan]Initializing Cameras[/bold cyan]"))
    caps = [cv2.VideoCapture(idx) for idx in args.cams]

    for cap in caps:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    cal_sizes = [camera_array[i].size for i in range(len(args.cams))]
    for i, (cap, (w_cal, h_cal)) in enumerate(zip(caps, cal_sizes)):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_cal)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_cal)

    if not all(cap.isOpened() for cap in caps):
        console.print("[bold red]ERROR:[/bold red] Cannot open one or more webcams")
        raise RuntimeError("Cannot open one or more webcams")

    time.sleep(0.5)

    actual_sizes = [(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) for cap in caps]
    actual_fps = [cap.get(cv2.CAP_PROP_FPS) or 30.0 for cap in caps]

    cam_table = Table(title="Camera Status", show_header=True, header_style="bold magenta")
    cam_table.add_column("Camera", style="bold cyan", justify="center")
    cam_table.add_column("Requested", justify="center")
    cam_table.add_column("Actual", justify="center")
    cam_table.add_column("FPS", justify="center")
    cam_table.add_column("Match", justify="center")

    for i, (idx, (w, h), fps, (w_cal, h_cal)) in enumerate(zip(args.cams, actual_sizes, actual_fps, cal_sizes)):
        match = (w, h) == (w_cal, h_cal)
        match_str = "[bold green]OK[/bold green]" if match else "[bold red]MISMATCH[/bold red]"
        cam_table.add_row(
            f"Cam {idx}",
            f"{w_cal} x {h_cal}",
            f"{w} x {h}",
            f"{fps:.1f}",
            match_str
        )
        if not match:
            console.print(f"  [bold yellow]WARNING:[/bold yellow] Cam {idx} resolution {w}x{h} != calibration {w_cal}x{h_cal}")

    console.print()
    console.print(cam_table)

    if args.save:
        info_path = write_info_toml(args.output, args, cal_sizes, actual_sizes, actual_fps, session_start)
        console.print(f"\n  [dim]Session info:[/dim] [cyan]{info_path}[/cyan]")

    writers = []
    if args.save:
        console.print()
        console.print(Panel(
            f"[bold green]Recording video feeds to:[/bold green]\n[cyan]{args.output.resolve()}[/cyan]",
            title="[bold green]SAVE ENABLED (MP4 + info.toml)[/bold green]",
            border_style="green",
            expand=False,
        ))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for i, idx in enumerate(args.cams):
            w, h = actual_sizes[i]
            fps = actual_fps[i]
            out_path = args.output / f"cam_{idx}.mp4"
            writers.append(ThreadedVideoWriter(out_path, fourcc, fps, (w, h)))

    widths = [int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) for c in caps]
    heights = [int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) for c in caps]
    viewer = PoseViewer(
        window_name="Playback 3D Viewer",
        base_widths=widths,
        base_heights=heights,
        view_3d_base_width=heights[0],
        confidence_threshold=args.conf,
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

    def make_frame_image_points(frame_index, packets):
        rows = []
        for cam_id, packet in enumerate(packets):
            if not packet: continue
            for point_id, (x, y), confidence in zip(packet.keypoint_id, packet.img_loc, packet.confidence):
                if float(confidence) >= args.conf:
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

    input_q = queue.Queue(maxsize=2)
    output_q = queue.Queue(maxsize=2)

    def worker_thread():
        while True:
            item = input_q.get()
            if item is None: break
            frame_idx, frames = item

            packets = [tracker._detect(f, cam_id=i) for i, f in enumerate(frames)]
            xyz_df = triangulate_frame(frame_idx, packets)
            output_q.put((frame_idx, packets, xyz_df))

    threading.Thread(target=worker_thread, daemon=True).start()

    frame_index = 0
    last_result = (0, [None] * len(caps), None)

    try:
        while True:
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    console.print("[bold red]ERROR:[/bold red] Failed to read from webcam. Exiting.")
                    break
                frames.append(frame)

            if len(frames) != len(caps):
                break

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

            if args.save:
                for i, writer in enumerate(writers):
                    writer.write(frames[i])

            if not viewer.show(frames, packets, xyz_df, frame_index):
                break

            frame_index += 1

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Keyboard interrupt detected. Exiting...[/bold yellow]")

    console.print()
    console.print(Rule("[bold cyan]Cleaning Up[/bold cyan]"))
    input_q.put(None)

    for cap in caps:
        cap.release()

    for writer in writers:
        writer.release()

    viewer.destroy()
    ros_publisher.cleanup()
    console.print("  [dim]Cameras, writers, and viewer released.[/dim]")

    if args.save:
        append_summary_toml(args.output, frame_index)

        saved_files = []
        for idx in args.cams:
            saved_files.append(f"  [cyan]•[/cyan] [white]cam_{idx}.mp4[/white]")
        saved_files.append(f"  [cyan]•[/cyan] [white]info.toml[/white]")

        console.print()
        console.print(Panel(
            f"[bold green]Session completed successfully![/bold green]\n\n"
            f"[white]Total Frames:[/white] [bold cyan]{frame_index}[/bold cyan]\n\n"
            f"[white]Saved files:[/white]\n" +
            "\n".join(saved_files) +
            f"\n\n[white]Output:[/white] [bold cyan]{args.output.resolve()}[/bold cyan]",
            title="[bold green]DONE[/bold green]",
            style="bold green",
            expand=False,
        ))
    else:
        console.print()
        console.print(Panel(
            f"[bold green]Session completed.[/bold green]\n\n"
            f"[white]Total Frames Processed:[/white] [bold cyan]{frame_index}[/bold cyan]\n\n"
            f"[dim]Save was disabled - no files written.[/dim]",
            title="[bold green]DONE[/bold green]",
            style="bold green",
            expand=False,
        ))


if __name__ == "__main__":
    main()
