import os
import cv2
import time
import queue
import threading
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

from caliscope.trackers.onnx_tracker import OnnxTracker
from caliscope.trackers.model_card import ModelCard
from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import ImagePoints

from utils.viewer import PoseViewer

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Replay videos with 3D tracking")
    parser.add_argument(
        "--videos", 
        nargs="+", 
        type=Path, 
        required=True, 
        help="Video files to replay (e.g., cam_0.mp4 cam_1.mp4)"
    )
    parser.add_argument(
        "--camera-array", 
        type=Path, 
        required=True, 
        help="Path to camera_array.toml"
    )
    parser.add_argument(
        "--model", 
        type=Path, 
        default=Path("models/rtmpose_l_coco_wholebody.onnx"), 
        help="Path to RTMPose ONNX model (default: models/rtmpose_l_coco_wholebody.onnx)"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.3, 
        help="Confidence threshold (default: 0.3)"
    )
    parser.add_argument(
        "--speed", 
        type=float, 
        default=1.0, 
        help="Playback speed multiplier (default: 1.0, e.g., 2.0 for 2x speed, 0.5 for half speed)"
    )
    args = parser.parse_args()

    # Ensure speed is valid
    speed_multiplier = max(0.1, args.speed)

    for path in [args.camera_array, args.model] + args.videos:
        if not path.exists(): 
            console.print(f"[bold red]ERROR:[/bold red] File not found: [cyan]{path.resolve()}[/cyan]")
            sys.exit(1)

    console.print()
    console.print(Panel(
        "[bold white]VIDEO PLAYBACK WITH 3D TRACKING[/bold white]",
        style="bold blue",
        expand=False,
    ))

    info_table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    info_table.add_column("Setting", style="bold cyan")
    info_table.add_column("Value", style="white")
    info_table.add_row("Camera Array", str(args.camera_array.resolve()))
    info_table.add_row("Model", str(args.model.resolve()))
    info_table.add_row("Confidence Threshold", f"[bold yellow]{args.conf}[/bold yellow]")
    info_table.add_row("Playback Speed", f"[bold green]{speed_multiplier}x[/bold green]")
    info_table.add_row("Video Files", str(len(args.videos)))
    console.print()
    console.print(info_table)

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
    console.print(Rule("[bold cyan]Opening Videos[/bold cyan]"))
    caps = [cv2.VideoCapture(str(f)) for f in args.videos]
    
    if not all(c.isOpened() for c in caps):
        console.print("[bold red]ERROR:[/bold red] Failed to open one or more videos")
        sys.exit(1)

    widths = [int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) for c in caps]
    heights = [int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) for c in caps]
    fps_list = [c.get(cv2.CAP_PROP_FPS) or 30.0 for c in caps]
    frame_counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    
    base_fps = fps_list[0]
    # Adjust target frame time based on speed multiplier
    target_frame_time = (1.0 / base_fps) / speed_multiplier

    vid_table = Table(title="Video Files", show_header=True, header_style="bold magenta")
    vid_table.add_column("Camera", style="bold cyan", justify="center")
    vid_table.add_column("File", style="white")
    vid_table.add_column("Resolution", justify="center")
    vid_table.add_column("FPS", justify="center")
    vid_table.add_column("Frames", justify="right")
    vid_table.add_column("Duration", justify="right")

    for i, (f, w, h, fps, fc) in enumerate(zip(args.videos, widths, heights, fps_list, frame_counts)):
        duration_sec = fc / fps if fps > 0 else 0
        duration_str = f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s"
        vid_table.add_row(f"Cam {i}", f.name, f"{w}x{h}", f"{fps:.1f}", str(fc), duration_str)

    console.print()
    console.print(vid_table)

    viewer = PoseViewer(
        window_name="Caliscope 3D Viewer",
        base_widths=widths,
        base_heights=heights,
        view_3d_base_width=heights[0],
        confidence_threshold=args.conf,
        skeleton_format="coco_wholebody_133"
    )

    num_cams = len(caps)

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

    console.print()
    effective_fps = base_fps * speed_multiplier
    console.print(Panel(
        f"[bold white]Starting playback with 3D tracking...[/bold white]\n"
        f"Speed: [bold green]{speed_multiplier}x ({effective_fps:.1f} effective FPS)[/bold green]\n"
        "[dim]Press ESC to exit[/dim]",
        title="[bold cyan]PLAYBACK[/bold cyan]",
        border_style="cyan",
        expand=False,
    ))

    frame_index = 0
    last_result = (0, [None] * num_cams, None)
    last_frame_time = time.perf_counter()

    try:
        while True:
            # --- Speed-controlled Throttling ---
            elapsed = time.perf_counter() - last_frame_time
            sleep_time = target_frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_frame_time = time.perf_counter()
            # -----------------------------------

            frames = []
            all_ok = True
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    all_ok = False
                    break
                frames.append(frame)
            
            if not all_ok or len(frames) != num_cams:
                console.print("\n[bold yellow]End of video stream reached.[/bold yellow]")
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
            
            if not viewer.show(frames, packets, xyz_df, frame_index):
                break
                
            frame_index += 1

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Keyboard interrupt detected.[/bold yellow]")

    console.print()
    console.print(Rule("[bold cyan]Cleaning Up[/bold cyan]"))
    input_q.put(None) 
    
    for cap in caps:
        cap.release()
        
    viewer.destroy()
    console.print("  [dim]Videos released and viewer closed.[/dim]")

    console.print()
    console.print(Panel(
        f"[bold green]Playback finished[/bold green]\n\n"
        f"[white]Total Frames Processed:[/white] [bold cyan]{frame_index}[/bold cyan]",
        title="[bold green]DONE[/bold green]",
        style="bold green",
        expand=False,
    ))


if __name__ == "__main__":
    main()