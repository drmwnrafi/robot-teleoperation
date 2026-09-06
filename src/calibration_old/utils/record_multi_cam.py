import cv2
import time
import sys
import argparse
import numpy as np
import math
import threading
from queue import Queue
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def draw_readable_text(img, text, position, font_scale=0.6, color=(255, 255, 255), thickness=2):
    """Draws text with a black outline/shadow for maximum readability on any background."""
    # Black shadow/outline (slightly thicker and offset by 1px)
    cv2.putText(img, text, (position[0] + 1, position[1] + 1), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Main colored text
    cv2.putText(img, text, position, 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


class ThreadedVideoWriter:
    def __init__(self, filename, fourcc, fps, frame_size):
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.queue = Queue(maxsize=60)
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
        self.thread.join()
        self.writer.release()


def record_session(cam_indices, recordings_dir, width=None, height=None):
    """
    Runs a standalone multi-camera recording session.
    Returns a dictionary of {cam_idx: Path_to_video} if stopped with 'S', 
    or None if quit with 'Q' or on failure.
    """
    recordings_dir = Path(recordings_dir)
    recordings_dir.mkdir(parents=True, exist_ok=True)

    cams = []
    cam_info = []
    
    console.print("\n[bold cyan]Initializing cameras...[/bold cyan]")
    for idx in cam_indices:
        cam = cv2.VideoCapture(idx)
        if not cam.isOpened():
            console.print(f"[bold red]Error:[/bold red] Failed to open camera {idx}")
            for c in cams:
                c.release()
            return None
        
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        
        if width is not None and height is not None:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        
        cam.set(cv2.CAP_PROP_FPS, 60)
        
        w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cam.get(cv2.CAP_PROP_FPS) or 60.0
        
        console.print(f"  [bold yellow]Camera {idx}:[/bold yellow] {w}x{h} @ {fps:.2f} FPS")
        cams.append(cam)
        cam_info.append({"idx": idx, "w": w, "h": h, "fps": fps})

    is_recording = False
    is_counting_down = False
    countdown_start = 0
    COUNTDOWN_DURATION = 3.0
    writers = []
    
    count = [0] * len(cams)
    fps_vals = [0.0] * len(cams)
    timer = time.perf_counter()

    console.print(
        "\n",
        Panel(
            "[bold green]R[/bold green]  [white]Start recording[/white] [dim](3s countdown)[/dim]\n"
            "[bold yellow]S[/bold yellow]  [white]Stop recording[/white]\n"
            "[bold red]Q[/bold red]  [white]Quit[/white]",
            title="[bold cyan]RECORDING CONTROLS[/bold cyan]",
            border_style="cyan",
            expand=False,
            padding=(1, 3),
        )
    )

    PREVIEW_W, PREVIEW_H = 640, 360
    recorded_files = {}
    WINDOW_NAME = "Multi-Cam Viewer" 

    try:
        while True:
            frames = []
            for i, cam in enumerate(cams):
                ret, frame = cam.read()
                if not ret:
                    console.print(f"[bold red]Camera {cam_indices[i]}:[/bold red] failed to read")
                    continue
                count[i] += 1
                frames.append(frame)
            
            if len(frames) != len(cams):
                continue

            now = time.perf_counter()
            if now - timer >= 1.0:
                elapsed = now - timer
                for i in range(len(cams)):
                    fps_vals[i] = count[i] / elapsed
                    count[i] = 0
                timer = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                console.print("\n[bold red]Quitting...[/bold red]")
                break
            elif key in (ord("r"), ord("R")):
                if not is_recording and not is_counting_down:
                    is_counting_down = True
                    countdown_start = time.perf_counter()
                    console.print("[bold yellow]Countdown started...[/bold yellow]")
            elif key in (ord("s"), ord("S")):
                if is_recording:
                    console.print("\n[bold yellow]Stopping recording...[/bold yellow]")
                    break
                else:
                    console.print("[bold yellow]Not currently recording. Press 'R' to start.[/bold yellow]")

            if is_counting_down:
                remaining = COUNTDOWN_DURATION - (time.perf_counter() - countdown_start)
                if remaining <= 0:
                    is_counting_down = False
                    is_recording = True
                    
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    for info in cam_info:
                        filename = str(recordings_dir / f"cam_{info['idx']}.mp4")
                        recorded_files[info['idx']] = Path(filename)
                        out = ThreadedVideoWriter(filename, fourcc, info["fps"], (info["w"], info["h"]))
                        writers.append(out)
                    console.print(f"[bold green]Recording started![/bold green] Saving to: [cyan]{recordings_dir.resolve()}[/cyan]")
                else:
                    countdown_num = str(int(remaining) + 1)
                    for frame in frames:
                        h, w = frame.shape[:2]
                        # Readable countdown text
                        draw_readable_text(frame, countdown_num, (w//2 - 60, h//2 + 60), font_scale=4.0, color=(0, 0, 255), thickness=6)

            if is_recording:
                for i, out in enumerate(writers):
                    out.write(frames[i])
                
                resized_frames = [cv2.resize(f, (PREVIEW_W, PREVIEW_H)) for f in frames]
                for i, disp in enumerate(resized_frames):
                    h_disp, w_disp = disp.shape[:2]
                    text = f"Cam {cam_indices[i]}: {fps_vals[i]:.1f} FPS"
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Bottom-right positioning (10px padding from edges)
                    x = w_disp - text_w - 10
                    y = h_disp - 10
                    
                    # Readable FPS text
                    draw_readable_text(disp, text, (x, y), font_scale=0.6, color=(255, 0, 255), thickness=2)
                
                num_cams = len(cams)
                cols, rows = (2 if num_cams >= 2 else 1), math.ceil(num_cams / 2)
                while len(resized_frames) < rows * cols:
                    resized_frames.append(np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8))
                
                row_images = [cv2.hconcat(resized_frames[r*cols : (r+1)*cols]) for r in range(rows)]
                combined = cv2.vconcat(row_images)
                
                # REC indicator
                cv2.circle(combined, (30, 30), 15, (0, 0, 255), -1)
                draw_readable_text(combined, "REC", (55, 40), font_scale=1.0, color=(0, 0, 255), thickness=2)
                draw_readable_text(combined, "Press 'S' to stop", (20, combined.shape[0] - 20), font_scale=0.6, color=(0, 255, 255), thickness=2)
                
                cv2.imshow(WINDOW_NAME, combined)
            else:
                resized_frames = [cv2.resize(f, (PREVIEW_W, PREVIEW_H)) for f in frames]
                for i, disp in enumerate(resized_frames):
                    h_disp, w_disp = disp.shape[:2]
                    text = f"Cam {cam_indices[i]}: {fps_vals[i]:.1f} FPS"
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Bottom-right positioning (10px padding from edges)
                    x = w_disp - text_w - 10
                    y = h_disp - 10
                    
                    # Readable FPS text
                    draw_readable_text(disp, text, (x, y), font_scale=0.6, color=(255, 0, 255), thickness=2)
                
                num_cams = len(cams)
                cols, rows = (2 if num_cams >= 2 else 1), math.ceil(num_cams / 2)
                while len(resized_frames) < rows * cols:
                    resized_frames.append(np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8))
                
                row_images = [cv2.hconcat(resized_frames[r*cols : (r+1)*cols]) for r in range(rows)]
                combined = cv2.vconcat(row_images)
                
                if not is_counting_down:
                    # Readable instruction text
                    draw_readable_text(combined, "Press 'R' to record | 'S' to stop | 'Q' to quit", 
                                       (20, combined.shape[0] - 20), font_scale=0.6, color=(0, 255, 255), thickness=2)
                
                cv2.imshow(WINDOW_NAME, combined)

    except KeyboardInterrupt:
        console.print("\n[bold red]Keyboard interrupt detected.[/bold red]")

    console.print("\n[bold cyan]Cleaning up cameras...[/bold cyan]")
    for out in writers:
        out.release()
    for cam in cams:
        cam.release()
    cv2.destroyAllWindows()
    
    if is_recording and all(p.exists() for p in recorded_files.values()):
        return recorded_files
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a standalone multi-camera recording session."
    )
    parser.add_argument(
        "--cams",
        nargs="+",
        type=int,
        required=True,
        help="Camera indices to record from (e.g., 0 1)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs"),
        help="Base directory to save recordings (default: outputs)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Target width (optional, overrides default 2560)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Target height (optional, overrides default 1440)",
    )

    args = parser.parse_args()

    num_cams = len(args.cams)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{num_cams}cam_{timestamp}"

    recordings_dir = args.base_dir / folder_name
    recordings_dir.mkdir(parents=True, exist_ok=True)

    console.print()
    table = Table(title="Recording Session Setup", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Target Cameras", ", ".join(map(str, args.cams)))
    table.add_row("Output Directory", str(recordings_dir.resolve()))
    table.add_row("Target Resolution", f"{args.width or 2560} x {args.height or 1440}")
    console.print(table)
    console.print()

    recorded_files = record_session(
        cam_indices=args.cams,
        recordings_dir=recordings_dir,
        width=args.width,
        height=args.height,
    )

    if recorded_files:
        console.print()
        console.print(Panel(
            "[bold green]Recording completed successfully![/bold green]\n\n"
            "[white]Saved files:[/white]\n"
            + "\n".join(f"  [cyan]•[/cyan] Cam {idx}: [white]{p.name}[/white]" for idx, p in recorded_files.items())
            + f"\n\n[white]Saved at:[/white] [cyan]{recordings_dir.resolve()}[/cyan]",
            title="[bold green]DONE[/bold green]",
            style="bold green",
            expand=False,
        ))
    else:
        console.print()
        console.print(Panel(
            "[bold red]Recording was cancelled or failed.[/bold red]\nNo files were saved.",
            title="[bold red]CANCELLED[/bold red]",
            style="bold red",
            expand=False,
        ))