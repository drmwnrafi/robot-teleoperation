import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from caliscope.api import (
    Charuco,
    CharucoTracker,
    CameraArray,
    ConstraintSet,
    extract_image_points,
    extract_image_points_multicam,
    calibrate_intrinsics,
    calibrate_extrinsics,
)
from caliscope.reporting import (
    print_intrinsic_report,
    print_extrinsic_report,
    print_camera_pair_coverage,
)

console = Console()

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def run_calibration(folder: Path, frame_step: int = 5):
    """
    Runs intrinsic and extrinsic calibration on all videos found in the given folder.
    Calibration output is saved directly into the same folder.

    :param folder: Path to the folder containing calibration videos.
    :param frame_step: Number of frames to skip during point extraction.
    """
    folder = folder.resolve()

    if not folder.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {folder}")

    video_files = sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ])

    if not video_files:
        raise FileNotFoundError(f"No video files found in {folder}")

    if len(video_files) < 2:
        console.print("[bold yellow]WARNING:[/bold yellow] Found only 1 video file. "
                       "Extrinsic (multicam) calibration requires at least 2 cameras.")

    video_paths = {i: path for i, path in enumerate(video_files)}
    output_dir = folder

    table = Table(title="Calibration Setup", show_header=True, header_style="bold magenta")
    table.add_column("Camera", style="bold cyan", justify="center")
    table.add_column("File", style="white")
    table.add_column("Size", style="dim", justify="right")

    for cam_id, path in video_paths.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        table.add_row(f"Cam {cam_id}", path.name, f"{size_mb:.1f} MB")

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column(style="bold white")
    info_table.add_column(style="cyan")
    info_table.add_row("Folder (Input/Output)", str(folder))
    info_table.add_row("Frame Step", str(frame_step))
    info_table.add_row("Videos Found", str(len(video_paths)))

    console.print()
    console.print(table)
    console.print()
    console.print(info_table)

    console.print()
    console.print(Panel(
        "[bold white]CALIBRATION PROCESS STARTED[/bold white]",
        style="bold blue",
        expand=False,
    ))

    charuco = Charuco(
        columns=5,
        rows=7,
        board_height=29.7,
        board_width=21.0,
        dictionary="DICT_4X4_50",
        units="cm",
        aruco_scale=0.75,
        square_size_override_cm=5.0,
        inverted=True,
        legacy_pattern=False,
        thickness_cm=0.0,
    )
    tracker = CharucoTracker(charuco)

    cameras = CameraArray.from_video_metadata(video_paths)

    console.print(Rule("[bold cyan]Intrinsic Calibration[/bold cyan]"))

    for cam_id, video in video_paths.items():
        console.print(f"\n[bold yellow]Camera {cam_id}:[/bold yellow] {video.name}")
        points = extract_image_points(video, cam_id, tracker, frame_step=frame_step)
        with console.status(f"[bold green]Calibrating intrinsics for Cam {cam_id}...", spinner="dots"):
            cal = calibrate_intrinsics(points, cameras[cam_id])
        cameras[cam_id] = cal.camera
        print_intrinsic_report(cal)

    console.print(Rule("[bold cyan]Extrinsic Calibration[/bold cyan]"))

    ext_points = extract_image_points_multicam(video_paths, tracker, frame_step=frame_step)
    print_camera_pair_coverage(ext_points)

    constraints = ConstraintSet.from_charuco(charuco)

    with console.status("[bold green]Running bundle adjustment...", spinner="dots"):
        run = calibrate_extrinsics(ext_points, cameras, constraints)

    volume = run.capture_volume
    sync_idx = volume.unique_sync_indices[len(volume.unique_sync_indices) // 2]
    volume = volume.align_to_object(sync_idx)
    volume = volume.grounded(mode="pooled_1st_percentile")
    print_extrinsic_report(volume)

    output_dir.mkdir(parents=True, exist_ok=True)
    volume.save(str(output_dir / "capture_volume"))
    volume.camera_array.to_aniposelib_toml(str(output_dir / "camera_array_aniposelib.toml"))

    console.print()
    console.print(Panel(
        f"[bold green]Calibration saved successfully[/bold green]\n\n"
        f"[white]Output:[/white] [cyan]{output_dir}[/cyan]",
        title="[bold green]DONE[/bold green]",
        style="bold green",
        expand=False,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Caliscope calibration on all videos in a specified folder."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Path to the folder containing the calibration videos. Output will be saved here too.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Number of frames to skip during point extraction (default: 5).",
    )

    args = parser.parse_args()

    run_calibration(
        folder=args.folder,
        frame_step=args.frame_step,
    )
