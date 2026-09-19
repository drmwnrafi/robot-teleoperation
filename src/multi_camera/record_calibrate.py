import argparse
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

from utils.record_multi_cam import record_session
from utils.calibrate import run_calibration

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Record multi-cam session and auto-calibrate."
    )
    parser.add_argument(
        "cameras",
        nargs="+",
        type=int,
        help="Camera indices (e.g., 0 1 2)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Target width (default: 2560)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Target height (default: 1440)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Frames to skip during calibration point extraction (default: 5)",
    )
    args = parser.parse_args()

    num_cams = len(args.cameras)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{num_cams}cam_{timestamp}"

    PARENT_DIR = Path("outputs")
    SESSION_DIR = PARENT_DIR / session_name

    console.print()
    table = Table(title="Pipeline Setup", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Camera Indices", ", ".join(map(str, args.cameras)))
    table.add_row("Session Name", session_name)
    table.add_row("Session Folder", str(SESSION_DIR.resolve()))
    table.add_row("Target Resolution", f"{args.width or 2560} x {args.height or 1440}")
    table.add_row("Calibration Frame Step", str(args.frame_step))
    console.print(table)
    console.print()

    console.print(Panel(
        f"[bold white]PHASE 1 of 2[/bold white]\n"
        f"[white]Multi-Camera Recording[/white]",
        title="[bold cyan]RECORDING[/bold cyan]",
        border_style="cyan",
        expand=False,
    ))

    recorded_files = record_session(
        cam_indices=args.cameras,
        recordings_dir=SESSION_DIR,
        width=args.width,
        height=args.height,
    )

    if not recorded_files:
        console.print()
        console.print(Panel(
            "[bold red]Recording was cancelled or failed.[/bold red]\n"
            "Calibration will not run.",
            title="[bold red]PIPELINE STOPPED[/bold red]",
            style="bold red",
            expand=False,
        ))
        return

    console.print()
    console.print(Rule("[bold cyan]Transitioning to Calibration Phase[/bold cyan]"))
    console.print()
    console.print(Panel(
        f"[bold white]PHASE 2 of 2[/bold white]\n"
        f"[white]Intrinsic + Extrinsic Calibration[/white]\n\n"
        f"[dim]Folder:[/dim] [cyan]{SESSION_DIR.resolve()}[/cyan]",
        title="[bold cyan]CALIBRATION[/bold cyan]",
        border_style="cyan",
        expand=False,
    ))

    try:
        run_calibration(
            folder=SESSION_DIR,
            frame_step=args.frame_step,
        )
    except Exception as e:
        console.print()
        console.print(Panel(
            f"[bold red]Calibration failed with error:[/bold red]\n{e}\n\n"
            "[white]Please ensure:[/white]\n"
            "  - [cyan]caliscope[/cyan] is installed\n"
            "  - The Charuco board was clearly visible in all cameras",
            title="[bold red]CALIBRATION ERROR[/bold red]",
            style="bold red",
            expand=False,
        ))
        return

    console.print()
    console.print(Panel(
        f"[bold green]Full pipeline completed successfully![/bold green]\n\n"
        f"[white]Recorded files:[/white]\n"
        + "\n".join(f"  [cyan]•[/cyan] Cam {idx}: [white]{p.name}[/white]" for idx, p in recorded_files.items())
        + f"\n\n[white]Calibration output:[/white]\n"
        f"  [cyan]•[/cyan] [white]capture_volume/[/white]\n"
        f"  [cyan]•[/cyan] [white]camera_array_aniposelib.toml[/white]\n\n"
        f"[white]All data saved at:[/white] [bold cyan]{SESSION_DIR.resolve()}[/bold cyan]",
        title="[bold green]PIPELINE COMPLETE[/bold green]",
        style="bold green",
        expand=False,
    ))


if __name__ == "__main__":
    main()