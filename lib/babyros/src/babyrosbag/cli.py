# cli.py
import argparse
import os
import sys
import time
from babyrosbag.bag import Bag, Time
from loguru import logger
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)
from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel
from rich.table import Table


def discover_topics(timeout: float = 2.0):
    """Discover all active topics across the Zenoh network using liveliness queries."""
    try:
        import babyros
        session = babyros.node.SessionManager.get_session()
        discovered_topics = set()
        replies = session.liveliness().get("**/__liveliness__", timeout=timeout)
        for reply in replies:
            if reply.ok:
                key_expr = str(reply.ok.key_expr)
                if "/__liveliness__" in key_expr:
                    topic = key_expr.split("/__liveliness__")[0]
                    discovered_topics.add(topic)
        return list(discovered_topics)
    except Exception as e:
        logger.error(f"Error discovering topics: {e}")
        return []


# ── record ─────────────────────────────────────────────────────────────
def cmd_record(args):
    try:
        import babyros
    except ImportError:
        print("Error: babyros module not found. Cannot record topics.")
        sys.exit(1)

    console = Console()
    mode = 'a' if args.append else 'w'

    if mode == 'w' and os.path.exists(args.output):
        if not Confirm.ask(
            f"File [bold]{args.output}[/bold] already exists. Overwrite?",
            default=False, console=console,
        ):
            console.print("[bold red]Aborted.[/bold red]")
            sys.exit(0)

    console.print(f"[bold blue]Recording to[/bold blue] {args.output} ([cyan]{'append' if mode == 'a' else 'overwrite'}[/cyan] mode)...")

    topics = args.topic if args.topic else []
    logger.disable("babyros")

    if not topics:
        console.print("[yellow]No topics specified. Discovering active topics...[/yellow]")
        topics = discover_topics(timeout=2.0)
        if not topics:
            console.print("[bold red]No active topics discovered.[/bold red] Please specify topics manually, e.g.:")
            console.print("  [dim]babyrosbag record -O my_bag.db3 -t imu[/dim]")
            sys.exit(1)
        else:
            console.print(f"[green]Discovered {len(topics)} active topic(s):[/green] {', '.join(topics)}")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("[bold]{task.fields[msg_count]}"),
        TextColumn("msgs"),
        console=console,
        refresh_per_second=10,
    )

    with Bag(args.output, mode) as bag:
        subscribers = []
        for t in topics:
            console.print(f"  [green]● Subscribed to:[/green] {t}")

        task_id = progress.add_task("Recording", total=None, msg_count=0)

        def make_callback(topic):
            def callback(msg):
                bag.write(topic, msg)
                progress.update(task_id, advance=1, msg_count=progress.tasks[task_id].completed)
            return callback

        for t in topics:
            sub = babyros.node.Subscriber(t, make_callback(t))
            subscribers.append(sub)

        console.print("\n[bold yellow]Recording...[/bold yellow] Press Ctrl+C to stop.")
        with progress:
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                for sub in subscribers:
                    sub.delete()

        logger.enable("babyros")
        final_count = int(progress.tasks[task_id].completed)
        console.print(f"\n[bold green]✓ Stopped recording.[/bold green] {final_count} messages written to {args.output}")


# ── play ───────────────────────────────────────────────────────────────
def cmd_play(args):
    try:
        import babyros
    except ImportError:
        print("Error: babyros module not found. Cannot play topics.")
        sys.exit(1)

    console = Console()
    console.print(f"[bold blue]Playing[/bold blue] {args.input} at [cyan]{args.rate}x[/cyan] speed...")
    publishers = {}
    logger.disable("babyros")
    console.print("[dim]Opening bag file...[/dim]")

    topics_to_play = args.topic if args.topic else None

    t_start = time.time()
    with Bag(args.input, 'r') as bag:
        load_time = time.time() - t_start

        if getattr(args, 'verbose', False):
            console.print(f"[dim]Bag loaded in {load_time:.3f} seconds.[/dim]")
        else:
            console.print("[dim]Bag opened.[/dim]")

        info = bag.get_type_and_topic_info(topic_filters=topics_to_play)
        total_messages = sum(t.message_count for t in info.topics.values())

        if topics_to_play:
            console.print(f"[dim]Playing selected topics: {', '.join(topics_to_play)}[/dim]")
        console.print(f"[dim]Found {total_messages} messages.[/dim]")

        start_play_time = None
        first_msg_time = None
        msg_count = 0
        last_topic = ""
        interrupted = False

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("msgs"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=10,
        )

        with progress:
            task_id = progress.add_task("Playing", total=total_messages, topic="")
            try:
                for topic, msg, t in bag.read_messages(topics=topics_to_play):
                    if topic not in publishers:
                        publishers[topic] = babyros.node.Publisher(topic=topic)

                    if first_msg_time is None:
                        first_msg_time = t.to_sec()
                        start_play_time = time.time()

                    msg_elapsed = (t.to_sec() - first_msg_time) / args.rate
                    play_elapsed = time.time() - start_play_time

                    if msg_elapsed > play_elapsed:
                        time.sleep(msg_elapsed - play_elapsed)

                    publishers[topic].publish(msg)
                    msg_count += 1
                    last_topic = topic
                    progress.update(task_id, advance=1, topic=f"[dim]{last_topic[:30]}")

            except KeyboardInterrupt:
                interrupted = True

        logger.enable("babyros")
        if interrupted:
            console.print("\n[bold yellow]⚠ Playback interrupted.[/bold yellow]")
            return

    console.print(f"[bold green]✓ Playback finished.[/bold green] {msg_count} messages published.")


# ── info (Beautiful Rich Output) ──────────────────────────────────────
def cmd_info(args):
    console = Console()
    try:
        file_size = os.path.getsize(args.input)
        size_str = (
            f"{file_size / (1024**3):.2f} GiB" if file_size >= 1024**3
            else f"{file_size / (1024**2):.2f} MiB" if file_size >= 1024**2
            else f"{file_size / 1024:.2f} KiB"
        )

        with Bag(args.input, 'r') as bag:
            info = bag.get_type_and_topic_info()
            total_msgs = sum(t.message_count for t in info.topics.values())
            duration = 0.0
            if bag.start_time and bag.end_time:
                duration = bag.end_time.to_sec() - bag.start_time.to_sec()

            # 1. Beautiful Metadata Panel
            meta_text = (
                f"[bold]File:[/bold]      {args.input}\n"
                f"[bold]Bag size:[/bold]  {size_str}\n"
                f"[bold]Storage:[/bold]   sqlite3\n"
                f"[bold]Duration:[/bold]  {duration:.2f}s\n"
                f"[bold]Start:[/bold]     {bag.start_time.to_sec() if bag.start_time else 0.0:.3f}\n"
                f"[bold]End:[/bold]       {bag.end_time.to_sec() if bag.end_time else 0.0:.3f}\n"
                f"[bold]Messages:[/bold]  {total_msgs}"
            )
            console.print(Panel(
                meta_text, 
                title="[bold blue]Bag Information[/bold blue]", 
                border_style="blue", 
                expand=False
            ))

            # 2. Beautiful Topics Table
            table = Table(
                title="[bold magenta]Topic Information[/bold magenta]", 
                show_header=True, 
                header_style="bold cyan",
                border_style="dim"
            )
            table.add_column("Topic", style="green", no_wrap=True)
            table.add_column("Type", style="yellow")
            table.add_column("Count", justify="right", style="magenta")
            table.add_column("Serialization", style="dim")

            for topic, t_info in info.topics.items():
                table.add_row(topic, t_info.msg_type, str(t_info.message_count), "json")

            console.print(table)
            
    except Exception as e:
        console.print(f"[bold red]✗ Error reading bag:[/bold red] {e}")


# ── filter ─────────────────────────────────────────────────────────────
def cmd_filter(args):
    console = Console()
    console.print(f"[bold blue]Filtering[/bold blue] {args.input} [dim]→[/dim] {args.output}...")
    console.print(f"[dim]Expression: {args.expression}[/dim]")

    with Bag(args.input, 'r') as in_bag, Bag(args.output, 'w') as out_bag:
        count = 0
        for topic, msg, t in in_bag.read_messages():
            try:
                if eval(args.expression):
                    out_bag.write(topic, msg, t)
                    count += 1
            except Exception as e:
                console.print(f"[bold red]✗ Error evaluating expression:[/bold red] {e}")
                sys.exit(1)
    console.print(f"[bold green]✓ Filtering complete.[/bold green] Wrote {count} messages.")


# ── main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(prog='babyrosbag', description='BabyROS bag file utilities (SQLite3)')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # record
    p_rec = subparsers.add_parser('record', help='Record topics to a bag file')
    p_rec.add_argument('-t', '--topic', nargs='*', help='Topics to record')
    p_rec.add_argument('-O', '--output', default='output.db3', help='Output bag file')
    p_rec.add_argument('-a', '--append', action='store_true', help='Append to existing file')

    # play
    p_play = subparsers.add_parser('play', help='Play a bag file')
    p_play.add_argument('input', help='Input bag file')
    p_play.add_argument('-r', '--rate', type=float, default=1.0, help='Playback rate multiplier')
    p_play.add_argument('-v', '--verbose', action='store_true', help='Print detailed timing')
    p_play.add_argument('-t', '--topic', nargs='*', help='Only play specified topics')

    # info
    p_info = subparsers.add_parser('info', help='Print bag file information')
    p_info.add_argument('input', help='Input bag file')

    # filter
    p_filt = subparsers.add_parser('filter', help='Filter messages to a new bag')
    p_filt.add_argument('input', help='Input bag file')
    p_filt.add_argument('output', help='Output bag file')
    p_filt.add_argument('expression', help='Python expression (e.g., "topic == \'hand_gestures\'")')

    args = parser.parse_args()

    if args.command == 'record':
        cmd_record(args)
    elif args.command == 'play':
        cmd_play(args)
    elif args.command == 'info':
        cmd_info(args)
    elif args.command == 'filter':
        cmd_filter(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()