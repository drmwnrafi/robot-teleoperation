"""
CLI for babyrosbridge
"""
import argparse
import sys
from rich.console import Console
import rclpy
from babyrosbridge import BabyRosBridge, load_config
from babyros.node import SessionManager  # <-- Import from babyros.node, not babyrosbridge


def cmd_bridge(args):
    """Bridge topics using a YAML configuration file."""
    console = Console()
    
    if not args.yaml:
        console.print("[red]Error: --yaml argument is required[/red]")
        console.print("Usage: babyrosbridge --yaml <config_file>")
        sys.exit(1)
    
    rclpy.init()
    
    console.print(f"Loading configuration from {args.yaml}...")
    ros_to_babyros, babyros_to_ros = load_config(args.yaml)
    
    console.print(f"[green]Found {len(ros_to_babyros)} ROS->babyros mappings[/green]")
    console.print(f"[green]Found {len(babyros_to_ros)} babyros->ROS mappings[/green]")
    
    bridge = BabyRosBridge(ros_to_babyros, babyros_to_ros)
    
    console.print("[green]Bridge started. Press Ctrl+C to stop.[/green]")
    
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_bridge()
        SessionManager.delete(force=True)
        rclpy.shutdown()
        console.print("[green]Bridge stopped.[/green]")


def main():
    parser = argparse.ArgumentParser(
        prog='babyrosbridge',
        description='Bridge between babyros (Zenoh) and ROS 2'
    )
    
    parser.add_argument(
        '--yaml', '-y',
        required=True,
        help='Path to YAML configuration file'
    )
    
    args = parser.parse_args()
    cmd_bridge(args)


if __name__ == '__main__':
    main()