import argparse
import sys
import time
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from loguru import logger


def _get_pid_file():
    """Get path to the router PID file."""
    pid_dir = Path(tempfile.gettempdir()) / "babyros"
    pid_dir.mkdir(exist_ok=True)
    return pid_dir / "router.pid"


def _is_process_running(pid: int) -> bool:
    """Check if a process with given PID is running."""
    try:
        if os.name == 'nt':  # Windows
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(1, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:  # Unix
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError):
        return False


def cmd_topics(args):
    """List all active topics in the Zenoh network."""
    try:
        import babyros
        
        print("Discovering active topics...")
        session = babyros.node.SessionManager.get_session()
        discovered_topics = set()
        
        print(f"Querying for active publishers (timeout: {args.timeout}s)...")
        
        try:
            replies = session.liveliness().get("**/__liveliness__", timeout=args.timeout)
            
            for reply in replies:
                if reply.ok:
                    key_expr = str(reply.ok.key_expr)
                    logger.debug(f"Found liveliness token: {key_expr}")
                    
                    if "/__liveliness__" in key_expr:
                        topic = key_expr.split("/__liveliness__")[0]
                        discovered_topics.add(topic)
        except Exception as e:
            logger.error(f"Error during discovery: {e}")
        
        if not discovered_topics:
            print("\nNo active topics found.")
            print(f"\nTroubleshooting:")
            print(f"  1. Make sure publishers are running in other terminals")
            print(f"  2. Try starting a router: babyros router start")
            print(f"  3. Try increasing timeout: babyros topics -t 5.0")
        else:
            print(f"\nActive topics ({len(discovered_topics)}):")
            for topic in sorted(discovered_topics):
                print(f"  {topic}")
                
    except Exception as e:
        logger.error(f"Error discovering topics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_router_start(args):
    """Start a Zenoh router in the background."""
    pid_file = _get_pid_file()
    
    # Check if already running
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if _is_process_running(pid):
                print(f"✓ Router is already running (PID: {pid})")
                return
            else:
                # Stale PID file
                pid_file.unlink()
        except Exception:
            pid_file.unlink(missing_ok=True)
    
    # Create the router script
    router_script = f"""
import sys
import time
import signal
import zenoh

def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    config = zenoh.Config()
    session = zenoh.open(config)
    print("ROUTER_READY", flush=True)
    while True:
        time.sleep(1)
except Exception as e:
    print(f"ROUTER_ERROR: {{e}}", flush=True)
    sys.exit(1)
"""
    
    print("Starting Zenoh router...")
    
    try:
        # Launch as a detached subprocess
        if os.name == 'nt':
            # Windows: use CREATE_NEW_PROCESS_GROUP for detachment
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
            process = subprocess.Popen(
                [sys.executable, "-c", router_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags,
                cwd=os.getcwd()
            )
        else:
            # Unix: use start_new_session for detachment
            process = subprocess.Popen(
                [sys.executable, "-c", router_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                cwd=os.getcwd()
            )
        
        # Wait a moment for the router to start
        time.sleep(1.5)
        
        # Check if it's still running
        if process.poll() is not None:
            # Process exited - get error
            stderr = process.stderr.read().decode() if process.stderr else ""
            print(f"❌ Router failed to start:\n{stderr}")
            sys.exit(1)
        
        # Save PID
        pid_file.write_text(str(process.pid))
        print(f"✓ Zenoh router started (PID: {process.pid})")
        print(f"  PID file: {pid_file}")
        print(f"\nRouter is listening on default Zenoh endpoints.")
        print(f"Other nodes (WSL, remote machines) can now connect to this router.")
        
    except Exception as e:
        logger.error(f"Failed to start router: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_router_stop(args):
    """Stop the running Zenoh router."""
    pid_file = _get_pid_file()
    
    if not pid_file.exists():
        print("✗ No router is currently running (no PID file found)")
        return
    
    try:
        pid = int(pid_file.read_text().strip())
    except Exception as e:
        print(f"✗ Corrupted PID file: {e}")
        pid_file.unlink(missing_ok=True)
        return
    
    if not _is_process_running(pid):
        print(f"✗ Router process (PID: {pid}) is not running (stale PID file)")
        pid_file.unlink()
        return
    
    print(f"Stopping Zenoh router (PID: {pid})...")
    
    try:
        if os.name == 'nt':
            # Windows: use taskkill for reliable termination
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], 
                         capture_output=True, check=False)
        else:
            # Unix: send SIGTERM
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.5)
            if _is_process_running(pid):
                os.kill(pid, signal.SIGKILL)
        
        pid_file.unlink()
        print(f"✓ Zenoh router stopped")
        
    except Exception as e:
        logger.error(f"Failed to stop router: {e}")
        sys.exit(1)


def cmd_router_status(args):
    """Show the status of the Zenoh router."""
    pid_file = _get_pid_file()
    
    if not pid_file.exists():
        print("✗ Router is not running")
        return
    
    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        print("✗ Router is not running (corrupted PID file)")
        pid_file.unlink(missing_ok=True)
        return
    
    if _is_process_running(pid):
        print(f"✓ Router is running (PID: {pid})")
        print(f"  PID file: {pid_file}")
    else:
        print(f"✗ Router is not running (stale PID file for PID: {pid})")
        pid_file.unlink()


def main():
    parser = argparse.ArgumentParser(prog='babyros', description='BabyROS command-line utilities')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Topics command
    parser_topics = subparsers.add_parser('topics', help='List all active topics')
    parser_topics.add_argument('-t', '--timeout', type=float, default=2.0, 
                              help='Discovery timeout in seconds (default: 2.0)')

    # Router command with subcommands
    parser_router = subparsers.add_parser('router', help='Manage the Zenoh router')
    router_subparsers = parser_router.add_subparsers(dest='router_command', help='Router command')
    
    router_subparsers.add_parser('start', help='Start the Zenoh router in the background')
    router_subparsers.add_parser('stop', help='Stop the running Zenoh router')
    router_subparsers.add_parser('status', help='Show router status')

    args = parser.parse_args()

    if args.command == 'topics':
        cmd_topics(args)
    elif args.command == 'router':
        if args.router_command == 'start':
            cmd_router_start(args)
        elif args.router_command == 'stop':
            cmd_router_stop(args)
        elif args.router_command == 'status':
            cmd_router_status(args)
        else:
            parser_router.print_help()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()