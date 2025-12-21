#!/usr/bin/env python3
"""
Multi-session orchestrator for SPADE-PGO.

This script orchestrates the processing of multiple rosbag files from different sessions
(e.g., multiple drones or multiple flights) into a unified map. It handles:
- Automatic roscore management (starts if not running)
- Recursive discovery of rosbag files
- Launching SPADE-PGO node (once at startup)
- Restarting FAST-LIO between sessions
- Calling the PGO reinit service for session transitions
- Playing rosbags and waiting for processing completion
- Colored terminal output to differentiate from ROS logs

USAGE
-----
    rosrun spade_pgo multiswarm_orchestrator.py /path/to/bags \\
        --config spade_pgo config/pgo.yaml

ARGUMENTS
---------
    bags_dir                Directory containing rosbag files (searched recursively)
    --config PKG PATH       Config file (package name and relative path), passed to both launch files
    --pgo-launch PKG FILE   SPADE-PGO launch file (default: spade_pgo spade_pgo_orchestrated.launch)
    --fastlio-launch PKG FILE
                            FAST-LIO launch file (default: spade_pgo fastlio_orchestrated.launch)
    --fastlio-startup-delay SECONDS
                            Time to wait after starting FAST-LIO (default: 3.0s)
    --bag-rate RATE         Rosbag playback rate multiplier (default: 3x)
    --min-processing-wait SECONDS
                            Minimum time to wait after rosbag finishes (default: 5.0s)
    --max-processing-wait SECONDS
                            Maximum time to wait for processing (default: none)

EXAMPLE
-------
    # Process all bags with a config file (recommended)
    rosrun spade_pgo multiswarm_orchestrator.py /data/flight_bags \\
        --config spade_pgo config/pgo_multiswarm.yaml

    # Process at 5x speed with custom launch files
    rosrun spade_pgo multiswarm_orchestrator.py /data/flight_bags \\
        --config spade_pgo config/pgo.yaml \\
        --pgo-launch my_pkg my_pgo.launch \\
        --fastlio-launch fast_lio mapping_velodyne.launch \\
        --bag-rate 5.0

NOTES
-----
    - Rosbag files are sorted alphabetically; name them accordingly (e.g., 01_drone1.bag)
    - FAST-LIO is restarted for each session to reset odometry
    - The PGO node runs continuously; only the reinit service is called between sessions
    - ScanContext database is preserved across sessions for inter-session loop closures
    - roscore is automatically started if not running
"""

import os
import sys
import time
import glob
import signal
import argparse
import subprocess

import rospy
import roslaunch
import rospkg
from spade_pgo.srv import ReinitSession
from spade_pgo.msg import PGOState


# ANSI color codes for terminal output
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def log_info(msg):
    """Print info message in cyan to differentiate from ROS logs."""
    print(f"{Colors.CYAN}[ORCHESTRATOR] {msg}{Colors.RESET}")


def log_warn(msg):
    """Print warning message in yellow."""
    print(f"{Colors.YELLOW}[ORCHESTRATOR WARNING] {msg}{Colors.RESET}")


def log_error(msg):
    """Print error message in red."""
    print(f"{Colors.RED}[ORCHESTRATOR ERROR] {msg}{Colors.RESET}")


def log_success(msg):
    """Print success message in green."""
    print(f"{Colors.GREEN}[ORCHESTRATOR] {msg}{Colors.RESET}")


class MultiswarmOrchestrator:
    def __init__(self, args):
        self.args = args
        self.current_state = None
        self.pgo_launch = None
        self.fastlio_launch = None
        self.rosbag_process = None
        self.state_sub = None
        self.roscore_process = None
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)
        self.rospack = rospkg.RosPack()
        self.config_path = None

        # Resolve config file path if specified
        if args.config:
            self.config_path = self.resolve_config_file(args.config[0], args.config[1])
            if self.config_path:
                log_info(f"Using config file: {self.config_path}")
            else:
                log_error(f"Failed to resolve config file: {args.config[0]}/{args.config[1]}")

    def resolve_config_file(self, package, config_path):
        """Resolve a config file path from package name and relative path."""
        try:
            pkg_path = self.rospack.get_path(package)
            full_path = os.path.join(pkg_path, config_path)
            if os.path.exists(full_path):
                return full_path
            log_error(f"Config file not found: {full_path}")
            return None
        except rospkg.ResourceNotFound:
            log_error(f"Package '{package}' not found")
            return None

    def resolve_launch_file(self, package, launch_file):
        """Resolve a launch file path from package name and launch file name."""
        try:
            pkg_path = self.rospack.get_path(package)
            candidates = [
                os.path.join(pkg_path, "launch", launch_file),
                os.path.join(pkg_path, launch_file),
            ]
            for path in candidates:
                if os.path.exists(path):
                    return path
            log_error(f"Launch file '{launch_file}' not found in package '{package}'")
            log_error(f"Searched: {candidates}")
            return None
        except rospkg.ResourceNotFound:
            log_error(f"Package '{package}' not found")
            return None

    def check_roscore(self):
        """Check if roscore is running, start it if not."""
        try:
            rospy.get_master().getSystemState()
            log_info("roscore is already running")
            return True
        except Exception:
            log_warn("roscore is not running, starting it...")
            try:
                self.roscore_process = subprocess.Popen(
                    ['roscore'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(2.0)  # Wait for roscore to start
                rospy.get_master().getSystemState()
                log_success("roscore started successfully")
                return True
            except Exception as e:
                log_error(f"Failed to start roscore: {e}")
                return False

    def stop_roscore(self):
        """Stop roscore if we started it."""
        if self.roscore_process is not None:
            log_info("Stopping roscore...")
            self.roscore_process.terminate()
            self.roscore_process.wait()
            self.roscore_process = None

    def launch(self, package, launch_file, pass_config=True):
        """
        Launch a ROS launch file and return the launch handle.
        If pass_config is True and a config file is set, passes it as 'config:=' argument.
        Returns None on failure.
        """
        launch_path = self.resolve_launch_file(package, launch_file)
        if not launch_path:
            return None

        # Build launch arguments
        launch_args = []
        if pass_config and self.config_path:
            launch_args.append(f"config:={self.config_path}")

        if launch_args:
            log_info(f"Launching: {package}/{launch_file} with args: {launch_args}")
        else:
            log_info(f"Launching: {package}/{launch_file}")

        try:
            # Ensure run_id on parameter server matches what we use
            # Write/update the run_id and configure logging accordingly
            self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, True)
            roslaunch.configure_logging(self.uuid)

            # Use roslaunch API: pass a list of (launch_file, argv) tuples
            # This ensures arguments like 'config:=...' are forwarded correctly.
            roslaunch_files = [(launch_path, launch_args)]
            handle = roslaunch.parent.ROSLaunchParent(self.uuid, roslaunch_files)
            handle.start()
            return handle
        except Exception as e:
            log_error(f"Failed to launch {package}/{launch_file}: {e}")
            return None

    def shutdown_launch(self, handle, name="process"):
        """Safely shutdown a roslaunch handle."""
        if handle is not None:
            log_info(f"Stopping {name}...")
            try:
                handle.shutdown()
            except Exception as e:
                log_warn(f"Error stopping {name}: {e}")
            time.sleep(1.0)

    def find_rosbags(self):
        """
        Recursively find all rosbag files in the specified directory.
        Returns a sorted list of rosbag paths.
        """
        pattern = os.path.join(self.args.bags_dir, "**", "*.bag")
        bags = glob.glob(pattern, recursive=True)
        bags.sort()

        if not bags:
            log_error(f"No rosbag files found in {self.args.bags_dir}")
            return []

        log_info(f"Found {len(bags)} rosbag files:")
        for i, bag in enumerate(bags):
            log_info(f"  [{i}] {bag}")

        return bags

    def state_callback(self, msg):
        """Callback for PGO state updates."""
        self.current_state = msg

    def start_pgo_node(self):
        """Start SPADE-PGO node."""
        if not self.args.pgo_launch:
            log_error("No SPADE-PGO launch file specified")
            return False

        self.pgo_launch = self.launch(self.args.pgo_launch[0], self.args.pgo_launch[1])
        if self.pgo_launch is None:
            return False

        time.sleep(2.0)
        return True

    def stop_pgo_node(self):
        """Stop SPADE-PGO node."""
        self.shutdown_launch(self.pgo_launch, "SPADE-PGO")
        self.pgo_launch = None

    def wait_for_pgo_node(self, timeout=30.0):
        """Wait for the PGO node to be available."""
        log_info("Waiting for SPADE-PGO node...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                rospy.wait_for_service("/spade_pgo/reinit_session", timeout=1.0)
                log_success("SPADE-PGO node is ready")
                return True
            except rospy.ROSException:
                pass

        log_error("Timeout waiting for SPADE-PGO node")
        return False

    def start_fastlio(self):
        """Start FAST-LIO."""
        if not self.args.fastlio_launch:
            log_warn("No FAST-LIO launch file specified, skipping")
            return True

        self.fastlio_launch = self.launch(self.args.fastlio_launch[0], self.args.fastlio_launch[1])
        if self.fastlio_launch is None:
            return False

        time.sleep(self.args.fastlio_startup_delay)
        return True

    def stop_fastlio(self):
        """Stop FAST-LIO."""
        self.shutdown_launch(self.fastlio_launch, "FAST-LIO")
        self.fastlio_launch = None

    def call_reinit_service(self):
        """Call the reinit_session service."""
        log_info("Calling reinit_session service...")

        try:
            rospy.wait_for_service("/spade_pgo/reinit_session", timeout=5.0)
            reinit = rospy.ServiceProxy("/spade_pgo/reinit_session", ReinitSession)
            response = reinit()

            if response.success:
                log_success(f"Session reinitialized: session_id={response.drone_id}, "
                            f"next_kf_index={response.next_keyframe_index}")
                return True
            else:
                log_error(f"Reinit service failed: {response.message}")
                return False
        except rospy.ROSException as e:
            log_error(f"Failed to call reinit service: {e}")
            return False

    def play_rosbag(self, bag_path):
        """Play a rosbag file and wait for it to finish."""
        log_info(f"Playing rosbag: {bag_path}")

        cmd = ["rosbag", "play", bag_path]
        if self.args.bag_rate:
            cmd.extend(["-r", str(self.args.bag_rate)])

        try:
            self.rosbag_process = subprocess.Popen(cmd)
            self.rosbag_process.wait()
            exit_code = self.rosbag_process.returncode
            self.rosbag_process = None

            if exit_code == 0:
                log_success("Rosbag playback finished")
                return True
            else:
                log_error(f"Rosbag playback failed with exit code {exit_code}")
                return False
        except Exception as e:
            log_error(f"Failed to play rosbag: {e}")
            return False

    def wait_for_processing(self):
        """
        Wait for PGO processing to complete.
        Waits until the loop closure queue is empty OR a minimum time has passed.
        """
        log_info("Waiting for PGO processing to complete...")

        min_wait = self.args.min_processing_wait
        max_wait = self.args.max_processing_wait

        start_time = time.time()
        queue_emptied_at = None
        last_queue_log = 0
        
        # Loop until completion; if max_wait is provided, enforce timeout
        while True:
            if self.current_state is not None:
                queue_size = self.current_state.lc_candidate_queue_size

                if queue_size == 0:
                    if queue_emptied_at is None:
                        queue_emptied_at = time.time()
                        log_info("Loop closure queue is empty")

                    elapsed_since_empty = time.time() - queue_emptied_at
                    elapsed_total = time.time() - start_time

                    if elapsed_total >= min_wait and elapsed_since_empty >= 2.0:
                        log_success(f"Processing complete after {elapsed_total:.1f}s")
                        return True
                else:
                    queue_emptied_at = None
                    # Throttle queue size logging
                    if time.time() - last_queue_log >= 2.0:
                        log_info(f"LC queue size: {queue_size}")
                        last_queue_log = time.time()

            time.sleep(0.5)

            # Check timeout only if max_wait is set
            if max_wait is not None:
                if time.time() - start_time >= max_wait:
                    elapsed = time.time() - start_time
                    log_warn(f"Processing wait timeout after {elapsed:.1f}s")
                    return True

        elapsed = time.time() - start_time
        log_warn(f"Processing wait timeout after {elapsed:.1f}s")
        return True

    def process_drone(self, bag_path, drone_index):
        """Process a single drone's rosbag."""
        log_info(f"\n{'='*60}")
        log_info(f"Processing session {drone_index}: {os.path.basename(bag_path)}")
        log_info(f"{'='*60}\n")

        # Restart FAST-LIO (for all sessions, to reset odometry)
        self.stop_fastlio()
        if not self.start_fastlio():
            return False

        # Call reinit service (skip for first session)
        if drone_index > 0:
            if not self.call_reinit_service():
                return False

        # Play the rosbag
        if not self.play_rosbag(bag_path):
            return False

        # Wait for processing to complete
        self.wait_for_processing()

        return True

    def run(self):
        """Main orchestration loop."""
        # Check/start roscore first
        if not self.check_roscore():
            return False

        rospy.init_node("multiswarm_orchestrator", anonymous=True)

        def signal_handler(sig, frame):
            log_info("Shutdown requested...")
            self.stop_fastlio()
            self.stop_pgo_node()
            if self.rosbag_process:
                self.rosbag_process.terminate()
            self.stop_roscore()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        bags = self.find_rosbags()
        if not bags:
            self.stop_roscore()
            return False

        if not self.start_pgo_node():
            self.stop_roscore()
            return False

        if not self.wait_for_pgo_node():
            self.stop_pgo_node()
            self.stop_roscore()
            return False

        self.state_sub = rospy.Subscriber("/spade_pgo/state", PGOState, self.state_callback)

        success_count = 0
        for i, bag in enumerate(bags):
            if self.process_drone(bag, i):
                success_count += 1
            else:
                log_error(f"Failed to process session {i}")

        self.stop_fastlio()
        self.stop_pgo_node()
        self.stop_roscore()

        log_info(f"\n{'='*60}")
        if success_count == len(bags):
            log_success(f"Processing complete: {success_count}/{len(bags)} sessions successful")
        else:
            log_warn(f"Processing complete: {success_count}/{len(bags)} sessions successful")
        log_info(f"{'='*60}\n")

        return success_count == len(bags)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-drone swarm orchestrator for SPADE-PGO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE:
    rosrun spade_pgo multiswarm_orchestrator.py /data/bags \\
        --pgo-launch spade_pgo graphslam.launch \\
        --fastlio-launch fast_lio mapping.launch

NOTES:
    - Rosbag files are discovered recursively and sorted alphabetically
    - Name bags to control processing order (e.g., 01_drone1.bag, 02_drone2.bag)
    - FAST-LIO is restarted for each drone to reset odometry
    - ScanContext database is preserved for inter-drone loop closures
        """
    )
    parser.add_argument(
        "bags_dir",
        help="Directory containing rosbag files (searched recursively)"
    )
    parser.add_argument(
        "--pgo-launch",
        nargs=2,
        metavar=("PKG", "FILE"),
        default=["spade_pgo", "spade_pgo_orchestrated.launch"],
        help="SPADE-PGO launch file (default: spade_pgo spade_pgo_orchestrated.launch)"
    )
    parser.add_argument(
        "--fastlio-launch",
        nargs=2,
        metavar=("PKG", "FILE"),
        default=["spade_pgo", "fastlio_orchestrated.launch"],
        help="FAST-LIO launch file (default: spade_pgo fastlio_orchestrated.launch)"
    )
    parser.add_argument(
        "--fastlio-startup-delay",
        type=float,
        default=3.0,
        help="Time to wait after starting FAST-LIO (default: 3.0s)"
    )
    parser.add_argument(
        "--bag-rate",
        type=float,
        default=3,
        help="Rosbag playback rate multiplier (default: normal speed)"
    )
    parser.add_argument(
        "--min-processing-wait",
        type=float,
        default=5.0,
        help="Minimum time to wait after rosbag finishes (default: 5.0s)"
    )
    parser.add_argument(
        "--max-processing-wait",
        type=lambda s: None if s.lower() == "none" else float(s),
        default=None,
        help="Maximum time to wait for processing (default: none)"
    )
    parser.add_argument(
        "--config",
        nargs=2,
        metavar=("PKG", "PATH"),
        default=["spade_pgo", "config/multiswarm_mid360.yaml"],
        help="Config file as: package_name path/to/config.yaml (passed to both launch files)"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.bags_dir):
        print(f"Error: {args.bags_dir} is not a valid directory")
        sys.exit(1)

    orchestrator = MultiswarmOrchestrator(args)
    success = orchestrator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
