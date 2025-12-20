#!/usr/bin/env python3
"""
Multi-drone swarm orchestrator for SPADE-PGO.

This script orchestrates the processing of multiple rosbag files from different drones
into a unified map. It handles:
- Recursive discovery of rosbag files
- Launching SPADE-PGO node (once at startup)
- Restarting FAST-LIO between drone sessions
- Calling the PGO reinit service for session transitions
- Playing rosbags and waiting for processing completion

USAGE
-----
    rosrun spade_pgo multiswarm_orchestrator.py /path/to/bags \\
        --pgo-launch spade_pgo graphslam.launch \\
        --fastlio-launch fast_lio mapping.launch

ARGUMENTS
---------
    bags_dir                Directory containing rosbag files (searched recursively)
    --pgo-launch PKG FILE   SPADE-PGO launch file (package name and launch file)
    --fastlio-launch PKG FILE
                            FAST-LIO launch file (package name and launch file)
    --fastlio-startup-delay SECONDS
                            Time to wait after starting FAST-LIO (default: 3.0s)
    --bag-rate RATE         Rosbag playback rate multiplier (default: normal speed)
    --min-processing-wait SECONDS
                            Minimum time to wait after rosbag finishes (default: 5.0s)
    --max-processing-wait SECONDS
                            Maximum time to wait for processing (default: 60.0s)

EXAMPLE
-------
    # Process all bags in /data/flight_bags using default launch files
    rosrun spade_pgo multiswarm_orchestrator.py /data/flight_bags \\
        --pgo-launch spade_pgo graphslam.launch \\
        --fastlio-launch fast_lio mapping_velodyne.launch

    # Process at 2x speed with longer wait times
    rosrun spade_pgo multiswarm_orchestrator.py /data/flight_bags \\
        --pgo-launch spade_pgo graphslam.launch \\
        --fastlio-launch fast_lio mapping.launch \\
        --bag-rate 2.0 \\
        --min-processing-wait 10.0

NOTES
-----
    - Rosbag files are sorted alphabetically; name them accordingly (e.g., 01_drone1.bag)
    - FAST-LIO is restarted for each drone to reset odometry
    - The PGO node runs continuously; only the reinit service is called between drones
    - ScanContext database is preserved across sessions for inter-drone loop closures
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


class MultiswarmOrchestrator:
    def __init__(self, args):
        self.args = args
        self.current_state = None
        self.pgo_launch = None
        self.fastlio_launch = None
        self.rosbag_process = None
        self.state_sub = None
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)
        self.rospack = rospkg.RosPack()

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
            rospy.logerr(f"Launch file '{launch_file}' not found in package '{package}'")
            rospy.logerr(f"Searched: {candidates}")
            return None
        except rospkg.ResourceNotFound:
            rospy.logerr(f"Package '{package}' not found")
            return None

    def launch(self, package, launch_file):
        """
        Launch a ROS launch file and return the launch handle.
        Returns None on failure.
        """
        launch_path = self.resolve_launch_file(package, launch_file)
        if not launch_path:
            return None

        rospy.loginfo(f"Launching: {package}/{launch_file}")

        try:
            handle = roslaunch.parent.ROSLaunchParent(self.uuid, [launch_path])
            handle.start()
            return handle
        except Exception as e:
            rospy.logerr(f"Failed to launch {package}/{launch_file}: {e}")
            return None

    def shutdown_launch(self, handle, name="process"):
        """Safely shutdown a roslaunch handle."""
        if handle is not None:
            rospy.loginfo(f"Stopping {name}...")
            try:
                handle.shutdown()
            except Exception as e:
                rospy.logwarn(f"Error stopping {name}: {e}")
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
            rospy.logerr(f"No rosbag files found in {self.args.bags_dir}")
            return []

        rospy.loginfo(f"Found {len(bags)} rosbag files:")
        for i, bag in enumerate(bags):
            rospy.loginfo(f"  [{i}] {bag}")

        return bags

    def state_callback(self, msg):
        """Callback for PGO state updates."""
        self.current_state = msg

    def start_pgo_node(self):
        """Start SPADE-PGO node."""
        if not self.args.pgo_launch:
            rospy.logerr("No SPADE-PGO launch file specified")
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
        rospy.loginfo("Waiting for SPADE-PGO node...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                rospy.wait_for_service("/spade_pgo/reinit_session", timeout=1.0)
                rospy.loginfo("SPADE-PGO node is ready")
                return True
            except rospy.ROSException:
                pass

        rospy.logerr("Timeout waiting for SPADE-PGO node")
        return False

    def start_fastlio(self):
        """Start FAST-LIO."""
        if not self.args.fastlio_launch:
            rospy.logwarn("No FAST-LIO launch file specified, skipping")
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
        rospy.loginfo("Calling reinit_session service...")

        try:
            rospy.wait_for_service("/spade_pgo/reinit_session", timeout=5.0)
            reinit = rospy.ServiceProxy("/spade_pgo/reinit_session", ReinitSession)
            response = reinit()

            if response.success:
                rospy.loginfo(f"Session reinitialized: drone_id={response.drone_id}, "
                              f"next_kf_index={response.next_keyframe_index}")
                return True
            else:
                rospy.logerr(f"Reinit service failed: {response.message}")
                return False
        except rospy.ROSException as e:
            rospy.logerr(f"Failed to call reinit service: {e}")
            return False

    def play_rosbag(self, bag_path):
        """Play a rosbag file and wait for it to finish."""
        rospy.loginfo(f"Playing rosbag: {bag_path}")

        cmd = ["rosbag", "play", bag_path]
        if self.args.bag_rate:
            cmd.extend(["-r", str(self.args.bag_rate)])

        try:
            self.rosbag_process = subprocess.Popen(cmd)
            self.rosbag_process.wait()
            exit_code = self.rosbag_process.returncode
            self.rosbag_process = None

            if exit_code == 0:
                rospy.loginfo("Rosbag playback finished")
                return True
            else:
                rospy.logerr(f"Rosbag playback failed with exit code {exit_code}")
                return False
        except Exception as e:
            rospy.logerr(f"Failed to play rosbag: {e}")
            return False

    def wait_for_processing(self):
        """
        Wait for PGO processing to complete.
        Waits until the loop closure queue is empty OR a minimum time has passed.
        """
        rospy.loginfo("Waiting for PGO processing to complete...")

        min_wait = self.args.min_processing_wait
        max_wait = self.args.max_processing_wait

        start_time = time.time()
        queue_emptied_at = None

        while time.time() - start_time < max_wait:
            if self.current_state is not None:
                queue_size = self.current_state.lc_candidate_queue_size

                if queue_size == 0:
                    if queue_emptied_at is None:
                        queue_emptied_at = time.time()
                        rospy.loginfo("Loop closure queue is empty")

                    elapsed_since_empty = time.time() - queue_emptied_at
                    elapsed_total = time.time() - start_time

                    if elapsed_total >= min_wait and elapsed_since_empty >= 2.0:
                        rospy.loginfo(f"Processing complete after {elapsed_total:.1f}s")
                        return True
                else:
                    queue_emptied_at = None
                    rospy.loginfo_throttle(2.0, f"LC queue size: {queue_size}")

            time.sleep(0.5)

        elapsed = time.time() - start_time
        rospy.logwarn(f"Processing wait timeout after {elapsed:.1f}s")
        return True

    def process_drone(self, bag_path, drone_index):
        """Process a single drone's rosbag."""
        rospy.loginfo(f"\n{'='*60}")
        rospy.loginfo(f"Processing drone {drone_index}: {os.path.basename(bag_path)}")
        rospy.loginfo(f"{'='*60}\n")

        # Restart FAST-LIO (for all drones, to reset odometry)
        self.stop_fastlio()
        if not self.start_fastlio():
            return False

        # Call reinit service (skip for first drone)
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
        rospy.init_node("multiswarm_orchestrator", anonymous=True)

        def signal_handler(sig, frame):
            rospy.loginfo("Shutdown requested...")
            self.stop_fastlio()
            self.stop_pgo_node()
            if self.rosbag_process:
                self.rosbag_process.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        bags = self.find_rosbags()
        if not bags:
            return False

        if not self.start_pgo_node():
            return False

        if not self.wait_for_pgo_node():
            self.stop_pgo_node()
            return False

        self.state_sub = rospy.Subscriber("/spade_pgo/state", PGOState, self.state_callback)

        success_count = 0
        for i, bag in enumerate(bags):
            if self.process_drone(bag, i):
                success_count += 1
            else:
                rospy.logerr(f"Failed to process drone {i}")

        self.stop_fastlio()
        self.stop_pgo_node()

        rospy.loginfo(f"\n{'='*60}")
        rospy.loginfo(f"Processing complete: {success_count}/{len(bags)} drones successful")
        rospy.loginfo(f"{'='*60}\n")

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
        required=True,
        help="SPADE-PGO launch file as: package_name launch_file.launch"
    )
    parser.add_argument(
        "--fastlio-launch",
        nargs=2,
        metavar=("PKG", "FILE"),
        help="FAST-LIO launch file as: package_name launch_file.launch"
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
        type=float,
        default=60.0,
        help="Maximum time to wait for processing (default: 60.0s)"
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
