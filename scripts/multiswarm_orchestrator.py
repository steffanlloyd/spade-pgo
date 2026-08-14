#!/usr/bin/env python3
"""
Multi-session orchestrator for SPADE-PGO.

This script orchestrates the processing of multiple rosbag files from different sessions
(e.g., multiple drones or multiple flights) into a unified map. It handles:
- Automatic roscore management (starts if not running)
- Recursive discovery of rosbag files
- Preflighting every bag: which GNSS/orientation topics it actually carries
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
    --rviz                  Launch RViz alongside the PGO node (default: off)
    --bag-rate RATE         Rosbag playback rate multiplier (default: 3x)
    --bag-progress          Show rosbag play's progress bar (suppressed by default)
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
    - Every bag is preflighted before anything launches. Missing lidar or IMU aborts the
      run; missing GNSS or orientation only degrades that session, with a warning
    - The PGO node runs continuously; only the reinit service is called between sessions
    - ScanContext database is preserved across sessions for inter-session loop closures
    - roscore is automatically started if not running
"""

import os
import sys
import json
import time
import glob
import signal
import shutil
import datetime
import argparse
import subprocess

import yaml
import rospy
import roslaunch
import rospkg
from spade_pgo.srv import ReinitSession
from spade_pgo.msg import PGOState


# Preflight thresholds, used when the config does not set spade_pgo/preflight
DEFAULT_PREFLIGHT = {
    "min_lidar_rate": 5.0,        # Hz - fatal below
    "min_imu_rate": 25.0,        # Hz - fatal below
    "min_gnss_rate": 0.2,         # Hz - degrade below
    "min_orientation_rate": 1.0,  # Hz - degrade below
}


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
        self.config = None
        self.session_log = []
        self.started_utc = None

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

    def load_config(self):
        """Load and cache the config YAML. Preflight runs before the node, so rospy.get_param
        is not an option."""
        if self.config is not None:
            return self.config
        self.config = {}
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    self.config = yaml.safe_load(f) or {}
            except Exception as e:
                log_warn(f"Could not parse config {self.config_path}: {e}")
        return self.config

    def preflight_bag(self, bag_path):
        """
        Inspect a bag before it is played and decide which topics this session can use.

        Aircraft differ in which GNSS topic they publish, so no single topic name works for
        a whole run. Missing lidar or IMU is fatal (FAST-LIO is tightly coupled and has no
        degraded mode); missing GNSS or orientation degrades the session with a warning.

        Returns a dict of topic choices, measured rates, warnings and errors.
        """
        cfg = self.load_config()
        pgo_cfg = cfg.get("spade_pgo") or {}
        ros_cfg = pgo_cfg.get("ros") or {}
        thresholds = dict(DEFAULT_PREFLIGHT)
        thresholds.update(pgo_cfg.get("preflight") or {})
        common_cfg = cfg.get("common") or {}

        # The IMU name must come from FAST-LIO's common/imu_topic; spade_pgo/ros/imu_topic
        # is a dead key that no nh.param call reads.
        lidar_topic = common_cfg.get("lid_topic", "/livox/lidar")
        imu_topic = common_cfg.get("imu_topic", "/livox/imu")
        gps_topic = ros_cfg.get("gps_topic", "") or ""
        gps_fallback = ros_cfg.get("gps_topic_fallback", "") or ""
        orientation_topic = ros_cfg.get("orientation_topic", "") or ""

        result = {
            "bag": os.path.abspath(bag_path),
            "lidar_topic": lidar_topic,
            "lidar_rate": None,
            "imu_topic": imu_topic,
            "imu_rate": None,
            "gnss_topic_used": "",
            "gnss_source": "none",
            "gnss_rate": None,
            "orientation_topic_used": "",
            "orientation_rate": None,
            "warnings": [],
            "errors": [],
        }

        try:
            import rosbag
        except ImportError as e:
            result["errors"].append(f"cannot import rosbag: {e}")
            return result

        try:
            with rosbag.Bag(bag_path) as bag:
                info = bag.get_type_and_topic_info().topics
                duration = bag.get_end_time() - bag.get_start_time()
        except Exception as e:
            result["errors"].append(f"cannot read bag: {e}")
            return result

        def rate(topic):
            """Hz from the message count, not TopicTuple.frequency, which is None
            for sparse topics."""
            if not topic or topic not in info or duration <= 0:
                return None
            return info[topic].message_count / duration

        # Lidar and IMU: fatal
        for key, topic, threshold, label in (
            ("lidar_rate", lidar_topic, thresholds["min_lidar_rate"], "lidar"),
            ("imu_rate", imu_topic, thresholds["min_imu_rate"], "IMU"),
        ):
            r = rate(topic)
            result[key] = r
            if r is None:
                result["errors"].append(f"no {label} on {topic}")
            elif r < threshold:
                result["errors"].append(
                    f"{label} on {topic} is {r:.2f} Hz, below the {threshold:.2f} Hz minimum")

        # GNSS: primary, then fallback, then disabled
        for topic, source in ((gps_topic, "primary"), (gps_fallback, "fallback")):
            if not topic:
                continue
            r = rate(topic)
            if r is not None and r >= thresholds["min_gnss_rate"]:
                result["gnss_topic_used"] = topic
                result["gnss_source"] = source
                result["gnss_rate"] = r
                break
            result["warnings"].append(
                f"{source} GNSS topic {topic} unusable "
                f"({'absent' if r is None else '%.2f Hz' % r})")
        if not result["gnss_topic_used"]:
            result["warnings"].append(
                "GNSS DISABLED for this session: it will start at the identity pose and be "
                "tied to the rest of the graph only by loop closures")

        # Orientation: present or disabled
        if orientation_topic:
            r = rate(orientation_topic)
            result["orientation_rate"] = r
            if r is not None and r >= thresholds["min_orientation_rate"]:
                result["orientation_topic_used"] = orientation_topic
            else:
                result["warnings"].append(
                    f"orientation topic {orientation_topic} unusable "
                    f"({'absent' if r is None else '%.2f Hz' % r})")
        if orientation_topic and not result["orientation_topic_used"]:
            result["warnings"].append(
                "ORIENTATION DISABLED for this session. Two knock-on effects: the session "
                "anchor's rotation sigma falls back from graph/orientation_noise to "
                "graph/prior_noise_rot, and GNSS factors are no longer added immediately -- "
                "they are buffered until gnss_min_initialization_distance of travel")

        return result

    def log_preflight(self, index, bag_path, pf):
        """Print one bag's preflight result."""
        name = os.path.basename(bag_path)

        def fmt(r):
            return "absent" if r is None else f"{r:.2f} Hz"

        log_info(f"  [{index}] {name}")
        log_info(f"        lidar {pf['lidar_topic']}: {fmt(pf['lidar_rate'])}")
        log_info(f"        imu   {pf['imu_topic']}: {fmt(pf['imu_rate'])}")
        if pf["gnss_topic_used"]:
            log_info(f"        gnss  {pf['gnss_topic_used']}: {fmt(pf['gnss_rate'])} "
                     f"({pf['gnss_source']})")
        else:
            log_info("        gnss  none")
        if pf["orientation_topic_used"]:
            log_info(f"        orient {pf['orientation_topic_used']}: "
                     f"{fmt(pf['orientation_rate'])}")
        else:
            log_info("        orient none")
        for w in pf["warnings"]:
            log_warn(f"  [{index}] {name}: {w}")
        for e in pf["errors"]:
            log_error(f"  [{index}] {name}: {e}")

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

    def launch(self, package, launch_file, pass_config=True, extra_args=None):
        """
        Launch a ROS launch file and return the launch handle.
        If pass_config is True and a config file is set, passes it as 'config:=' argument.
        extra_args is a list of additional 'name:=value' strings.
        Returns None on failure.
        """
        launch_path = self.resolve_launch_file(package, launch_file)
        if not launch_path:
            return None

        # Build launch arguments
        launch_args = []
        if pass_config and self.config_path:
            launch_args.append(f"config:={self.config_path}")
        if extra_args:
            launch_args.extend(extra_args)

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

        # The launch file defaults rviz to true, so always pass it explicitly.
        self.pgo_launch = self.launch(
            self.args.pgo_launch[0], self.args.pgo_launch[1],
            extra_args=[f"rviz:={'true' if self.args.rviz else 'false'}"])
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

    def call_reinit_service(self, gps_topic="", orientation_topic=""):
        """Call the reinit_session service, handing the node this session's topics."""
        log_info(f"Calling reinit_session service (gps='{gps_topic}', "
                 f"orientation='{orientation_topic}')...")

        try:
            rospy.wait_for_service("/spade_pgo/reinit_session", timeout=5.0)
            reinit = rospy.ServiceProxy("/spade_pgo/reinit_session", ReinitSession)
            response = reinit(gps_topic, orientation_topic)

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

        # -q drops the carriage-return progress bar, which otherwise buries the node's own
        # logs. Pass --bag-progress to get it back.
        cmd = ["rosbag", "play", bag_path]
        if not getattr(self.args, "bag_progress", False):
            cmd.append("-q")
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

                    # Must exceed the 0.4 Hz detection period, or this exits between passes.
                    if elapsed_total >= min_wait and elapsed_since_empty >= 10.0:
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

    def process_drone(self, bag_path, drone_index, preflight):
        """Process a single drone's rosbag."""
        log_info(f"\n{'='*60}")
        log_info(f"Processing session {drone_index}: {os.path.basename(bag_path)}")
        log_info(f"{'='*60}\n")

        # Restart FAST-LIO (for all sessions, to reset odometry)
        self.stop_fastlio()
        if not self.start_fastlio():
            return False

        # Reinit for every session, including the first: this is how the node learns which
        # topics this bag carries. The node does not bump the session id on an empty session.
        for w in preflight["warnings"]:
            log_warn(f"session {drone_index}: {w}")
        if not self.call_reinit_service(preflight["gnss_topic_used"],
                                        preflight["orientation_topic_used"]):
            return False

        # Play the rosbag
        if not self.play_rosbag(bag_path):
            return False

        # Wait for processing to complete
        self.wait_for_processing()

        self.session_log.append({
            "session_index": drone_index,
            "bag": os.path.abspath(bag_path),
            "bag_size_bytes": os.path.getsize(bag_path) if os.path.exists(bag_path) else None,
            "finished_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "gnss_topic_used": preflight["gnss_topic_used"],
            "gnss_source": preflight["gnss_source"],
            "gnss_rate_hz": preflight["gnss_rate"],
            "orientation_topic_used": preflight["orientation_topic_used"],
            "orientation_rate_hz": preflight["orientation_rate"],
            "lidar_topic": preflight["lidar_topic"],
            "lidar_rate_hz": preflight["lidar_rate"],
            "imu_topic": preflight["imu_topic"],
            "imu_rate_hz": preflight["imu_rate"],
            "preflight_warnings": preflight["warnings"],
        })

        return True

    def write_run_record(self, success_count, n_bags):
        """
        Record what this run consumed, into the pose graph's save directory.

        The save directory is deleted and recreated on every node launch, and nothing
        previously wrote down which bags produced a given optimized_poses.txt. That is why
        none of the existing release clouds could be traced back to their inputs.
        assemble.py folds this file into the provenance it writes beside each LAS.
        """
        try:
            save_dir = rospy.get_param("spade_pgo/ros/save_directory",
                                       "/home/ros/save/pointclouds/")
        except Exception:
            save_dir = "/home/ros/save/pointclouds/"
        if not os.path.isdir(save_dir):
            log_warn(f"Save directory {save_dir} does not exist; skipping run record")
            return

        def git_commit(path):
            try:
                return subprocess.check_output(
                    ["git", "-C", path, "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL).decode().strip()
            except Exception:
                return None

        record = {
            "generator": "spade_pgo/multiswarm_orchestrator.py",
            "started_utc": self.started_utc,
            "finished_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "git_commit": git_commit(os.path.dirname(os.path.abspath(__file__))),
            "config_file": self.config_path,
            "bag_rate": getattr(self.args, "bag_rate", None),
            "input_directory": os.path.abspath(self.args.bags_dir)
                               if hasattr(self.args, "bags_dir") else None,
            "preflight_thresholds": {
                **DEFAULT_PREFLIGHT,
                **((self.load_config().get("spade_pgo") or {}).get("preflight") or {}),
            },
            "sessions": self.session_log,
            "sessions_successful": success_count,
            "sessions_total": n_bags,
        }

        # The config drives every parameter of the solution; keep a verbatim copy.
        if self.config_path and os.path.exists(self.config_path):
            try:
                shutil.copy2(self.config_path,
                             os.path.join(save_dir, "config_used.yaml"))
                record["config_copied_as"] = "config_used.yaml"
            except Exception as exc:
                log_warn(f"Could not copy config into save directory: {exc}")

        out = os.path.join(save_dir, "pgo_run.json")
        try:
            with open(out, "w") as f:
                json.dump(record, f, indent=2)
            log_success(f"Run record written: {out}")
        except Exception as exc:
            log_warn(f"Could not write run record: {exc}")

    def run(self):
        """Main orchestration loop."""
        # Check/start roscore first
        if not self.check_roscore():
            return False

        rospy.init_node("multiswarm_orchestrator", anonymous=True)
        self.started_utc = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

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

        # Preflight every bag before anything is launched. A missing IMU aborts the whole
        # run, and discovering that on bag 5 wastes twenty minutes of playback.
        log_info("Preflighting bags...")
        preflights = [self.preflight_bag(b) for b in bags]
        for i, (bag, pf) in enumerate(zip(bags, preflights)):
            self.log_preflight(i, bag, pf)
        if any(pf["errors"] for pf in preflights):
            log_error("Preflight failed; aborting before launching anything.")
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
            if self.process_drone(bag, i, preflights[i]):
                success_count += 1
            else:
                log_error(f"Failed to process session {i}")

        self.write_run_record(success_count, len(bags))

        self.stop_fastlio()
        self.stop_pgo_node()
        self.stop_roscore()

        log_info(f"{'='*60}")
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
        "--rviz",
        action="store_true",
        help="Launch RViz alongside the PGO node (default: off)"
    )
    parser.add_argument(
        "--bag-rate",
        type=float,
        default=4,
        help="Rosbag playback rate multiplier (default: normal speed)"
    )
    parser.add_argument(
        "--bag-progress",
        action="store_true",
        help="Show rosbag play's progress bar (suppressed by default)"
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
