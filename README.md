# SPADE-PGO

A real-time LiDAR SLAM package that integrates LIO and ScanContext.
- **Any LIO algorithm** for odometry (e.g., FAST-LIO)
- **ScanContext** for coarse global localization that can deal with big drifts (place recognition)
- **iSAM2** from GTSAM for pose-graph optimization
- **Multi-session support** for merging maps from multiple drones/flights

## Dependencies

- ROS (geometry_msgs, nav_msgs, sensor_msgs, roscpp, rospy, std_msgs, tf)
- PCL
- OpenMP
- GTSAM
- GeographicLib
- small_gicp
- laspy (Python, for map assembly)
- pypcd (Python, for map assembly)
- open3d (Python, for map assembly)

## Features

1. Strong place recognition and loop closing via ScanContext
2. Modular implementation
3. Complete GPS/GNSS integration with orientation support
4. GeographicLib for easy conversion to various global coordinate systems
5. Multi-session mode for merging multiple drone/flight datasets into a unified map

## Single-Drone Mode

### Running the SLAM

Launch both the pose graph optimizer and FAST-LIO. After building successfully:

```bash
roslaunch spade_pgo graphslam.launch
```

This will use the parameters in `config/pgo.yaml`. Key parameters:
- `ros/gps_topic`: GNSS topic (empty = disabled)
- `ros/orientation_topic`: External orientation topic (empty = disabled)
- `ros/pointcloud_topic`: Point cloud topic from LIO
- `ros/lio_odometry_topic`: Odometry topic from LIO
- `ros/save_directory`: Output directory for poses and scans

Once running, RViz will start. Play back your rosbag and the mapping will proceed.

### Assembling the Map

After the run completes, assemble the point clouds into a LAS file:

```bash
rosrun spade_pgo assemble.py -i ~/save/pointclouds
```

A file called `map.las` will be saved in the input directory.

## Multi-Session Mode

SPADE-PGO supports processing multiple rosbag files from different sessions (e.g., multiple drones or flights) into a unified map. The system:
- Preserves the ScanContext database across sessions for inter-session loop closures
- Uses GNSS to position each session in a common coordinate frame
- Tracks keyframe and session IDs for downstream analysis

### Prerequisites

- GNSS and external orientation must be enabled for reliable multi-session alignment
- Rosbag files should be named to sort in desired processing order (e.g., `01_drone1.bag`, `02_drone2.bag`)
- A config file specifying PGO parameters

### Running Multi-Session Processing

Use the orchestrator script to automatically process multiple bags:

```bash
rosrun spade_pgo multiswarm_orchestrator.py /path/to/bags \
    --config spade_pgo config/multiswarm_mid360.yaml
```
For exmaple:
```bash
rosrun spade_pgo multiswarm_orchestrator.py /home/ros/save/rosbags/spade-swarm/20251217/attempt2 --config spade_pgo config/multiswarm_mid360.yaml
```

The config file (`config/multiswarm_mid360.yaml`) contains combined parameters for both FAST-LIO and SPADE-PGO in a single file. Copy and modify it for different lidar setups.

The orchestrator will:
1. Start roscore if not already running
2. Find all `.bag` files recursively in the specified directory
3. Launch SPADE-PGO with RViz once at startup
4. For each session:
   - Restart FAST-LIO to reset odometry
   - Call the reinit service (for sessions after the first)
   - Play the rosbag at the configured speed
   - Wait for processing to complete

#### Orchestrator Arguments

| Argument | Description |
|----------|-------------|
| `bags_dir` | Directory containing rosbag files (searched recursively) |
| `--config PKG PATH` | Config file (package name and relative path) - passed to launch files |
| `--pgo-launch PKG FILE` | SPADE-PGO launch file (default: `spade_pgo spade_pgo_orchestrated.launch`) |
| `--fastlio-launch PKG FILE` | FAST-LIO launch file (default: `spade_pgo fastlio_orchestrated.launch`) |
| `--fastlio-startup-delay` | Seconds to wait after starting FAST-LIO (default: 3.0) |
| `--bag-rate` | Rosbag playback rate multiplier (default: 3.0) |
| `--min-processing-wait` | Minimum wait after rosbag finishes (default: 5.0s) |
| `--max-processing-wait` | Maximum wait for processing (default: 60.0s) |

#### Example with Custom Settings

```bash
rosrun spade_pgo multiswarm_orchestrator.py /home/ros/data/rosbags \
    --config spade_pgo config/multiswarm_mid360.yaml \
    --bag-rate 5.0 \
    --min-processing-wait 10.0
```

### Assembling Multi-Session Maps

After processing completes, assemble with the `--exclude-start` flag to skip potentially noisy initial keyframes from each session:

```bash
rosrun spade_pgo assemble.py --exclude-start 2 -o /path/to/output.las
```

The output LAS file includes per-point `keyframe` and `session_id` attributes for filtering and analysis.

### Point Cloud Cleanup

If the merger didn't go perfectly, you'll get duplicate trees (since point clouds are slightly offset from one another). To fix this, you can use the grid_cleanup script:
```bash
rosrun spade_pgo grid_cleanup.py --input /path/to/map.las --grid-size 5 --verbose
```

This script removes temporal ghosting / duplication artifacts in a LAS file
produced by SPADE-PGO by keeping only the dominant keyframe cluster per spatial cell. Note however, that you will lose a significant number of the points in the cloud.

CONSTRAINT
----------
Keyframes originating from the SAME drone_id are NEVER allowed to be in the same cluster.

PROCESS
-------
1. Divide XY space into a regular grid (square cells, infinite height)
2. For each cell:
   - Count points per (keyframe, drone_id)
   - Cluster keyframes by temporal proximity with drone_id exclusivity
   - Keep only points belonging to the dominant keyframe cluster
3. Save the cleaned result as a new LAS file


### ROS Service and Topics

Multi-session mode adds:
- **Service** `/spade_pgo/reinit_session`: Reinitialize for a new session
- **Topic** `/spade_pgo/state` (PGOState): Current processing state including queue sizes and session info

## Output Format

The `optimized_poses.txt` file format:
```
# keyframe_id session_id r00 r01 r02 t0 r10 r11 r12 t1 r20 r21 r22 t2
0 0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
1 0 ...
```

Each row contains the keyframe ID, session ID, and 3x4 transformation matrix (row-major).

## Docker

A docker setup is available at [https://github.com/steffanlloyd/spade-pgo-docker](https://github.com/steffanlloyd/spade-pgo-docker)
