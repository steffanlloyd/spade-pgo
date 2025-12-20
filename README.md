# SPADE-PGO

A real-time LiDAR SLAM package that integrates LIO and ScanContext.
- **Any LIO algorithm** for odometry (e.g., FAST-LIO)
- **ScanContext** for coarse global localization that can deal with big drifts (place recognition)
- **iSAM2** from GTSAM for pose-graph optimization
- **Multi-drone swarm support** for merging maps from multiple drones

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
5. Multi-drone swarm mode for merging multiple drone datasets into a unified map

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

## Multi-Drone Swarm Mode

SPADE-PGO supports processing multiple rosbag files from different drones into a unified map. The system:
- Preserves the ScanContext database across drone sessions for inter-drone loop closures
- Uses GNSS to position each drone in a common coordinate frame
- Tracks keyframe and drone IDs for downstream analysis

### Prerequisites

- GNSS and external orientation must be enabled for reliable multi-drone alignment
- Rosbag files should be named to sort in desired processing order (e.g., `01_drone1.bag`, `02_drone2.bag`)

### Running Multi-Drone Processing

Use the orchestrator script to automatically process multiple bags:

```bash
rosrun spade_pgo multiswarm_orchestrator.py /path/to/bags \
    --pgo-launch spade_pgo graphslam.launch \
    --fastlio-launch fast_lio mapping.launch
```

For example:
```bash
rosrun spade_pgo multiswarm_orchestrator.py /home/ros/save/rosbags/spade-swarm/20251217/attempt2 --pgo-launch spade graphslam_multiswarm_orchestrated.launch --fastlio-launch fast_lio mapping_mid360.launch
```

The orchestrator will:
1. Find all `.bag` files recursively in the specified directory
2. Launch SPADE-PGO once at startup
3. For each drone:
   - Restart FAST-LIO to reset odometry
   - Call the reinit service (for drones after the first)
   - Play the rosbag
   - Wait for processing to complete

#### Orchestrator Arguments

| Argument | Description |
|----------|-------------|
| `bags_dir` | Directory containing rosbag files (searched recursively) |
| `--pgo-launch PKG FILE` | SPADE-PGO launch file (required) |
| `--fastlio-launch PKG FILE` | FAST-LIO launch file |
| `--fastlio-startup-delay` | Seconds to wait after starting FAST-LIO (default: 3.0) |
| `--bag-rate` | Rosbag playback rate multiplier |
| `--min-processing-wait` | Minimum wait after rosbag finishes (default: 5.0s) |
| `--max-processing-wait` | Maximum wait for processing (default: 60.0s) |

### Assembling Multi-Drone Maps

After processing completes, assemble with the `--exclude-start` flag to skip potentially noisy initial keyframes from each drone:

```bash
rosrun spade_pgo assemble.py -i ~/save/pointclouds --exclude-start 2
```

The output LAS file includes per-point `keyframe` and `drone_id` attributes for filtering and analysis.

### ROS Service and Topics

Multi-drone mode adds:
- **Service** `/spade_pgo/reinit_session`: Reinitialize for a new drone session
- **Topic** `/spade_pgo/state` (PGOState): Current processing state including queue sizes and session info

## Output Format

The `optimized_poses.txt` file format:
```
# keyframe_id drone_id r00 r01 r02 t0 r10 r11 r12 t1 r20 r21 r22 t2
0 0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
1 0 ...
```

Each row contains the keyframe ID, drone ID, and 3x4 transformation matrix (row-major).

## Docker

A docker setup is available at [https://github.com/steffanlloyd/spade-pgo-docker](https://github.com/steffanlloyd/spade-pgo-docker)
