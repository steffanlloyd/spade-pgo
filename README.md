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

## Configuration Parameters

All SPADE-PGO parameters are under the `spade_pgo/` namespace. See `config/multiswarm_mid360.yaml` for a complete example.

### ROS I/O (`ros/`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gps_topic` | string | `""` | GNSS NavSatFix topic. Empty string disables GNSS integration. |
| `orientation_topic` | string | `""` | External orientation topic (ENU frame). Empty string disables. |
| `orientation_msg_type` | string | `"odometry"` | Message type: `"odometry"` or `"quaternion"`. |
| `pointcloud_topic` | string | `"/cloud_registered_body"` | Input point cloud topic (body frame). |
| `lio_odometry_topic` | string | `"/Odometry"` | LIO odometry topic. |
| `save_directory` | string | `"/home/ros/save/pointclouds/"` | Output directory for poses and scans. |

### ICP Registration (`icp/`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | string | `"small_gicp"` | ICP algorithm: `"small_gicp"` or `"small_vgicp"`. |
| `num_kf_accumulate_past` | int | `5` | Number of keyframes to accumulate for the "old" submap. |
| `num_kf_accumulate_now` | int | `2` | Number of keyframes to accumulate for the "current" submap. |
| `fitness_threshold` | double | `0.3` | Maximum ICP MSE to accept a loop closure. |
| `voxel_size` | double | `0.2` | Voxel size for downsampling before ICP [m]. |
| `max_correspondence_distance` | double | `2.0` | Maximum point-to-point distance for ICP correspondences [m]. |
| `max_iterations` | int | `50` | Maximum ICP iterations. |
| `max_height_above_ground` | double | `0` | Filter points above this height relative to the lowest point [m]. Useful for forest environments to exclude canopy. Set to 0 to disable. |
| `save_pointclouds` | bool | `false` | Save point clouds to disk. |
| `publish_pointclouds` | bool | `false` | Publish ICP debug point clouds. |

### Pose Graph (`graph/`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kf_gap_lin` | double | `1.0` | Linear distance between keyframes [m]. |
| `kf_gap_rot` | double | `0.2` | Rotational distance between keyframes [rad]. |
| `lio_noise_lin` | double | `1e-3` | LIO linear noise standard deviation [m]. |
| `lio_noise_rot` | double | `1e-2` | LIO rotational noise standard deviation [rad]. |
| `prior_noise_lin` | double | `5.0` | Prior noise on initial position [m]. |
| `prior_noise_rot` | double | `0.2` | Prior noise on initial orientation [rad]. |
| `use_gnss_altitude` | bool | `true` | Include GNSS altitude in optimization. |
| `gnss_min_initialization_distance` | double | `5.0` | Minimum travel distance before GNSS initializes [m]. |
| `gnss_time_delta` | double | `0.1` | Maximum keyframe-to-fix timestamp gap for a GNSS match [s]. |
| `gps_noise_threshold` | double | `4.0` | Maximum GNSS covariance to accept a fix. |
| `gps_noise_scale` | double | `1.0` | Scaling factor for GNSS XY variance. |
| `gps_noise_z_scale` | double | `100.0` | Scaling factor for GNSS altitude variance. |
| `loop_closure_noise_scale` | double | `1.0` | Scaling factor for loop closure noise. |
| `orientation_noise` | double | `0.1` | External orientation noise standard deviation [rad]. |

### Near-Keyframe Loop Closure (`near_kf/`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable proximity-based loop detection. |
| `distance_threshold` | double | `10.0` | Consider keyframes within this radius [m]. |
| `min_consecutive_kf_distance` | double | `5.0` | Minimum travel distance between tested candidates [m]. |
| `min_kf_seperation` | int | `30` | Minimum keyframe index separation for loop candidates. |

### Visualization (`visualize/`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voxel_size` | double | `0.4` | Voxel size for visualization point cloud downsampling [m]. |

### Extrinsics (`extrinsics/`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lidar_to_body_rpy` | double[3] | `[0, 0, 0]` | LiDAR to body transform as [roll, pitch, yaw] in radians. |

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
