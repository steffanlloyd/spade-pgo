# CLAUDE.md — `spade_pgo`

GTSAM/iSAM2 pose-graph SLAM back end for the SPADE below-canopy drone case. Consumes FAST-LIO2
odometry plus GNSS, external orientation and loop closures; produces optimised keyframe poses and a
merged point cloud. Supports **multi-session** operation: several aircraft (or several flights)
merged into one graph.

Built and run inside the `fastlio-slam:latest` container — see the parent repo's `CLAUDE.md`.

## Source map

| File | Role |
|---|---|
| `src/spade_pgo_node.cpp` | Entry point. Reads ~40 params, wires subscribers, starts 7 worker threads |
| `src/PoseGraphManager.cpp` | Core. Keyframe creation, all factor construction, GNSS handling, session management, iSAM2 optimisation, file output |
| `src/LoopClosureManager.cpp` | Candidate queue, GICP registration, forest point-cloud conditioning |
| `src/NearKFDetector.cpp` | Proximity-based loop-closure candidate search (session-aware) |
| `src/DataBuffer.cpp` | Thread-safe input queues + timestamp synchronisation |
| `src/Visualizer.cpp` | RViz topics |
| `include/scancontext/` | **Vendored third party** (Kim & Kim 2018) — do not restructure |
| `src/bak/` | Archived Vertliner original, not built. Kept for provenance |
| `scripts/multiswarm_orchestrator.py` | Drives multi-bag processing end to end |
| `scripts/assemble.py` | Keyframe PCDs + optimised poses → LAS |
| `scripts/sanity_check.py` | Post-assembly checks: CRS, extent, keyframe jumps, drift, 3D GNSS residual |
| `scripts/gravity_align.py` | Rotates an assembled LAS onto gravity, measured from the source bag's IMU |
| `scripts/trim_cloud.py` | Cuts the sparse XY fringe off a gravity-corrected LAS |
| `scripts/grid_cleanup.py` | Removes duplicate-stem ghosting from a merged LAS |

## Threading model

`spade_pgo_node.cpp` runs seven loops concurrently; keep expensive work off the ingestion path.

| Thread | Rate | Job |
|---|---|---|
| `process_pg` | data-driven | Keyframe creation + factor insertion |
| `process_lcd` | 0.4 Hz | Loop-closure candidate detection |
| `process_icp` | continuous | GICP registration of candidates |
| `process_isam` | 1 Hz | iSAM2 update, writes `optimized_poses.txt` |
| `process_viz_map` | 0.1 Hz | Map publish (heavy) |
| `process_viz_path` | 10 Hz | Path/marker publish |
| `process_state_publisher` | 2 Hz | `/spade_pgo/state` |

Shared state is guarded by four mutexes in `PoseGraphManager` (`kf_mtx_`, `kf_updated_mtx_`,
`graph_mtx_`, `current_pose_mtx_`). Lock ordering matters — `dataAvailable(bool lock_mutex)` exists
precisely to allow calls from a context that already holds `buffer_mutex_`. Do ICP outside the
candidate mutex (`LoopClosureManager::processCandidateQueue` is written that way deliberately).

## Factor graph

- **Nodes**: keyframes, created after `kf_gap_lin` (1.0 m) or `kf_gap_rot` (0.3 rad) of motion.
  Intermediate scans are accumulated into the keyframe cloud, then voxelised at
  `min(icp.voxel_size, sc.voxel_size)`.
- **Odometry**: `BetweenFactor<Pose3>` from FAST-LIO relative pose, tight noise (σ ≈ 5e-4 m).
- **GNSS**: `GPSFactor` (position only) or `PriorFactor<Pose3>` (when orientation is available),
  noise from the `NavSatFix` covariance × `gps_noise_scale` (× `gps_noise_z_scale` vertically).
  Buffered until `gnss_min_initialization_distance` of travel unless external orientation is
  configured, which bypasses the delay.
- **Loop closure**: `BetweenFactor<Pose3>` with a **robust Cauchy** m-estimator; variance = ICP
  fitness × `loop_closure_noise_scale²`.
- **Prior**: only when GNSS is unavailable at session start.

`optimizeGraph()` pushes the accumulated graph into iSAM2, clears it, and re-reads all estimates.
`triggerExtraOptimization()` forces five extra sweeps after a loop closure or GNSS initialisation.

## Multi-session semantics

One graph, one continuous keyframe index space, all sessions. There is **no** odometry edge between
sessions — they are tied only by GNSS priors and inter-session loop closures.

`/spade_pgo/reinit_session` (srv `ReinitSession`) increments `current_session_id_`, clears buffers,
resets the odometry reference and sets `awaiting_session_init_`. **The ScanContext database and the
existing graph are preserved** — that is the point.

On the next `processData()` the session start pose is built directly in the global frame: orientation
from the external orientation topic, position by converting the first GNSS fix through the
*existing* `GeographicLib::LocalCartesian` (datum **not** reset for sessions after the first).

`NearKFDetector` applies `min_kf_seperation` **only within a session**, so cross-aircraft keyframe
pairs are always eligible — this is what stitches sessions together. It logs candidates as
`Intra-session` / `Inter-session`.

Multi-session without both GNSS and external orientation produces poor results (the code warns);
the new session starts at identity.

## Forest-specific ICP conditioning

`LoopClosureManager::preprocessPointCloud_` filters submaps before registration:

- **Canopy exclusion** — ground estimated as the 2nd percentile of z, everything above
  `ground + max_height_above_ground` (7 m) dropped. Canopy returns are viewpoint-dependent and
  wind-affected, so they hurt registration.
- **Radius gating** — points beyond `max_radius_from_keyframe` (20 m) dropped.

These affect ICP only; saved keyframe clouds keep everything.

Acceptance criteria for a loop closure: ≥100 points after filtering, converged,
`num_inliers > 0`, inlier ratio ≥ `min_inlier_ratio`, fitness (`error / num_inliers`) <
`fitness_threshold`. Each pair is tested at most once (`tested_candidates_`).

## Outputs

Into `ros/save_directory`, **which is deleted and recreated on every launch**:

- `scans/NNNNNN.pcd` — per-keyframe cloud, body frame
- `optimized_poses.txt` — `keyframe_id session_id` + 3×4 row-major pose, rewritten each optimisation
- `laser_timestamps.txt`, `gnss_origin.txt`

`assemble.py` → LAS 1.2 point format 3 with extra dims `keyframe` (uint16) and `drone_id` (uint8);
drops returns within 2 m of the sensor; `--exclude-start 2` skips unsettled session-start keyframes.
Coordinates are **local ENU about the datum in `gnss_origin.txt`, not UTM**.

## Conventions

- C++17, poses are `Eigen::Isometry3d` throughout (an earlier `Pose6D` struct was removed — don't
  reintroduce Euler-angle poses).
- Parameters live in `PGOParams.hpp` as nested structs, read once in `main()` with defaults; mirror
  any new parameter into `config/multiswarm_mid360.yaml` and the README table.
- `params->useGNSS()` / `useExternalOrientation()` are driven by whether the topic string is empty —
  that is how features are switched off.
- Loop-closure and ICP diagnostics go through `ROS_INFO`/`ROS_WARN` with a `[Loop Closure]` prefix
  and stable wording; log-grepping those lines is the current substitute for a metrics harness.

## Gotchas

- **ScanContext is disabled** (`sc.enabled: false`) in every operational config. Loop closure is
  proximity-based. The code path still exists and is compiled.
- `gps_noise_threshold: 1e9` in the multiswarm config effectively disables the covariance gate —
  all fixes enter, weighting does the work.
- `session_id`/`drone_id` is a `uint8` assigned by processing order (bags are sorted alphabetically
  by the orchestrator), not a hardware identity.
- `grid_cleanup.py` typically discards ~1/3 of points. It hides ghosting; it does not fix
  registration. Never run it before computing an inter-aircraft agreement metric. It is **not**
  the same thing as `trim_cloud.py`, which cuts only the outer XY fringe and keeps or drops a
  cell whole, never thinning the interior.
- **Post-assembly order is fixed: `gravity_align.py` then `trim_cloud.py`.** The trim projects
  to XY, so trimming a tilted cloud cuts the wrong footprint. `gravity_align.py` rotates the
  LAS, the odometry twin and the trajectory CSVs, but deliberately **not** `optimized_poses.txt`
  / `odometry_poses.txt`: those are in the pipeline's local ENU frame, not the projected CRS the
  rotation was measured in, and they are what a cloud is regenerated from.
- `gravity_align.py` refuses above `--max-tilt` (30° default) rather than silently applying a
  large correction, and its `between-window spread` p95 is the number to look at — a large p95
  against a small median means the IMU windows disagree and the estimate is contaminated.
- README says `--bag-rate` defaults to 3.0; the script's default is 4.0.
- Almost no tests, no CI. Validate changes by re-running a known bag set and comparing
  `optimized_poses.txt` trajectories and loop-closure counts. The one exception is
  `test/test_voxelizer.cpp`, a standalone regression test for the map downsampler — not wired
  into `CMakeLists.txt`; build and run it by hand, see its header.
- **The map publisher uses `pcl::octree::OctreePointCloudVoxelCentroid`, not `pcl::VoxelGrid`.**
  `VoxelGrid` sizes a dense `nx*ny*nz` index from the bounding box and, above `INT32_MAX`,
  returns the cloud *unfiltered* — one stray return is enough to hand RViz the full-resolution
  map. `ApproximateVoxelGrid` avoids that but merges only within a fixed 512-slot cache, so it
  needs spatially coherent input; `assembleGlobalPointCloud()` concatenates keyframe by
  keyframe, and it measured 1.26x reduction against the exact 12.1x. Do not "simplify" this
  back to either.
- The same `VoxelGrid` overflow still affects the **save** path (`PoseGraphManager.cpp:63`, leaf
  `min(icp.voxel_size, sc.voxel_size)` = 0.05 m), so every `scans/*.pcd` is written at full
  resolution. Known, deferred: it costs ICP runtime, not released geometry.
