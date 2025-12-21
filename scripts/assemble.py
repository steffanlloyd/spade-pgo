#!/usr/bin/env python3
"""
Assemble point clouds from SPADE-PGO output into a LAS file.

Reads optimized poses and point cloud scans, transforms them to world coordinates,
and saves the merged result as a LAS file with per-point keyframe and drone metadata.

Supports both single-drone and multi-drone (swarm) output formats.

USAGE
-----
    rosrun spade_pgo assemble.py -i ~/save/pointclouds

ARGUMENTS
---------
    -i, --input DIR         Input data directory containing optimized_poses.txt and scans/
    -o, --output FILE       Output LAS file path (default: <input>/map.las)
    -s, --start-frame N     Start keyframe index (inclusive)
    -e, --end-frame N       End keyframe index (inclusive)
    -ir, --ignore-range STR Ignore keyframe ranges, e.g., '100-150' or '50-60,100-150'
    -es, --exclude-start N       Exclude first N keyframes from each drone session (default: 0)
    -ee, --exclude-end N         Exclude last N keyframes from each drone session (default: 0)

EXAMPLES
--------
    # Basic usage (single drone)
    rosrun spade_pgo assemble.py -i ~/save/pointclouds

    # Multi-drone: exclude first 2 keyframes per drone session (recommended)
    rosrun spade_pgo assemble.py -i ~/save/pointclouds --exclude-start 2

    # Process specific frame range
    rosrun spade_pgo assemble.py -i ~/save/pointclouds -s 100 -e 500

    # Ignore problematic frame ranges
    rosrun spade_pgo assemble.py -i ~/save/pointclouds -ir "50-60,200-220"
"""

import os
import re
import argparse
import numpy as np
import open3d as o3d
from numpy import linalg as LA
import laspy
from laspy import ExtraBytesParams
from pypcd import pypcd


def parse_kf_index(filename: str) -> int:
    """Extract keyframe index from filename."""
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else -1


def parse_ignore_ranges(ranges_str: str):
    """Parse ignore ranges like '50-60,100-150' into a list of (start, end) tuples."""
    ranges = []
    if ranges_str:
        for part in ranges_str.split(','):
            part = part.strip()
            if '-' in part:
                a, b = part.split('-')
                ranges.append((int(a), int(b)))
    return ranges


def load_poses(filepath):
    """
    Load poses from optimized_poses.txt.

    Handles both formats:
    - Legacy: 12 floats per line (3x4 matrix row-major)
    - Extended: keyframe_id drone_id + 12 floats (multi-drone format)

    Returns:
        poses: dict mapping keyframe_id -> 4x4 numpy array
        drone_ids: dict mapping keyframe_id -> drone_id
        session_starts: list of keyframe indices where new sessions start
    """
    poses = {}
    drone_ids = {}
    session_starts = []
    offset_x = 0.0
    offset_y = 0.0
    prev_drone_id = None

    with open(filepath, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            # Skip comment lines
            if s.startswith('#'):
                continue

            # Handle offset lines
            if "OFFSET_X" in s.upper() or "OFFSET_Y" in s.upper():
                parts = s.replace(":", " ").replace(",", " ").split()
                for i, tok in enumerate(parts):
                    T = tok.upper()
                    if T == "OFFSET_X" and i + 1 < len(parts):
                        offset_x = float(parts[i + 1])
                    elif T == "OFFSET_Y" and i + 1 < len(parts):
                        offset_y = float(parts[i + 1])
                continue

            nums = s.split()

            # Extended format: keyframe_id drone_id + 12 matrix values
            if len(nums) == 14:
                kf_id = int(nums[0])
                drone_id = int(nums[1])
                matrix_vals = [float(x) for x in nums[2:14]]

                # Track session boundaries
                if prev_drone_id is not None and drone_id != prev_drone_id:
                    session_starts.append(kf_id)
                prev_drone_id = drone_id

                M = np.array(matrix_vals, dtype=np.float64).reshape(3, 4)
                T = np.eye(4, dtype=np.float64)
                T[:3, :4] = M

                poses[kf_id] = T
                drone_ids[kf_id] = drone_id

            # Legacy format: 12 floats only
            elif len(nums) == 12:
                kf_id = len(poses)  # Infer from order
                matrix_vals = [float(x) for x in nums]

                M = np.array(matrix_vals, dtype=np.float64).reshape(3, 4)
                T = np.eye(4, dtype=np.float64)
                T[:3, :4] = M

                poses[kf_id] = T
                drone_ids[kf_id] = 0  # Single drone

    return poses, drone_ids, session_starts, offset_x, offset_y


def main():
    parser = argparse.ArgumentParser(description="Assemble point clouds into LAS file.")
    parser.add_argument("-i", "--input", required=False, default="/home/ros/save/pointclouds", help="Input data directory")
    parser.add_argument("-o", "--output", required=False, help="Output LAS file path")
    parser.add_argument("-s", "--start-frame", type=int, default=None,
                        help="Start keyframe index (inclusive)")
    parser.add_argument("-e", "--end-frame", type=int, default=None,
                        help="End keyframe index (inclusive)")
    parser.add_argument("-ir", "--ignore-range", type=str, default=None,
                        help="Ignore keyframe ranges, e.g., '100-150' or '50-60,100-150'")
    parser.add_argument("-es", "--exclude-start", type=int, default=0,
                        help="Exclude first N keyframes from each drone session (default: 0)")
    parser.add_argument("-ee", "--exclude-end", type=int, default=0,
                        help="Exclude last N keyframes from each drone session (default: 0)")
    args = parser.parse_args()

    # Setup paths
    data_dir = args.input if args.input.endswith('/') else args.input + '/'
    scan_dir = os.path.join(data_dir, "scans")
    output_path = args.output if args.output else os.path.join(data_dir, "map.las")

    # Parameters
    NODE_SKIP = 1
    NEAR_REMOVAL = True
    NEAR_THRESH = 2.0  # meters
    T_body_lidar = np.eye(4)

    # Parse ignore ranges
    ignore_ranges = parse_ignore_ranges(args.ignore_range)

    def is_ignored(idx: int) -> bool:
        for (a, b) in ignore_ranges:
            if a <= idx <= b:
                return True
        return False

    # Load scan file list
    all_files = [f for f in os.listdir(scan_dir) if f.lower().endswith(".pcd")]
    if not all_files:
        raise FileNotFoundError(f"No .pcd files in {scan_dir}")

    kf_files = []
    for f in all_files:
        idx = parse_kf_index(f)
        if idx >= 0:
            kf_files.append((idx, f))
        else:
            print(f"WARNING: couldn't parse index from {f}; skipping.")

    # Load poses
    poses_path = os.path.join(data_dir, "optimized_poses.txt")
    poses, drone_ids, session_starts, offset_x, offset_y = load_poses(poses_path)

    if not poses:
        raise RuntimeError("No poses parsed from optimized_poses.txt")

    # Add implicit first session start
    all_session_starts = [0] + session_starts

    # Determine session end keyframes (last keyframe before next session, or max keyframe for final session)
    max_kf = max(poses.keys()) if poses else 0
    session_ends = [s - 1 for s in session_starts] + [max_kf]

    # Build set of keyframes to exclude (first/last N from each session)
    exclude_set = set()
    if args.exclude_start > 0:
        for session_start in all_session_starts:
            for i in range(args.exclude_start):
                exclude_set.add(session_start + i)
        print(f"Excluding first {args.exclude_start} keyframes from each of {len(all_session_starts)} sessions")

    if args.exclude_end > 0:
        for session_end in session_ends:
            for i in range(args.exclude_end):
                exclude_set.add(session_end - i)
        print(f"Excluding last {args.exclude_end} keyframes from each of {len(session_ends)} sessions")

    if exclude_set:
        print(f"Excluded keyframes: {sorted(exclude_set)}")

    # Apply filters
    kf_files = [(idx, fname) for (idx, fname) in kf_files
                if (args.start_frame is None or idx >= args.start_frame) and
                   (args.end_frame is None or idx <= args.end_frame) and
                   not is_ignored(idx) and
                   idx not in exclude_set and
                   idx in poses]

    if not kf_files:
        raise RuntimeError("No parsable keyframe indices in scans/ filenames after filtering.")

    kf_files.sort(key=lambda x: x[0])

    # Report stats
    unique_drones = set(drone_ids.values())
    print(f"Loaded {len(kf_files)} scans and {len(poses)} poses.")
    print(f"Detected {len(unique_drones)} drone(s): {sorted(unique_drones)}")
    print(f"Session starts: {all_session_starts}")
    if offset_x != 0 or offset_y != 0:
        print(f"Using offsets: ({offset_x}, {offset_y})")
    print(f"Output LAS will be saved to: {output_path}")
    if args.start_frame or args.end_frame:
        print(f"Frame filter: start={args.start_frame}, end={args.end_frame}")

    def pose_for_idx(idx: int) -> np.ndarray:
        T = poses[idx]
        return T @ T_body_lidar

    def o3d_transform_copy(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
        q = o3d.geometry.PointCloud(pcd)
        q.transform(T)
        return q

    # Build global arrays
    kf_chunks = []
    drone_chunks = []
    xyz_chunks = []
    int_chunks = []

    processed = 0
    for idx, fname in kf_files:
        if (processed % NODE_SKIP) != 0:
            processed += 1
            continue

        scan_path = os.path.join(scan_dir, fname)
        if not os.path.exists(scan_path):
            print(f"WARNING: missing file {scan_path}; skipping.")
            processed += 1
            continue

        pcd = o3d.io.read_point_cloud(scan_path)
        if len(pcd.points) == 0:
            print(f"WARNING: empty point cloud {fname}; skipping.")
            processed += 1
            continue

        xyz_local = np.asarray(pcd.points, dtype=np.float32)
        pc = pypcd.PointCloud.from_path(scan_path)
        if 'intensity' in pc.pc_data.dtype.names:
            intens = pc.pc_data['intensity'].astype(np.float32)
        else:
            intens = np.zeros((xyz_local.shape[0],), dtype=np.float32)

        if NEAR_REMOVAL:
            ranges = LA.norm(xyz_local, axis=1)
            eff = np.where(ranges > NEAR_THRESH)[0]
            if eff.size == 0:
                processed += 1
                continue
            pcd = pcd.select_by_index(eff)
            intens = intens[eff]

        T_w_l = pose_for_idx(idx)
        pcd_w = o3d_transform_copy(pcd, T_w_l)
        xyz = np.asarray(pcd_w.points, dtype=np.float32)

        xyz_chunks.append(xyz)
        int_chunks.append(intens.reshape(-1, 1))

        # Per-point metadata
        n_pts = xyz.shape[0]
        kf_chunks.append(np.full((n_pts, 1), idx, dtype=np.uint16))
        drone_chunks.append(np.full((n_pts, 1), drone_ids.get(idx, 0), dtype=np.uint8))

        processed += 1

    if not xyz_chunks:
        raise RuntimeError("No points assembled; check filters and indices.")

    np_xyz_all = np.vstack(xyz_chunks)
    np_intensity_all = np.vstack(int_chunks)
    np_xyz_all[:, 0] += offset_x
    np_xyz_all[:, 1] += offset_y
    np_kf_all = np.vstack(kf_chunks)
    np_drone_all = np.vstack(drone_chunks)

    # Save LAS
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0.0, 0.0, 0.0])

    # Add extra dimensions for keyframe and drone ID
    header.add_extra_dim(ExtraBytesParams(name="keyframe", type=np.uint16))
    header.add_extra_dim(ExtraBytesParams(name="drone_id", type=np.uint8))

    las = laspy.LasData(header)
    las.x = np_xyz_all[:, 0]
    las.y = np_xyz_all[:, 1]
    las.z = np_xyz_all[:, 2]
    las.intensity = np.clip(np_intensity_all[:, 0], 0, 65535).astype(np.uint16)
    las["keyframe"] = np_kf_all[:, 0]
    las["drone_id"] = np_drone_all[:, 0]
    las.write(output_path)
    print(f"LAS file saved: {output_path}  (points: {np_xyz_all.shape[0]:,})")


if __name__ == "__main__":
    main()
