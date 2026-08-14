#!/usr/bin/env python3
"""
Assemble point clouds from SPADE-PGO output into a LAS file.

Reads optimized poses and point cloud scans, transforms them to world coordinates,
and saves the merged result as a LAS file with per-point keyframe and drone metadata.

Supports both single-drone and multi-drone (swarm) output formats.

The output is georeferenced to EPSG:5972 (ETRS89 / UTM zone 32N + NN2000 height) using the
datum in gnss_origin.txt. This is not a plain offset: the pose graph works in local ENU, and
the conversion to grid coordinates removes the meridian convergence (~1.54 deg here) and the
grid scale factor. Assembly aborts if the datum is missing or was written at insufficient
precision, unless --allow-ungeoreferenced is given. Each run also writes
<output>.provenance.json recording the datum, flags, keyframes and code version.

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
    --origin LAT,LON,ALT    Override the GNSS datum instead of reading gnss_origin.txt
    --epsg N                Output CRS (default: 5972)
    --allow-ungeoreferenced Write a local-ENU file with no CRS instead of failing

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
import json
import shutil
import datetime
import argparse
import subprocess
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


def load_gnss_origin(filepath):
    """
    Read gnss_origin.txt, written by PoseGraphManager::saveOptimizedPoses.

    Format: 'latitude: <lat> longitude: <lon> altitude: <alt>', where the altitude is the
    ellipsoidal height carried by the NavSatFix message. Returns (lat, lon, h) in degrees
    and metres.
    """
    with open(filepath, "r") as f:
        text = f.read()
    vals = {}
    for key in ("latitude", "longitude", "altitude"):
        m = re.search(key + r"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
        if not m:
            raise RuntimeError("Could not parse '%s' from %s" % (key, filepath))
        vals[key] = float(m.group(1))

    # Guard against the six-significant-figure truncation written by versions of
    # PoseGraphManager.cpp before the std::setprecision(12) fix. At this latitude 1e-4 deg
    # is ~11 m north-south, so a coarsely rounded datum cannot georeference a stem-level map.
    for key in ("latitude", "longitude"):
        s = re.search(key + r"\s*:\s*(-?\d+(?:\.\d+)?)", text).group(1)
        decimals = len(s.split(".")[1]) if "." in s else 0
        if decimals < 7:
            raise RuntimeError(
                "%s in %s has only %d decimal places (%s). This datum was written by an "
                "older spade_pgo without the setprecision(12) fix and is accurate to no "
                "better than a few metres. Re-run the pose graph, or pass --origin "
                "LAT,LON,ALT explicitly if you have the full-precision values."
                % (key, filepath, decimals, s))
    return vals["latitude"], vals["longitude"], vals["altitude"]


def build_georeference(lat0, lon0, h0, epsg_horizontal=25832, epsg_out=5972):
    """
    Build the transform from the pose graph's local ENU frame to a projected CRS.

    The maps come out of the pose graph as local ENU about the first GNSS fix. ENU is NOT
    UTM: at 10.79 deg E the meridian convergence is ~1.54 deg, so simply adding the origin's
    easting/northing to the ENU coordinates rotates the map by that angle -- about 7.6 m of
    error across a 400 m span. The grid scale factor (~0.99970) adds a further 0.03 %.

    Both are removed here with a 2x2 linear map derived from a rigorous PROJ pipeline
    (ENU -> geocentric -> geodetic -> UTM) evaluated at the origin. Compared against the
    rigorous per-point transform over a +/-200 m grid the residual is under 1 mm, so the
    linear form is exact for our purposes and costs one matrix multiply instead of a
    projection call per point.

    Vertically the origin's ellipsoidal height is converted to NN2000 orthometric height via
    the geoid grid and the ENU 'up' component is added to it. Neglected: the geoid slope
    across a site (order 1 cm over 300 m in this region) and the earth-curvature term in the
    topocentric vertical (3 mm at 200 m).

    Returns a dict describing the transform.
    """
    import pyproj
    from pyproj.transformer import TransformerGroup

    src_crs = pyproj.CRS.from_epsg(4937)
    dst_crs = pyproj.CRS.from_epsg(epsg_out)

    # PROJ does not raise when a geoid grid is missing. It quietly substitutes a "ballpark"
    # vertical transformation that returns the ellipsoidal height unchanged -- a ~39 m error
    # in Norway, shipped without any warning. Three independent guards, because getting this
    # wrong is silent and the resulting file looks entirely plausible.
    #
    # Guard 1: no candidate operation may be unavailable for want of a grid.
    tg = TransformerGroup(src_crs, dst_crs, always_xy=True)
    missing = [g.short_name for op in tg.unavailable_operations for g in op.grids
               if not g.available]
    if missing:
        raise RuntimeError(
            "PROJ is missing transformation grid(s) %s, so it would silently fall back to a "
            "ballpark vertical transformation and write ellipsoidal heights labelled as "
            "NN2000 -- an error of roughly 39 m. Rebuild the Docker image (the grid is baked "
            "in; see the Dockerfile) or set PROJ_NETWORK=ON." % ", ".join(sorted(set(missing))))

    tr_v = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    e0, n0, H0 = tr_v.transform(lon0, lat0, h0)

    # Guard 2: the description, once PROJ has resolved an operation. Note this is only
    # populated after a transform has actually been run, and reads "unavailable until
    # proj_trans is called" beforehand -- hence the ordering here, and guards 1 and 3.
    if "ballpark" in (tr_v.description or "").lower():
        raise RuntimeError(
            "PROJ selected a ballpark vertical transformation (%s). Heights would be "
            "ellipsoidal rather than NN2000." % tr_v.description)

    # Guard 3: the arithmetic itself. The geoid-ellipsoid separation in southern Norway is
    # roughly 35-45 m, so a genuine conversion always moves the height by tens of metres. An
    # unchanged height means no grid was applied, whatever PROJ reported.
    if not np.isfinite(H0):
        raise RuntimeError("Vertical transformation returned a non-finite height.")
    if abs(H0 - h0) < 1.0:
        raise RuntimeError(
            "Vertical transformation changed the height by only %.4f m (%.4f -> %.4f). The "
            "geoid-ellipsoid separation here is tens of metres, so the grid was evidently "
            "not applied and the height is still ellipsoidal." % (abs(H0 - h0), h0, H0))

    # --- horizontal: local ENU -> projected easting/northing ---
    pipe = ("+proj=pipeline "
            "+step +inv +proj=topocentric +ellps=GRS80 "
            "+lat_0=%.12f +lon_0=%.12f +h_0=%.6f "
            "+step +inv +proj=cart +ellps=GRS80 "
            "+step +proj=utm +zone=32 +ellps=GRS80" % (lat0, lon0, h0))
    tr_h = pyproj.Transformer.from_pipeline(pipe)

    x0, y0, _ = tr_h.transform(0.0, 0.0, 0.0)
    L = 1000.0
    xe, ye, _ = tr_h.transform(L, 0.0, 0.0)   # ENU east  basis vector
    xn, yn, _ = tr_h.transform(0.0, L, 0.0)   # ENU north basis vector
    A = np.array([[(xe - x0) / L, (xn - x0) / L],
                  [(ye - y0) / L, (yn - y0) / L]], dtype=np.float64)

    scale = float(np.hypot(A[0, 0], A[1, 0]))
    convergence_deg = float(np.degrees(np.arctan2(-A[1, 0], A[0, 0])))

    # Sanity: the two derived bases must agree with the CRS transform of the origin itself.
    if abs(x0 - e0) > 0.01 or abs(y0 - n0) > 0.01:
        raise RuntimeError(
            "Horizontal pipeline and CRS transform disagree at the origin: "
            "(%.4f, %.4f) vs (%.4f, %.4f)" % (x0, y0, e0, n0))

    return {
        "origin_lat": lat0, "origin_lon": lon0, "origin_ellipsoidal_h": h0,
        "epsg": epsg_out, "epsg_horizontal": epsg_horizontal,
        "easting": x0, "northing": y0, "orthometric_h": H0,
        "linear_map": A, "grid_scale_factor": scale,
        "meridian_convergence_deg": -convergence_deg,
        "vertical_operation": tr_v.description,
        # The applied geoid-ellipsoid separation. Recorded because it is the one number
        # that proves a real vertical transformation happened: a ballpark fallback leaves
        # it at exactly zero.
        "geoid_separation_m": h0 - H0,
    }


def write_trajectories(output_path, data_dir, georef):
    """
    Write the keyframe trajectories beside the cloud, in the cloud's own CRS.

    Two files where both pose sets exist: the optimised trajectory and the odometry-only
    one. Their difference is the drift the pose graph removed. Neither was retained by any
    previous run, which is why the drift figures quoted in D4.2 cannot currently be
    reproduced from the released data.
    """
    stem = os.path.splitext(output_path)[0]
    written = []
    for src_name, suffix in (("optimized_poses.txt", ".traj.csv"),
                             ("odometry_poses.txt", ".traj_odom.csv")):
        src = os.path.join(data_dir, src_name)
        if not os.path.exists(src):
            continue
        poses, drone_ids, _, off_x, off_y = load_poses(src)
        if not poses:
            continue
        path = stem + suffix
        with open(path, "w") as f:
            if georef:
                f.write("# CRS EPSG:%d\n" % georef["epsg"])
                f.write("keyframe,session,easting,northing,height\n")
                A = georef["linear_map"]
                for kf in sorted(poses):
                    t = poses[kf][:3, 3]
                    e, n = t[0] + off_x, t[1] + off_y
                    f.write("%d,%d,%.4f,%.4f,%.4f\n" % (
                        kf, drone_ids.get(kf, 0),
                        georef["easting"] + A[0, 0] * e + A[0, 1] * n,
                        georef["northing"] + A[1, 0] * e + A[1, 1] * n,
                        georef["orthometric_h"] + t[2]))
            else:
                f.write("# local ENU, no CRS\n")
                f.write("keyframe,session,x,y,z\n")
                for kf in sorted(poses):
                    t = poses[kf][:3, 3]
                    f.write("%d,%d,%.4f,%.4f,%.4f\n" % (
                        kf, drone_ids.get(kf, 0), t[0] + off_x, t[1] + off_y, t[2]))
        written.append(os.path.basename(path))
        print(f"Trajectory written: {path}")
    return written


def write_provenance(output_path, data_dir, args, georef, kf_files, drone_ids,
                     session_starts, n_points, filter_params):
    """
    Write <output>.provenance.json next to the LAS.

    The existing release clouds carry no record of which bags, flags or code version
    produced them, which is why none of them could be reproduced or georeferenced after the
    fact. Every assembly from here on states its own provenance.
    """
    def git_describe(path):
        # spade-pgo is a submodule: its .git is a file pointing into the parent repo's
        # .git/modules, so rev-parse fails whenever only ros1_ws/ is bind-mounted into the
        # container. Let the caller supply the hash instead of silently recording nothing.
        env_commit = os.environ.get("SPADE_PGO_COMMIT")
        if env_commit:
            return env_commit.strip()
        # safe.directory: the checkout is bind-mounted from the host and owned by a
        # different uid inside the container, which git otherwise refuses to read.
        try:
            return subprocess.check_output(
                ["git", "-c", "safe.directory=*", "-C", path, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    kf_indices = [idx for idx, _ in kf_files]

    prov = {
        "generated_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "generator": "spade_pgo/assemble.py",
        "git_commit": git_describe(script_dir),
        "input_directory": os.path.abspath(data_dir),
        "output": os.path.abspath(output_path),
        "arguments": {k: v for k, v in vars(args).items()},
        "filters": filter_params,
        "keyframes": {
            "count": len(kf_indices),
            "min": min(kf_indices) if kf_indices else None,
            "max": max(kf_indices) if kf_indices else None,
            "session_starts": list(session_starts),
            "drones": sorted(set(drone_ids.values())),
        },
        "n_points": int(n_points),
        "georeference": None,
    }
    if georef:
        prov["georeference"] = {
            "epsg": georef["epsg"],
            "datum_lat": georef["origin_lat"],
            "datum_lon": georef["origin_lon"],
            "datum_ellipsoidal_h": georef["origin_ellipsoidal_h"],
            "origin_easting": georef["easting"],
            "origin_northing": georef["northing"],
            "origin_orthometric_h": georef["orthometric_h"],
            "linear_map_row_major": georef["linear_map"].ravel().tolist(),
            "grid_scale_factor": georef["grid_scale_factor"],
            "meridian_convergence_deg": georef["meridian_convergence_deg"],
            "vertical_operation": georef["vertical_operation"],
            "geoid_separation_m": georef["geoid_separation_m"],
        }

    # The pose graph's own outputs are the expensive artefact and are wiped on the next
    # node launch. Keep a copy of the datum alongside the cloud it produced.
    for name in ("gnss_origin.txt", "optimized_poses.txt", "odometry_poses.txt",
                 "pgo_run.json", "config_used.yaml"):
        src = os.path.join(data_dir, name)
        if os.path.exists(src):
            dst = os.path.splitext(output_path)[0] + "." + name
            shutil.copy2(src, dst)
            prov.setdefault("copied_alongside", []).append(os.path.basename(dst))

    prov_path = os.path.splitext(output_path)[0] + ".provenance.json"
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)
    print(f"Provenance written: {prov_path}")


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
    parser.add_argument("--origin", type=str, default=None,
                        help="Override the GNSS datum, as LAT,LON,ELLIPSOIDAL_ALT. "
                             "Default: read gnss_origin.txt from the input directory")
    parser.add_argument("--epsg", type=int, default=5972,
                        help="Output CRS (default: 5972, ETRS89 / UTM 32N + NN2000 height)")
    parser.add_argument("--poses", type=str, default="optimized_poses.txt",
                        help="Pose file to assemble against, relative to the input "
                             "directory. Pass 'odometry_poses.txt' to build the "
                             "odometry-only cloud for a drift comparison "
                             "(default: optimized_poses.txt)")
    parser.add_argument("--drone", type=int, default=None,
                        help="Keep only keyframes belonging to this drone/session id. "
                             "Lets the per-aircraft clouds of a multi-session run be "
                             "regenerated from the merged pose graph, in the merged "
                             "frame, without re-running it")
    parser.add_argument("--allow-ungeoreferenced", action="store_true",
                        help="Write a local-ENU cloud with no CRS when the datum is missing, "
                             "instead of failing. Produces a file that cannot be combined "
                             "with any other survey; use only for quick inspection.")
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

    # Resolve the georeferencing datum before doing any work, so a missing or truncated
    # origin fails in the first second rather than after an hour of assembly.
    georef = None
    if args.origin:
        lat0, lon0, h0 = [float(v) for v in args.origin.split(",")]
        georef = build_georeference(lat0, lon0, h0, epsg_out=args.epsg)
    else:
        origin_path = os.path.join(data_dir, "gnss_origin.txt")
        if os.path.exists(origin_path):
            lat0, lon0, h0 = load_gnss_origin(origin_path)
            georef = build_georeference(lat0, lon0, h0, epsg_out=args.epsg)
        elif not args.allow_ungeoreferenced:
            raise RuntimeError(
                "No gnss_origin.txt in %s and no --origin given, so the cloud cannot be "
                "georeferenced. Every LAS produced before this check existed was written in "
                "an anonymous local ENU frame and is not recoverable after the fact -- that "
                "is precisely the failure this guard prevents. Pass --origin LAT,LON,ALT, or "
                "--allow-ungeoreferenced if you knowingly want a local-frame file."
                % data_dir)
        else:
            print("WARNING: writing an UNGEOREFERENCED local-ENU cloud with no CRS.")

    if georef:
        print("Georeferencing to EPSG:%d" % georef["epsg"])
        print("  datum          : %.9f, %.9f, %.4f m (ellipsoidal)"
              % (georef["origin_lat"], georef["origin_lon"], georef["origin_ellipsoidal_h"]))
        print("  origin easting : %.4f" % georef["easting"])
        print("  origin northing: %.4f" % georef["northing"])
        print("  origin height  : %.4f m NN2000 (was %.4f m ellipsoidal)"
              % (georef["orthometric_h"], georef["origin_ellipsoidal_h"]))
        print("  convergence    : %.6f deg   grid scale: %.9f"
              % (georef["meridian_convergence_deg"], georef["grid_scale_factor"]))
        print("  geoid sep      : %.4f m applied" % georef["geoid_separation_m"])

    # Load poses
    poses_path = os.path.join(data_dir, args.poses)
    if not os.path.exists(poses_path):
        raise FileNotFoundError(f"Pose file not found: {poses_path}")
    poses, drone_ids, session_starts, offset_x, offset_y = load_poses(poses_path)

    if not poses:
        raise RuntimeError(f"No poses parsed from {poses_path}")

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
                   idx in poses and
                   (args.drone is None or drone_ids.get(idx, 0) == args.drone)]

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

    # Local ENU -> projected coordinates. Done in float64: an easting of ~600 700 m has a
    # float32 spacing of ~0.06 m, which would quantise the map far more coarsely than the
    # 0.001 m LAS scale suggests.
    if georef:
        A = georef["linear_map"]
        e = np_xyz_all[:, 0].astype(np.float64)
        n = np_xyz_all[:, 1].astype(np.float64)
        out_x = georef["easting"] + A[0, 0] * e + A[0, 1] * n
        out_y = georef["northing"] + A[1, 0] * e + A[1, 1] * n
        out_z = georef["orthometric_h"] + np_xyz_all[:, 2].astype(np.float64)
        del e, n
        las_offsets = np.array([georef["easting"], georef["northing"],
                                georef["orthometric_h"]])
    else:
        out_x = np_xyz_all[:, 0].astype(np.float64)
        out_y = np_xyz_all[:, 1].astype(np.float64)
        out_z = np_xyz_all[:, 2].astype(np.float64)
        las_offsets = np.array([0.0, 0.0, 0.0])

    # Save LAS. Version 1.4 so the CRS can be written as a WKT VLR -- the GeoTIFF keys of
    # LAS 1.2 cannot express a compound horizontal+vertical CRS, and a file that claims
    # EPSG:25832 while carrying NN2000 heights is worse than one that claims nothing.
    # The point format is unchanged, so readers see the same record layout as before.
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = las_offsets

    # Add extra dimensions for keyframe and drone ID
    header.add_extra_dim(ExtraBytesParams(name="keyframe", type=np.uint16))
    header.add_extra_dim(ExtraBytesParams(name="drone_id", type=np.uint8))

    if georef:
        import pyproj
        header.add_crs(pyproj.CRS.from_epsg(georef["epsg"]))

    las = laspy.LasData(header)
    las.x = out_x
    las.y = out_y
    las.z = out_z
    las.intensity = np.clip(np_intensity_all[:, 0], 0, 65535).astype(np.uint16)
    las["keyframe"] = np_kf_all[:, 0]
    las["drone_id"] = np_drone_all[:, 0]
    # Mirror the aircraft identity into the standard field as well. drone_id is a LAS extra
    # dimension, which readers that do not understand extra bytes silently drop; 0 in
    # point_source_id conventionally means "this file", so the ids are stored 1-based.
    las.point_source_id = (np_drone_all[:, 0].astype(np.uint16) + 1)
    las.write(output_path)
    print(f"LAS file saved: {output_path}  (points: {np_xyz_all.shape[0]:,})")

    write_trajectories(output_path, data_dir, georef)
    write_provenance(output_path, data_dir, args, georef, kf_files, drone_ids,
                     all_session_starts, np_xyz_all.shape[0],
                     dict(node_skip=NODE_SKIP, near_removal=NEAR_REMOVAL,
                          near_thresh=NEAR_THRESH))


if __name__ == "__main__":
    main()
