#!/usr/bin/env python3
"""
Correct the gravity alignment of an assembled LAS using the IMU in its source rosbag.

The maps come out of the pose graph tilted when FAST-LIO's initial attitude is wrong or the
lidar-to-body extrinsic does not match the physical mount. A stationary or hovering IMU
measures gravity directly, which is the quantity we want; a ground-plane or PCA fit measures
terrain, which is only vertical on flat ground.

Method:
  1. Find IMU windows where the platform is not accelerating -- |a| close to g and low
     angular rate. Pre-takeoff and steady hover both qualify.
  2. For each window, take the mean specific force as "up" in body frame and rotate it into
     the map frame with the keyframe pose nearest that timestamp.
  3. Take the median of those directions, and rotate it onto true up.

The correction is the minimal rotation between two vectors, so its axis lies in the
horizontal plane and heading is preserved exactly.

Output is a new <stem>_rotation_fixed.las plus a .gravity_align.json record. The input is
never modified.
"""

import argparse
import json
import os
import sys

import numpy as np


G = 9.80665
CHUNK = 2_000_000


# --------------------------------------------------------------------------- measurement

def find_imu_topic(bag, explicit=None):
    """Pick the IMU topic, preferring an explicit name."""
    info = bag.get_type_and_topic_info().topics
    if explicit:
        if explicit not in info:
            raise RuntimeError("IMU topic %s not in bag (have: %s)"
                               % (explicit, ", ".join(sorted(info))))
        return explicit
    imu = [t for t, m in info.items() if m.msg_type == "sensor_msgs/Imu"]
    if not imu:
        raise RuntimeError("No sensor_msgs/Imu topic in bag")
    # Prefer the lidar's own IMU: it is the frame FAST-LIO integrates.
    for pref in ("/livox/imu",):
        if pref in imu:
            return pref
    return sorted(imu, key=lambda t: -info[t].message_count)[0]


def read_imu(bag_path, imu_topic=None):
    """Return (t, accel Nx3, gyro Nx3) from the bag, in body frame."""
    import rosbag
    with rosbag.Bag(bag_path, "r") as bag:
        topic = find_imu_topic(bag, imu_topic)
        t, a, w = [], [], []
        for _, msg, _ in bag.read_messages(topics=[topic]):
            t.append(msg.header.stamp.to_sec())
            a.append((msg.linear_acceleration.x, msg.linear_acceleration.y,
                      msg.linear_acceleration.z))
            w.append((msg.angular_velocity.x, msg.angular_velocity.y,
                      msg.angular_velocity.z))
    if not t:
        raise RuntimeError("IMU topic %s carried no messages" % topic)
    a = np.asarray(a)
    # The Livox driver publishes acceleration in units of g, not m/s^2.
    scale = G if abs(np.median(np.linalg.norm(a, axis=1)) - 1.0) < 0.3 else 1.0
    return topic, np.asarray(t), a * scale, np.asarray(w), scale


def rpy_to_matrix(rpy):
    """Rz(yaw) Ry(pitch) Rx(roll), matching spade_pgo_node.cpp:327."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_extrinsic(pgo_dir, override=None):
    """Rotation body<-lidar. The Livox IMU sits in the lidar, so gravity is measured there."""
    if override is not None:
        return rpy_to_matrix(override), list(override)
    path = os.path.join(pgo_dir, "config_used.yaml")
    if not os.path.exists(path):
        return np.eye(3), None
    import yaml
    cfg = yaml.safe_load(open(path)) or {}
    rpy = (cfg.get("spade_pgo", {}).get("extrinsics", {}) or {}).get("lidar_to_body_rpy")
    if not rpy or len(rpy) != 3:
        return np.eye(3), None
    return rpy_to_matrix([float(v) for v in rpy]), [float(v) for v in rpy]


def load_poses(pgo_dir):
    """Return (kf_times, rotations Nx3x3) with rotations map<-body."""
    poses_path = os.path.join(pgo_dir, "optimized_poses.txt")
    times_path = os.path.join(pgo_dir, "laser_timestamps.txt")
    rots = []
    for line in open(poses_path):
        p = line.split()
        if len(p) < 14 or p[0].startswith("#"):
            continue
        rots.append(np.asarray([float(x) for x in p[2:14]]).reshape(3, 4)[:, :3])
    times = np.asarray([float(x) for x in open(times_path) if x.strip()])
    n = min(len(rots), len(times))
    if n == 0:
        raise RuntimeError("No poses/timestamps in %s" % pgo_dir)
    return times[:n], np.asarray(rots[:n])


def static_windows(t, a, w, win_s, accel_tol, gyro_tol, dir_tol=0.05):
    """
    Yield (t_mid, mean_accel) for each non-overlapping window in which the platform is not
    accelerating: |a| within accel_tol of g, and angular rate below gyro_tol throughout.
    """
    if len(t) < 2:
        return []
    dt = float(np.median(np.diff(t)))
    n = max(int(round(win_s / dt)), 10)
    out = []
    for i in range(0, len(t) - n, n):
        aw, ww = a[i:i + n], w[i:i + n]
        mag = np.linalg.norm(aw, axis=1)
        if np.abs(np.mean(mag) - G) > accel_tol:
            continue
        if np.max(np.linalg.norm(ww, axis=1)) > gyro_tol:
            continue
        # Reject windows whose direction is still swinging: a coordinated turn can hold |a|
        # near g while the vector rotates. dir_tol is a chord length on the unit sphere, so
        # 0.05 is about 2.9 deg -- tight enough that a walking operator passes nothing, which
        # is why it is tunable rather than fixed.
        u = aw / mag[:, None]
        if np.linalg.norm(u - u.mean(axis=0), axis=1).max() > dir_tol:
            continue
        out.append((float(t[i:i + n].mean()), aw.mean(axis=0)))
    return out


def measure_up(bag_path, pgo_dir, imu_topic, win_s, accel_tol, gyro_tol, rpy_override=None,
               outlier_deg=5.0, dir_tol=0.05):
    """Median 'up' direction in the map (ENU) frame, plus diagnostics."""
    topic, t, a, w, unit_scale = read_imu(bag_path, imu_topic)
    kf_t, kf_R = load_poses(pgo_dir)
    R_body_lidar, rpy = load_extrinsic(pgo_dir, rpy_override)

    wins = static_windows(t, a, w, win_s, accel_tol, gyro_tol, dir_tol)
    if not wins:
        raise RuntimeError(
            "No non-accelerating IMU window found (window %.1fs, |a|-g < %.2f m/s^2, "
            "|w| < %.2f rad/s, direction stable to %.3f). Loosen with --accel-tol / "
            "--gyro-tol / --dir-tol, and shorten --window." %
            (win_s, accel_tol, gyro_tol, dir_tol))

    ups, used = [], 0
    for tm, av in wins:
        # Nearest keyframe. Windows before the first or after the last keyframe clamp to it,
        # which is what we want for the pre-takeoff window.
        k = int(np.argmin(np.abs(kf_t - tm)))
        u_lidar = av / np.linalg.norm(av)
        # Same chain that places the points: map <- body <- lidar.
        ups.append(kf_R[k] @ (R_body_lidar @ u_lidar))
        used += 1

    ups = np.asarray(ups)

    def med(v):
        m = np.median(v, axis=0)
        return m / np.linalg.norm(m)

    # Reject windows that disagree with the bulk before taking the final median. A window can
    # pass the |a| and gyro gates and still be wrong -- a steady climb, or a hover the keyframe
    # nearest in time does not actually correspond to -- and one such window in a small sample
    # drags the median. Rejection is reported, not silent: if most windows go, distrust the
    # result rather than the outliers.
    up = med(ups)
    keep = np.degrees(np.arccos(np.clip(ups @ up, -1.0, 1.0))) <= outlier_deg
    n_rejected = int((~keep).sum())
    if keep.sum() >= 3 and n_rejected:
        ups_used = ups[keep]
        up = med(ups_used)
    else:
        ups_used = ups
        n_rejected = 0

    spread = np.degrees(np.arccos(np.clip(ups_used @ up, -1.0, 1.0)))
    return {
        "imu_topic": topic,
        "imu_unit_scale": unit_scale,
        "lidar_to_body_rpy": rpy,
        "n_windows": used,
        "n_windows_used": int(len(ups_used)),
        "n_windows_rejected": n_rejected,
        "outlier_threshold_deg": float(outlier_deg),
        "n_imu_samples": int(len(t)),
        "up_enu": up.tolist(),
        "window_spread_deg_median": float(np.median(spread)),
        "window_spread_deg_p95": float(np.percentile(spread, 95)),
    }


# --------------------------------------------------------------------------- correction

def minimal_rotation(v_from, v_to):
    """Smallest rotation taking v_from to v_to. Axis is perpendicular to both."""
    a = v_from / np.linalg.norm(v_from)
    b = v_to / np.linalg.norm(v_to)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.dot(a, b))
    if s < 1e-12:
        return np.eye(3), 0.0, np.array([0.0, 0.0, 1.0])
    axis = axis / s
    ang = float(np.arctan2(s, c))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return R, np.degrees(ang), axis


def enu_up_to_grid(up_enu, linear_map):
    """
    Re-express an ENU direction in the projected grid frame. The ENU->grid map is horizontal
    only (rotation by meridian convergence plus grid scale), so z passes through untouched.
    """
    M = np.asarray(linear_map, dtype=float).reshape(2, 2)
    M = M / np.sqrt(abs(np.linalg.det(M)))       # drop the scale, keep the rotation
    v = np.array([*(M @ up_enu[:2]), up_enu[2]])
    return v / np.linalg.norm(v)


def terrain_normal(las_path, cell=5.0, max_points=8_000_000):
    """
    Independent cross-check: plane through the lowest return in each grid cell. This is the
    terrain, not gravity -- it is reported so a slope can be told apart from a tilt.
    """
    import laspy
    with laspy.open(las_path) as f:
        step = max(int(f.header.point_count // max_points), 1)
        xs, ys, zs = [], [], []
        for pts in f.chunk_iterator(CHUNK):
            xs.append(np.asarray(pts.x)[::step])
            ys.append(np.asarray(pts.y)[::step])
            zs.append(np.asarray(pts.z)[::step])
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    ix = np.floor((x - x.min()) / cell).astype(np.int64)
    iy = np.floor((y - y.min()) / cell).astype(np.int64)
    key = ix * (iy.max() + 1) + iy
    order = np.lexsort((z, key))
    key_s = key[order]
    first = np.concatenate(([True], key_s[1:] != key_s[:-1]))
    sel = order[first]
    if sel.size < 10:
        return None
    A = np.column_stack([x[sel] - x[sel].mean(), y[sel] - y[sel].mean(), np.ones(sel.size)])
    coef, *_ = np.linalg.lstsq(A, z[sel] - z[sel].mean(), rcond=None)
    n = np.array([-coef[0], -coef[1], 1.0])
    n /= np.linalg.norm(n)
    return n


def rotate_traj_csv(src, dst, R, centre):
    """
    Rewrite a <stem>.traj.csv with its easting/northing/height rotated about centre.

    These sit in the cloud's own projected CRS, the same frame R was measured in, so the same
    matrix applies. The pose files in pgo/ are deliberately NOT rotated: they are the pipeline's
    raw output in its local ENU frame, and they are what a cloud is regenerated from.
    """
    lines = open(src).read().splitlines()
    out, hdr, n = [], None, 0
    for line in lines:
        if line.startswith("#") or hdr is None and not line[:1].isdigit():
            if not line.startswith("#"):
                hdr = [c.strip() for c in line.split(",")]
            out.append(line)
            continue
        p = line.split(",")
        if len(p) < 5:
            out.append(line)
            continue
        xyz = np.array([float(p[2]), float(p[3]), float(p[4])])
        xyz = R @ (xyz - centre) + centre
        out.append("%s,%s,%.4f,%.4f,%.4f" % (p[0], p[1], xyz[0], xyz[1], xyz[2]))
        n += 1
    open(dst, "w").write("\n".join(out) + "\n")
    return n


def rotate_las(src, dst, R, centre):
    """Rewrite src into dst with points rotated about centre. Chunked; header rebuilt."""
    import laspy
    with laspy.open(src) as f:
        hin = f.header
        hout = laspy.LasHeader(version=str(hin.version), point_format=hin.point_format.id)
        hout.scales = hin.scales
        hout.offsets = hin.offsets
        for vlr in hin.vlrs:
            hout.vlrs.append(vlr)
        for dim in hin.point_format.extra_dimensions:
            hout.add_extra_dim(laspy.ExtraBytesParams(name=dim.name, type=dim.dtype,
                                                      description=dim.description))
        n = 0
        with laspy.open(dst, mode="w", header=hout) as w:
            for pts in f.chunk_iterator(CHUNK):
                xyz = np.column_stack([pts.x, pts.y, pts.z])
                xyz = (xyz - centre) @ R.T + centre
                pts.x, pts.y, pts.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                w.write_points(pts)
                n += len(pts)
    return n


# --------------------------------------------------------------------------- entry point

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--las", required=True, help="Assembled LAS to correct")
    ap.add_argument("--bag", required=True, help="Source rosbag, for the IMU")
    ap.add_argument("--pgo-dir", default=None,
                    help="Directory with optimized_poses.txt and laser_timestamps.txt "
                         "(default: <las dir>/pgo)")
    ap.add_argument("--provenance", default=None,
                    help="assemble.py provenance json (default: <las stem>.provenance.json)")
    ap.add_argument("-o", "--output", default=None,
                    help="Output LAS (default: <stem>_rotation_fixed.las)")
    ap.add_argument("--imu-topic", default=None, help="Override IMU topic autodetection")
    ap.add_argument("--lidar-to-body-rpy", type=float, nargs=3, default=None,
                    metavar=("ROLL", "PITCH", "YAW"),
                    help="Override the extrinsic read from config_used.yaml [rad]")
    ap.add_argument("--window", type=float, default=1.0, help="Window length [s]")
    ap.add_argument("--accel-tol", type=float, default=0.30,
                    help="Max |mean|a|-g| for a usable window [m/s^2]")
    ap.add_argument("--gyro-tol", type=float, default=0.05,
                    help="Max angular rate in a usable window [rad/s]")
    ap.add_argument("--dir-tol", type=float, default=0.05,
                    help="Max swing of the acceleration direction within a window, as a chord "
                         "length on the unit sphere (0.05 ~ 2.9 deg)")
    ap.add_argument("--window-outlier-deg", type=float, default=5.0,
                    help="Discard windows further than this from the median direction before "
                         "taking the final median")
    ap.add_argument("--max-tilt", type=float, default=30.0,
                    help="Refuse corrections larger than this [deg]")
    ap.add_argument("--force", action="store_true", help="Apply even beyond --max-tilt")
    ap.add_argument("--dry-run", action="store_true", help="Measure and report, write nothing")
    ap.add_argument("--las-only", action="store_true",
                    help="Rotate only the named LAS. By default the odometry twin and every "
                         "trajectory CSV beside it are rotated too, so the whole product set "
                         "stays in one frame.")
    args = ap.parse_args()

    las = args.las
    stem = las[:-4] if las.lower().endswith(".las") else las
    pgo_dir = args.pgo_dir or os.path.join(os.path.dirname(las), "pgo")
    prov_path = args.provenance or (stem + ".provenance.json")
    out = args.output or (stem + "_rotation_fixed.las")

    for p in (las, args.bag, prov_path):
        if not os.path.exists(p):
            sys.exit("Not found: %s" % p)

    geo = json.load(open(prov_path)).get("georeference")
    if not geo:
        sys.exit("No georeference block in %s -- cannot place the rotation centre." % prov_path)

    print("Measuring gravity from %s" % args.bag)
    m = measure_up(args.bag, pgo_dir, args.imu_topic, args.window,
                   args.accel_tol, args.gyro_tol, args.lidar_to_body_rpy,
                   args.window_outlier_deg, args.dir_tol)
    print("  IMU topic            %s (%d samples)" % (m["imu_topic"], m["n_imu_samples"]))
    print("  accel units          %s" % ("g, rescaled to m/s^2"
                                         if m["imu_unit_scale"] != 1.0 else "m/s^2"))
    print("  lidar->body rpy      %s" % (m["lidar_to_body_rpy"] or "identity (none found)"))
    print("  usable windows       %d found, %d used, %d rejected as outliers"
          % (m["n_windows"], m["n_windows_used"], m["n_windows_rejected"]))
    if m["n_windows_used"] < 3:
        print("  WARNING              fewer than 3 windows agree; the estimate rests on very "
              "little. Loosen --accel-tol / --gyro-tol and compare.")
    print("  between-window spread %.2f deg median, %.2f deg p95"
          % (m["window_spread_deg_median"], m["window_spread_deg_p95"]))

    up_grid = enu_up_to_grid(np.asarray(m["up_enu"]), geo["linear_map_row_major"])
    R, tilt, axis = minimal_rotation(up_grid, np.array([0.0, 0.0, 1.0]))
    # Azimuth shift of horizontal directions. The geodesic rotation tilts them out of plane
    # rather than spinning them about the vertical, so this stays far below the tilt itself.
    az = np.linspace(0, 2 * np.pi, 361)
    v = np.column_stack([np.cos(az), np.sin(az), np.zeros_like(az)])
    vr = v @ R.T
    d = np.degrees(np.abs(np.arctan2(vr[:, 1], vr[:, 0]) - az))
    heading_change = float(np.max(np.minimum(d, 360.0 - d)))

    print("  measured up (grid)   [%.6f %.6f %.6f]" % tuple(up_grid))
    print("  tilt to correct      %.3f deg about axis [%.4f %.4f %.4f]"
          % (tilt, *axis))
    print("  max azimuth shift    %.3f deg (worst-case over all headings)" % heading_change)

    n = terrain_normal(las)
    if n is not None:
        slope = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
        after = float(np.degrees(np.arccos(np.clip((R @ n)[2], -1, 1))))
        print("  terrain normal       %.2f deg from vertical before, %.2f deg after"
              % (slope, after))
        print("                       (terrain, not gravity -- a real slope stays non-zero)")

    if tilt > args.max_tilt and not args.force:
        sys.exit("Refusing: %.2f deg exceeds --max-tilt %.2f. Check the IMU window and the "
                 "lidar_to_body extrinsic before forcing." % (tilt, args.max_tilt))

    record = dict(m, tilt_deg=tilt, axis=axis.tolist(), up_grid=up_grid.tolist(),
                  rotation_row_major=R.reshape(-1).tolist(),
                  heading_change_deg=heading_change, source_las=las, source_bag=args.bag)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    centre = np.array([geo["origin_easting"], geo["origin_northing"],
                       geo["origin_orthometric_h"]], dtype=float)
    record["rotation_centre"] = centre.tolist()
    print("\nRotating about the datum [%.3f %.3f %.3f]" % tuple(centre))

    out_stem = out[:-4] if out.lower().endswith(".las") else out
    record["n_points"] = rotate_las(las, out, R, centre)
    written = [{"file": out, "kind": "las", "n": record["n_points"]}]
    print("  %-58s %d points" % (os.path.basename(out), record["n_points"]))

    if not args.las_only:
        # The odometry twin gets the same matrix about the same pivot: it is the drift
        # baseline, so it only stays comparable if it moves with the optimised cloud.
        for src, dst, kind in (
                (stem + "_odom.las", out_stem + "_odom.las", "las"),
                (stem + ".traj.csv", out_stem + ".traj.csv", "csv"),
                (stem + ".traj_odom.csv", out_stem + ".traj_odom.csv", "csv"),
                (stem + "_odom.traj.csv", out_stem + "_odom.traj.csv", "csv"),
                (stem + "_odom.traj_odom.csv", out_stem + "_odom.traj_odom.csv", "csv")):
            if not os.path.exists(src):
                continue
            n = (rotate_las(src, dst, R, centre) if kind == "las"
                 else rotate_traj_csv(src, dst, R, centre))
            written.append({"file": dst, "kind": kind, "n": n})
            print("  %-58s %d %s" % (os.path.basename(dst), n,
                                     "points" if kind == "las" else "rows"))

    record["files_written"] = written
    record["pose_files_not_rotated"] = (
        "optimized_poses.txt and odometry_poses.txt are left in the pipeline's local ENU "
        "frame. They are the input a cloud is regenerated from, not a product; the correction "
        "is recorded here instead.")
    json.dump(record, open(out_stem + ".gravity_align.json", "w"), indent=2)


if __name__ == "__main__":
    main()
