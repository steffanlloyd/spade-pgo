#!/usr/bin/env python3
"""
Sanity-check an assembled release cloud.

Run after each assembly in the re-assembly sequence, so a broken solution is caught at the
step that produced it rather than thirteen clouds later. Nothing here replaces looking at the
cloud; these are the failures that can be detected without eyes.

    sanity_check.py CLOUD.las [--bags DIR] [--expect-points N] [--expect-drones a,b,c]

Exit status 0 = pass, 1 = at least one FAIL. WARNs do not change the exit status.

Checks
------
  CRS          the header must carry the release CRS, not nothing
  points       non-empty, and equal to --expect-points when given
  extent       the sites are ~200 x 300 m; a diverged solution blows up by orders of magnitude
  keyframes    the keyframe dimension must be present and cover a contiguous-ish range
  jumps        consecutive keyframes sit ~kf_gap_lin apart; a large jump means a broken graph
  gnss         the optimised trajectory is compared against the raw GNSS fixes in the bags.
               This is the strongest available check: it is an external reference the pose
               graph is only loosely tied to, so a solution that has folded or rotated shows
               up immediately as a large residual.
  gnss_z       the vertical half of the same comparison, reported separately. Nothing else
               constrains z strongly, so vertical error is where a bad solve hides.
  drift        optimised vs odometry-only trajectory, reported for the D4.2 drift columns
"""

import os
import re
import sys
import glob
import json
import argparse

import numpy as np
import laspy

# The stands are roughly 200 x 300 m. Ten times that is unambiguously a diverged solution
# rather than an unusually large site.
MAX_PLAUSIBLE_EXTENT_M = 2000.0
MAX_PLAUSIBLE_Z_M = 400.0
# kf_gap_lin is 1.0 m. Keyframes are created on motion, so gaps larger than this arise
# legitimately at session boundaries, but a 25 m step inside a session is a broken graph.
MAX_KF_STEP_M = 25.0
# GNSS below canopy is metre-level and enters the graph scaled down by gps_noise_scale, so
# the trajectory is not expected to sit on it. Tens of metres, though, means it has drifted
# free of its anchor.
MAX_GNSS_MEDIAN_M = 20.0
# Vertical is checked as spread about the median offset, not as an absolute difference: the
# altitude convention of a NavSatFix is not reliably ellipsoidal across our sources, so a
# constant tens-of-metres bias is a datum question, while scatter about it is a broken graph.
MAX_GNSS_VERTICAL_SPREAD_M = 15.0
# A bias this large is worth saying out loud even though it does not fail the check.
GNSS_VERTICAL_BIAS_WARN_M = 5.0

results = []


def record(level, name, msg):
    results.append((level, name, msg))
    print("%-5s %-10s %s" % (level, name, msg))


def check_crs(las):
    crs = las.header.parse_crs()
    if crs is None:
        record("FAIL", "crs", "no CRS in header -- this is the defect the release exists to fix")
        return None
    epsg = crs.to_epsg()
    if epsg != 5972:
        record("WARN", "crs", "CRS is EPSG:%s, expected 5972" % epsg)
    else:
        record("PASS", "crs", "EPSG:5972 (%s)" % crs.name)
    return crs


def check_points(las, expect):
    n = las.header.point_count
    if n == 0:
        record("FAIL", "points", "cloud is empty")
        return
    if expect is not None:
        if n == expect:
            record("PASS", "points", "%d, exactly as expected" % n)
        else:
            record("FAIL", "points", "%d, expected %d (difference %+d)" % (n, expect, n - expect))
    else:
        record("PASS", "points", "%d" % n)


def check_extent(las):
    ext = las.header.maxs - las.header.mins
    msg = "%.1f x %.1f x %.1f m" % tuple(ext)
    if max(ext[0], ext[1]) > MAX_PLAUSIBLE_EXTENT_M:
        record("FAIL", "extent", msg + " -- horizontal extent implausible, solution likely diverged")
    elif ext[2] > MAX_PLAUSIBLE_Z_M:
        # Vertical only warns. The hand-carried rigs are not expected to have their Z axis
        # perfectly aligned with the world, GNSS altitude is weakly weighted, and canopy
        # returns legitimately stretch the vertical extent. A tall cloud is worth a look,
        # not a failure.
        record("WARN", "extent", msg + " -- vertical extent large; check Z alignment")
    else:
        record("PASS", "extent", msg)


def check_drones(las, expect):
    if "drone_id" not in las.point_format.extra_dimension_names:
        record("WARN", "drones", "no drone_id dimension")
        return
    ids, counts = np.unique(las.drone_id, return_counts=True)
    summary = ", ".join("%d=%.2f M" % (i, c / 1e6) for i, c in zip(ids, counts))
    if expect:
        exp = [float(v) for v in expect.split(",")]
        got = [c / 1e6 for c in counts]
        if len(exp) != len(got):
            record("FAIL", "drones", "%d aircraft, expected %d (%s)" % (len(got), len(exp), summary))
            return
        worst = max(abs(a - b) for a, b in zip(exp, got))
        if worst > 0.02:
            record("FAIL", "drones", "%s -- expected %s, worst difference %.2f M" % (summary, expect, worst))
        else:
            record("PASS", "drones", "%s, matches expectation" % summary)
    else:
        record("PASS", "drones", summary)
    # point_source_id must mirror drone_id, 1-based
    if not np.array_equal(np.unique(las.point_source_id), ids + 1):
        record("WARN", "psid", "point_source_id does not mirror drone_id+1")


def check_session_coverage(las, bags_dir):
    """
    Every source bag must appear in the cloud as its own aircraft/session.

    The orchestrator counts a session as successful when its bag finishes playing, which
    says nothing about whether the pose graph accepted any keyframes from it. A session
    that fails to initialise therefore contributes nothing while still being reported as
    a success, and the result is a "merged" cloud silently containing fewer aircraft than
    were flown. Comparing distinct drone_ids against the number of source bags is the
    cheapest way to catch that.
    """
    if not bags_dir or not os.path.isdir(bags_dir):
        return
    n_bags = len(glob.glob(os.path.join(bags_dir, "**", "*.bag"), recursive=True))
    if n_bags <= 1:
        return
    if "drone_id" not in las.point_format.extra_dimension_names:
        record("WARN", "sessions", "cannot check: no drone_id dimension")
        return
    n_present = len(np.unique(las.drone_id))
    if n_present < n_bags:
        record("FAIL", "sessions",
               "%d of %d source bags contributed points (aircraft present: %s) -- the "
               "missing sessions produced no keyframes and this is NOT a complete merge"
               % (n_present, n_bags, sorted(int(v) for v in np.unique(las.drone_id))))
    else:
        record("PASS", "sessions", "all %d source bags contributed" % n_bags)


def load_traj(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("keyframe"):
                continue
            p = line.strip().split(",")
            if len(p) >= 5:
                rows.append([float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    return np.array(rows) if rows else None


def check_trajectory(stem):
    traj = load_traj(stem + ".traj.csv")
    if traj is None:
        record("WARN", "traj", "no trajectory file written")
        return None
    xyz = traj[:, 2:5]
    sess = traj[:, 1]
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    same = np.diff(sess) == 0          # ignore the discontinuity between sessions
    steps = steps[same] if same.any() else steps
    if steps.size == 0:
        record("WARN", "jumps", "too few keyframes to assess")
    elif steps.max() > MAX_KF_STEP_M:
        record("FAIL", "jumps", "largest within-session keyframe step %.1f m (median %.2f m) "
                                "-- discontinuity in the graph" % (steps.max(), np.median(steps)))
    else:
        record("PASS", "jumps", "largest within-session step %.1f m, median %.2f m"
               % (steps.max(), np.median(steps)))
    return traj


def check_drift(stem):
    a = load_traj(stem + ".traj.csv")
    b = load_traj(stem + ".traj_odom.csv")
    if a is None or b is None:
        record("WARN", "drift", "no odometry trajectory; drift figures unavailable")
        return
    ka = {int(r[0]): r[2:5] for r in a}
    kb = {int(r[0]): r[2:5] for r in b}
    common = sorted(set(ka) & set(kb))
    if not common:
        record("WARN", "drift", "no shared keyframes between the two trajectories")
        return
    d = np.linalg.norm(np.array([ka[k] for k in common]) - np.array([kb[k] for k in common]), axis=1)
    record("PASS", "drift", "optimised vs odometry over %d keyframes: median %.2f m, "
                            "95th %.2f m, max %.2f m"
           % (len(common), np.median(d), np.percentile(d, 95), d.max()))


def check_gnss(traj, bags_dir, crs):
    """
    Compare the optimised trajectory against the raw GNSS fixes in the source bags, in 3D.

    Horizontal and vertical are reported separately and on purpose. For each fix we take the
    horizontally nearest trajectory vertex and read off the height difference there, rather
    than taking a 3D nearest neighbour, which would let a vertical error hide as a horizontal
    one. Vertical is then split into a median offset and the scatter about it, because the
    altitude datum of a NavSatFix is not consistent across our GNSS sources while the scatter
    is meaningful regardless.
    """
    if traj is None or crs is None or not bags_dir:
        return
    try:
        import rosbag
        import pyproj
    except ImportError as e:
        record("WARN", "gnss", "cannot check: %s" % e)
        return

    bags = sorted(glob.glob(os.path.join(bags_dir, "**", "*.bag"), recursive=True))
    if not bags:
        record("WARN", "gnss", "no bags under %s" % bags_dir)
        return

    lat, lon, alt = [], [], []
    for b in bags:
        try:
            with rosbag.Bag(b) as bag:
                topics = bag.get_type_and_topic_info().topics
                t = next((x for x in ("/septentrio_gnss/navsatfix",
                                      "/mavros/global_position/raw/fix",
                                      "/mavros/global_position/global") if x in topics), None)
                if not t:
                    continue
                for _, m, _ in bag.read_messages(topics=[t]):
                    if np.isfinite(m.latitude) and m.latitude != 0.0 and np.isfinite(m.altitude):
                        lat.append(m.latitude); lon.append(m.longitude); alt.append(m.altitude)
        except Exception as e:
            record("WARN", "gnss", "%s: %s" % (os.path.basename(b), e))
    if not lat:
        record("WARN", "gnss", "no usable fixes in the source bags")
        return

    lat, lon, alt = np.array(lat), np.array(lon), np.array(alt)
    # 4979 rather than 4326: the vertical component has to go through the geoid model to land
    # in the cloud's orthometric datum. Falling back to 2D loses the height entirely.
    try:
        tr = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4979), crs, always_xy=True)
        E, N, H = tr.transform(lon, lat, alt)
        if not np.all(np.isfinite(H)):
            raise ValueError("vertical transform produced non-finite heights")
    except Exception as e:
        record("WARN", "gnss", "3D transform unavailable (%s); checking horizontal only" % e)
        tr = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs, always_xy=True)
        E, N = tr.transform(lon, lat)
        H = None

    T = traj[:, 2:4]
    Z = traj[:, 4]
    # Nearest trajectory vertex per fix, in blocks to bound memory.
    dmin = np.empty(len(E))
    idx = np.empty(len(E), dtype=int)
    for i in range(0, len(E), 2000):
        blk = np.column_stack([E[i:i+2000], N[i:i+2000]])
        d = np.linalg.norm(blk[:, None, :] - T[None, :, :], axis=2)
        dmin[i:i+2000] = d.min(axis=1)
        idx[i:i+2000] = d.argmin(axis=1)

    med = float(np.median(dmin))
    msg = ("%d fixes, horizontal distance to trajectory: median %.2f m, 95th %.2f m, "
           "max %.2f m" % (len(dmin), med, np.percentile(dmin, 95), dmin.max()))
    if med > MAX_GNSS_MEDIAN_M:
        record("FAIL", "gnss", msg + " -- trajectory has drifted free of its GNSS anchor")
    else:
        record("PASS", "gnss", msg)

    if H is None:
        record("WARN", "gnss_z", "no vertical comparison -- the 3D transform was unavailable")
        return

    dz = H - Z[idx]
    bias = float(np.median(dz))
    spread = float(np.median(np.abs(dz - bias)))
    zmsg = ("vertical offset median %+.2f m, scatter about it %.2f m (95th %.2f m)"
            % (bias, spread, np.percentile(np.abs(dz - bias), 95)))
    if spread > MAX_GNSS_VERTICAL_SPREAD_M:
        record("FAIL", "gnss_z", zmsg + " -- the trajectory does not track GNSS altitude")
    elif abs(bias) > GNSS_VERTICAL_BIAS_WARN_M:
        record("WARN", "gnss_z", zmsg + " -- large constant offset: either genuine vertical "
                                        "drift, or the source topic's altitude datum differs "
                                        "from NN2000")
    else:
        record("PASS", "gnss_z", zmsg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cloud")
    ap.add_argument("--bags", default=None, help="Directory of source bags, for the GNSS check")
    ap.add_argument("--expect-points", type=int, default=None)
    ap.add_argument("--expect-drones", default=None,
                    help="Expected per-aircraft point counts in millions, e.g. 15.52,14.00,13.73")
    args = ap.parse_args()

    print("=" * 78)
    print("SANITY CHECK  %s" % args.cloud)
    print("=" * 78)

    if not os.path.exists(args.cloud):
        record("FAIL", "file", "does not exist")
        sys.exit(1)

    las = laspy.read(args.cloud)
    stem = os.path.splitext(args.cloud)[0]

    crs = check_crs(las)
    check_points(las, args.expect_points)
    check_extent(las)
    check_drones(las, args.expect_drones)
    check_session_coverage(las, args.bags)
    traj = check_trajectory(stem)
    check_drift(stem)
    check_gnss(traj, args.bags, crs)

    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]
    print("-" * 78)
    print("%d passed, %d warnings, %d FAILURES" % (
        len([r for r in results if r[0] == "PASS"]), len(warns), len(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
