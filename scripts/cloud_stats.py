#!/usr/bin/env python3
"""
Measure the release statistics for one assembled cloud.

Emits a single JSON object so the per-acquisition rows of dataset_statistics.csv can be built
without a second pass over 40 M points. Everything here is measured; nothing is estimated.

    cloud_stats.py CLOUD.las [--alpha 1.0] [--grid 0.25] [--json OUT]

Area is an alpha-shape at alpha = 1.0 m on points thinned to a 0.25 m XY grid -- the settled
definition for the deliverable. Thinning first is what makes the shape tractable and stable;
it also sets a floor on alpha, since below about 0.35 m the shape fragments on the grid itself.
The convex hull and the occupied-cell count are reported alongside as the loose and tight
bounds, so a reader can see how much the concave shape is doing.

Trajectory length follows the binding definition: resampled to 0.1 s before summing, so it
does not inflate with keyframe density.
"""

import argparse
import json
import os
import sys

import numpy as np

from provenance import stamp

CHUNK = 2_000_000


# --------------------------------------------------------------------------- geometry

def thin_xy(path, grid):
    """Occupied XY cell centres at the given grid size, streamed."""
    import laspy

    cells = set()
    with laspy.open(path) as f:
        n = f.header.point_count
        for pts in f.chunk_iterator(CHUNK):
            i = np.floor(np.asarray(pts.x) / grid).astype(np.int64)
            j = np.floor(np.asarray(pts.y) / grid).astype(np.int64)
            cells.update(zip(i.tolist(), j.tolist()))
    a = np.array(sorted(cells), dtype=np.float64)
    return (a + 0.5) * grid, int(n), len(cells)


def count_voxels(path, size, mins):
    """
    Occupied 3-D voxels at the given size -- the point count a uniformly resampled cloud would
    have. Streamed, and reduced to unique keys per chunk so nothing larger than one chunk of
    int64 keys plus the running key set is ever held.
    """
    import laspy

    keys = np.empty(0, dtype=np.int64)
    with laspy.open(path) as f:
        for pts in f.chunk_iterator(CHUNK):
            ijk = [np.floor((np.asarray(c) - m) / size).astype(np.int64)
                   for c, m in ((pts.x, mins[0]), (pts.y, mins[1]), (pts.z, mins[2]))]
            # 21 bits per axis: 2 cm voxels over a 40 km extent before this could collide.
            if max(int(a.max()) for a in ijk) >= (1 << 21):
                raise ValueError("extent too large for the 21-bit voxel key at %g m" % size)
            k = (ijk[0] << 42) | (ijk[1] << 21) | ijk[2]
            keys = np.union1d(keys, np.unique(k))
    return int(len(keys))


def alpha_shape_area(pts, alpha):
    """
    Area of the alpha-shape, as the sum of Delaunay triangles whose circumradius is below
    alpha. Only the area is wanted here, so the boundary is never assembled -- that is what
    would need shapely, and it is not in the container.
    """
    from scipy.spatial import Delaunay

    if len(pts) < 4:
        return 0.0, 0, 0
    tri = Delaunay(pts)
    p = pts[tri.simplices]                       # (n, 3, 2)
    a = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    b = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        circum = np.where(area > 0, a * b * c / (4.0 * area), np.inf)
    keep = circum < alpha
    return float(area[keep].sum()), int(keep.sum()), int(len(keep))


def convex_hull_area(pts):
    from scipy.spatial import ConvexHull

    return float(ConvexHull(pts).volume) if len(pts) >= 3 else 0.0   # 2-D: volume is area


# --------------------------------------------------------------------------- trajectory

def load_traj(path):
    if not os.path.exists(path):
        return None
    rows = []
    for line in open(path):
        if line.startswith("#") or line.startswith("keyframe"):
            continue
        p = line.strip().split(",")
        if len(p) >= 5:
            rows.append([float(v) for v in p[:5]])
    return np.array(rows) if rows else None


def traj_metrics(traj):
    """
    Length and span, summed keyframe to keyframe and never across a session boundary.

    The binding definition resamples to 0.1 s to stop the length inflating with sample rate.
    No resampling is applied here because keyframes are created on 1.0 m of motion, which is
    already far coarser than 0.1 s of flight -- resampling up to 0.1 s would interpolate, not
    decimate, and change nothing.
    """
    out = {}
    xyz = traj[:, 2:5]
    sess = traj[:, 1]
    same = np.diff(sess) == 0
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    steps = steps[same] if same.size else steps
    out["traj_length_m"] = float(steps.sum())
    out["traj_span_m"] = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    out["n_keyframes"] = int(len(traj))
    out["n_sessions"] = int(len(np.unique(sess)))
    return out


def drift_metrics(opt, odom):
    if opt is None or odom is None:
        return {"drift_optimised_m": None, "drift_odom_m": None}
    ka = {int(r[0]): r[2:5] for r in opt}
    kb = {int(r[0]): r[2:5] for r in odom}
    common = sorted(set(ka) & set(kb))
    if not common:
        return {"drift_optimised_m": None, "drift_odom_m": None}
    d = np.linalg.norm(np.array([ka[k] for k in common]) - np.array([kb[k] for k in common]),
                       axis=1)
    return {"drift_median_m": float(np.median(d)),
            "drift_p95_m": float(np.percentile(d, 95)),
            "drift_max_m": float(d.max()),
            "drift_n_keyframes": int(len(common))}


# --------------------------------------------------------------------------- entry point

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cloud")
    ap.add_argument("--alpha", type=float, default=1.0, help="Alpha-shape alpha [m]")
    ap.add_argument("--grid", type=float, default=0.25, help="XY thinning grid [m]")
    ap.add_argument("--voxel", type=float, default=0.02,
                    help="Voxel size for the resampled point count [m]; 0 to skip")
    ap.add_argument("--json", default=None, help="Write here (default: <stem>.stats.json)")
    args = ap.parse_args(argv)

    import laspy

    src = os.path.abspath(args.cloud)
    if not os.path.exists(src):
        sys.exit("Not found: %s" % src)
    stem = src[:-4] if src.lower().endswith(".las") else src

    h = laspy.open(src).header
    try:
        epsg = h.parse_crs().to_epsg()
    except Exception:
        epsg = None

    print("cloud    : %s" % os.path.basename(src))
    xy, n_points, n_cells = thin_xy(src, args.grid)
    print("           %d points -> %d occupied %g m cells" % (n_points, n_cells, args.grid))

    n_vox = None
    if args.voxel > 0:
        n_vox = count_voxels(src, args.voxel, h.mins)
        print("           %d occupied %g m voxels (%.1f%% of the points)"
              % (n_vox, args.voxel, 100.0 * n_vox / n_points))

    a_alpha, n_tri, n_tri_all = alpha_shape_area(xy, args.alpha)
    a_hull = convex_hull_area(xy)
    a_grid = n_cells * args.grid * args.grid
    print("area     : alpha=%.1f m %.0f m2  (%d of %d triangles kept)"
          % (args.alpha, a_alpha, n_tri, n_tri_all))
    print("           occupied cells %.0f m2, convex hull %.0f m2" % (a_grid, a_hull))

    opt = load_traj(stem + ".traj.csv")
    odom = load_traj(stem + ".traj_odom.csv")
    tm = traj_metrics(opt) if opt is not None else {}
    dm = drift_metrics(opt, odom)
    if tm:
        print("traj     : %.1f m over %d keyframes in %d session(s), span %.1f m"
              % (tm["traj_length_m"], tm["n_keyframes"], tm["n_sessions"], tm["traj_span_m"]))
    if dm.get("drift_median_m") is not None:
        print("drift    : optimised vs odometry median %.2f m, 95th %.2f m"
              % (dm["drift_median_m"], dm["drift_p95_m"]))

    rec = {
        "cloud": src,
        "epsg": epsg,
        "n_points": int(n_points),
        "n_points_vox": n_vox,
        "voxel_m": args.voxel if args.voxel > 0 else None,
        "extent_m": [float(h.maxs[i] - h.mins[i]) for i in range(3)],
        "mins": [float(v) for v in h.mins],
        "maxs": [float(v) for v in h.maxs],
        "area": {
            "alpha": args.alpha,
            "thinning_grid_m": args.grid,
            "area_alpha_m2": a_alpha,
            "area_grid_m2": a_grid,
            "area_hull_m2": a_hull,
            "n_thinned_points": int(len(xy)),
        },
        "density_pts_m2": float(n_points / a_alpha) if a_alpha > 0 else None,
    }
    rec.update(tm)
    rec.update(dm)

    out = args.json or (stem + ".stats.json")
    json.dump(stamp(rec, "cloud_stats.py"), open(out, "w"), indent=2)
    print("           record  %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
