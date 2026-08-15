#!/usr/bin/env python3
"""
Do the per-aircraft clouds in a merged cloud actually agree with each other?

Nothing else in the pipeline asks this. `sanity_check.py` verifies that every bag contributed
keyframes, not that the resulting clouds occupy the same space -- and a merge where each
aircraft floats on its own GNSS anchor passes every other check while showing every stem two or
three times. That failure reached a hand review before anyone caught it.

The metric is a cloud-to-cloud distance between each pair of aircraft, computed only where they
overlap in XY. A well-registered pair sits at a few centimetres, the scale of the point spacing.
A disconnected pair sits at metres.

Canopy is excluded. Returns above `--max-height` over the local ground are viewpoint-dependent
and move in wind, so they disagree between aircraft even when the registration is perfect --
the same reason the loop-closure ICP filters them.

    interaircraft_check.py MERGED.las [--voxel 0.10] [--sample 200000]

Never run this on a grid_cleanup.py output. That tool deletes the duplicated points which are
the very evidence this check looks for, and would report a clean result on a broken merge.

Thresholds are calibrated against the four Campaign C merges, three accepted on hand inspection
and one rejected:

    20251209_attempt1   worst pair median 0.066 m   accepted
    20251217_attempt1   worst pair median 0.145 m   accepted
    20251217_attempt2   worst pair median 0.251 m   accepted
    20251209_attempt2   worst pair median 0.889 m   REJECTED (0.626 and 0.889 on two pairs)

So 0.40 m sits in the middle of a factor-2.5 gap between the worst accepted and the best
rejected pair. The p95 test catches the same rejected merge independently at 3.80 m against a
worst-accepted 1.70 m. Both are deliberately loose: this is a screen that says "look at this
one", not a proof of registration.

Note the rejected merge had one *well* registered pair, drone 0 vs 2 at 0.059 m. Judge a merge
on its worst pair, never its average -- one aircraft floating free is enough to ruin it.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

CHUNK = 4_000_000
GRID = 1.0                      # XY cell for the overlap test and the ground estimate


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cloud")
    ap.add_argument("--voxel", type=float, default=0.10,
                    help="Voxel size for the reference cloud of each pair [m]")
    ap.add_argument("--sample", type=int, default=200_000,
                    help="Points sampled from the query aircraft per pair")
    ap.add_argument("--max-height", type=float, default=7.0,
                    help="Drop returns more than this above the local ground [m]")
    ap.add_argument("--json", default=None)
    return ap.parse_args(argv)


def read_by_drone(path):
    """XYZ per drone_id, plus the per-cell ground height over the whole cloud."""
    import laspy

    with laspy.open(path) as f:
        if "drone_id" not in f.header.point_format.extra_dimension_names:
            sys.exit("no drone_id dimension -- not a merged multi-aircraft cloud")
        pts = {}
        for ch in f.chunk_iterator(CHUNK):
            d = np.asarray(ch["drone_id"])
            xyz = np.column_stack([ch.x, ch.y, ch.z])
            for v in np.unique(d):
                pts.setdefault(int(v), []).append(xyz[d == v])
    return {k: np.vstack(v) for k, v in pts.items()}


def ground_height(all_xyz):
    """2nd-percentile z per 1 m cell, as a dict keyed by cell -- the local ground."""
    i = np.floor(all_xyz[:, 0] / GRID).astype(np.int64)
    j = np.floor(all_xyz[:, 1] / GRID).astype(np.int64)
    key = i * 100_000_000 + j
    order = np.argsort(key, kind="stable")
    key_s, z_s = key[order], all_xyz[order, 2]
    bounds = np.flatnonzero(np.diff(key_s)) + 1
    out = {}
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(key_s)]):
        out[int(key_s[a])] = float(np.percentile(z_s[a:b], 2))
    return out


def cell_keys(xyz):
    return (np.floor(xyz[:, 0] / GRID).astype(np.int64) * 100_000_000
            + np.floor(xyz[:, 1] / GRID).astype(np.int64))


def voxel_unique(xyz, voxel):
    q = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    return xyz[idx]


def main(argv=None):
    args = parse_args(argv)
    from scipy.spatial import cKDTree

    src = os.path.abspath(args.cloud)
    print("cloud    : %s" % os.path.basename(src))
    per = read_by_drone(src)
    if len(per) < 2:
        sys.exit("only one aircraft in this cloud -- nothing to compare")

    allxyz = np.vstack(list(per.values()))
    g = ground_height(allxyz)
    del allxyz

    # Keep the near-ground slab per aircraft: that is where stems are, and where two aircraft
    # should agree exactly if the merge is real.
    keys, slab = {}, {}
    for d, xyz in per.items():
        k = cell_keys(xyz)
        gz = np.array([g.get(int(v), -np.inf) for v in k])
        m = xyz[:, 2] <= gz + args.max_height
        slab[d] = xyz[m]
        keys[d] = set(np.unique(k[m]).tolist())
        print("  drone %d: %d points, %d in the near-ground slab, %d cells"
              % (d, len(xyz), int(m.sum()), len(keys[d])))
    del per

    rng = np.random.default_rng(0)
    results = []
    for a, b in itertools.combinations(sorted(slab), 2):
        common = keys[a] & keys[b]
        if len(common) < 20:
            print("  drone %d vs %d: only %d shared cells -- no meaningful overlap"
                  % (a, b, len(common)))
            results.append({"pair": [a, b], "shared_cells": len(common), "median_m": None})
            continue
        ka, kb = cell_keys(slab[a]), cell_keys(slab[b])
        cm = np.fromiter(common, dtype=np.int64)
        A = slab[a][np.isin(ka, cm)]
        B = slab[b][np.isin(kb, cm)]
        if len(A) < 100 or len(B) < 100:
            results.append({"pair": [a, b], "shared_cells": len(common), "median_m": None})
            continue
        Bv = voxel_unique(B, args.voxel)
        q = A if len(A) <= args.sample else A[rng.choice(len(A), args.sample, replace=False)]
        dist, _ = cKDTree(Bv).query(q, k=1)
        med, p95 = float(np.median(dist)), float(np.percentile(dist, 95))
        verdict = ("registered" if (med < 0.40 and p95 < 2.50) else
                   "SUSPECT" if med < 1.00 else "DISCONNECTED")
        print("  drone %d vs %d: %d shared cells, %d sampled -> median %.3f m, 95th %.3f m  %s"
              % (a, b, len(common), len(q), med, p95, verdict))
        results.append({"pair": [a, b], "shared_cells": len(common),
                        "n_sampled": int(len(q)), "median_m": med, "p95_m": p95,
                        "verdict": verdict})

    worst = max((r["median_m"] for r in results if r["median_m"] is not None), default=None)
    failed = [r for r in results if r.get("verdict") in ("SUSPECT", "DISCONNECTED")]
    print("  worst pair median: %s -- %s"
          % ("%.3f m" % worst if worst is not None else "n/a",
             "all pairs registered" if not failed
             else "%d of %d pairs need a look" % (len(failed), len(results))))

    out = args.json or (src[:-4] + ".interaircraft.json")
    json.dump({"cloud": src, "voxel": args.voxel, "max_height": args.max_height,
               "pairs": results, "worst_median_m": worst,
               "n_pairs_flagged": len(failed)}, open(out, "w"), indent=2)
    print("  record  %s" % out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
