#!/usr/bin/env python3
"""
Trim the ragged XY fringe off an assembled LAS.

A below-canopy cloud reaches far past the flight path: on a typical Campaign B acquisition the
flight covers 104 x 58 m while the cloud spans 194 x 177 m, the difference being a halo of
cells holding a handful of returns each. Measured on 20251121_123511, cells below 100 points
per square metre are 38% of the occupied area but 0.27% of the points, and sit a median 26 m
from the trajectory against 2.5 m for the dense core. Cutting them makes the cloud look like
what was surveyed rather than what the sensor happened to catch at maximum range.

The cut is in XY only, and the input must already be gravity-corrected -- projecting a tilted
cloud to XY smears the footprint, so trimming before rotating cuts the wrong thing.

Method:
  1. Count points into a square XY grid.
  2. Keep cells at or above --min-points.
  3. Close, then fill enclosed holes, so genuine interior gaps -- dense stands, shadowed
     pockets -- are not punched out of the middle of the cloud.
  4. Keep the largest connected component, which drops detached islands such as the points of
     a stranded keyframe.
  5. Optionally dilate by a cell so the boundary does not clip the outermost real returns.

Interior density is never touched: a cell is kept whole or dropped whole. This is not
grid_cleanup.py, which discards points inside the cloud to hide ghosting.

Output is a new LAS plus a <stem>.trim.json recording the parameters and the mask itself, so
the same cut can be reapplied to another product of the same acquisition.
"""

import argparse
import json
import os
import sys

import numpy as np

from provenance import stamp

CHUNK = 2_000_000


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Gravity-corrected LAS to trim")
    ap.add_argument("-o", "--output", default=None,
                    help="Output LAS (default: <stem>_trimmed.las)")
    ap.add_argument("--cell", type=float, default=1.0, help="XY cell size [m]")
    ap.add_argument("--min-points", type=int, default=100,
                    help="Cells with fewer points than this are dropped")
    ap.add_argument("--close", type=float, default=2.0,
                    help="Morphological closing radius [m]; 0 disables")
    ap.add_argument("--no-fill-holes", action="store_true",
                    help="Do not fill enclosed interior gaps (they will be cut out)")
    ap.add_argument("--dilate", type=int, default=1,
                    help="Grow the final mask by this many cells")
    ap.add_argument("--keep-all-components", action="store_true",
                    help="Do not reduce the mask to its largest connected component")
    ap.add_argument("--apply-mask", default=None,
                    help="Reuse the mask from a previous run's trim.json instead of "
                         "computing one, so derived products get an identical cut")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be kept and exit without writing")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- mask

def count_cells(path, cell):
    """Point count per XY cell. Returns (counts 2-D, x0, y0) with cell (i,j) at x0+i*cell."""
    import laspy

    with laspy.open(path) as f:
        h = f.header
        x0 = np.floor(h.mins[0] / cell) * cell
        y0 = np.floor(h.mins[1] / cell) * cell
        nx = int(np.floor(h.maxs[0] / cell) - np.floor(h.mins[0] / cell)) + 1
        ny = int(np.floor(h.maxs[1] / cell) - np.floor(h.mins[1] / cell)) + 1
        counts = np.zeros(nx * ny, dtype=np.int64)
        for pts in f.chunk_iterator(CHUNK):
            i = ((np.asarray(pts.x) - x0) / cell).astype(np.int64)
            j = ((np.asarray(pts.y) - y0) / cell).astype(np.int64)
            np.clip(i, 0, nx - 1, out=i)
            np.clip(j, 0, ny - 1, out=j)
            counts += np.bincount(i * ny + j, minlength=nx * ny)
    return counts.reshape(nx, ny), float(x0), float(y0)


def disk(radius_cells):
    r = int(radius_cells)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def build_mask(counts, cell, min_points, close_m, fill_holes, dilate, largest_only):
    from scipy import ndimage

    steps = {}
    mask = counts >= min_points
    steps["above_threshold"] = int(mask.sum())
    if not mask.any():
        raise RuntimeError("no cell reaches %d points -- --min-points is too high for a %g m "
                           "cell on this cloud" % (min_points, cell))

    if close_m > 0:
        r = max(int(round(close_m / cell)), 1)
        mask = ndimage.binary_closing(mask, structure=disk(r))
        steps["after_closing"] = int(mask.sum())

    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
        steps["after_fill_holes"] = int(mask.sum())

    if largest_only:
        lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
        steps["n_components"] = int(n)
        if n > 1:
            sizes = ndimage.sum(mask, lab, range(1, n + 1))
            mask = lab == (int(np.argmax(sizes)) + 1)
            steps["dropped_components"] = int(n - 1)
            steps["largest_component_cells"] = int(mask.sum())

    if dilate > 0:
        mask = ndimage.binary_dilation(mask, structure=disk(dilate))
        steps["after_dilation"] = int(mask.sum())

    return mask, steps


# --------------------------------------------------------------------------- application

def apply_mask(src, dst, mask, cell, x0, y0):
    """Rewrite src into dst keeping only points whose XY cell is set. Header cloned."""
    import laspy

    nx, ny = mask.shape
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
        kept = 0
        with laspy.open(dst, mode="w", header=hout) as w:
            for pts in f.chunk_iterator(CHUNK):
                i = ((np.asarray(pts.x) - x0) / cell).astype(np.int64)
                j = ((np.asarray(pts.y) - y0) / cell).astype(np.int64)
                keep = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
                keep[keep] = mask[i[keep], j[keep]]
                sel = pts[keep]
                if len(sel):
                    w.write_points(sel)
                    kept += len(sel)
    return kept


def mask_to_record(mask, cell, x0, y0):
    """Sparse cell list, so the same cut can be reapplied without the source cloud."""
    ii, jj = np.nonzero(mask)
    return {"cell": cell, "x0": x0, "y0": y0,
            "shape": list(mask.shape),
            "cells_i": ii.astype(int).tolist(),
            "cells_j": jj.astype(int).tolist()}


def record_to_mask(rec):
    mask = np.zeros(tuple(rec["shape"]), dtype=bool)
    mask[np.array(rec["cells_i"], dtype=np.int64),
         np.array(rec["cells_j"], dtype=np.int64)] = True
    return mask, float(rec["cell"]), float(rec["x0"]), float(rec["y0"])


# --------------------------------------------------------------------------- entry point

def main(argv=None):
    args = parse_args(argv)
    import laspy

    src = os.path.abspath(args.input)
    if not os.path.exists(src):
        sys.exit("Not found: %s" % src)
    stem = src[:-4] if src.lower().endswith(".las") else src
    out = args.output or (stem + "_trimmed.las")
    out_stem = out[:-4] if out.lower().endswith(".las") else out

    n_src = laspy.open(src).header.point_count
    print("source   : %s  (%d points)" % (src, n_src))

    if args.apply_mask:
        rec = json.load(open(args.apply_mask))
        mask, cell, x0, y0 = record_to_mask(rec["mask"])
        steps = {"reused_from": args.apply_mask}
        print("mask     : reused from %s" % args.apply_mask)
    else:
        cell = args.cell
        counts, x0, y0 = count_cells(src, cell)
        occupied = int((counts > 0).sum())
        mask, steps = build_mask(counts, cell, args.min_points, args.close,
                                 not args.no_fill_holes, args.dilate,
                                 not args.keep_all_components)
        kept_cells = int(mask.sum())
        # What the threshold alone would have cost, before any morphology.
        lost_pts = int(counts[(counts > 0) & (counts < args.min_points)].sum())
        print("mask     : %g m cells, keep >= %d points" % (cell, args.min_points))
        print("           %d occupied -> %d retained (%.1f%% of occupied area)"
              % (occupied, kept_cells, 100.0 * kept_cells / max(occupied, 1)))
        print("           threshold alone would drop %d points (%.3f%%)"
              % (lost_pts, 100.0 * lost_pts / max(n_src, 1)))
        for k, v in steps.items():
            print("           %-24s %s" % (k, v))
        print("           retained area %.0f m2" % (kept_cells * cell * cell))

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    kept = apply_mask(src, out, mask, cell, x0, y0)
    print("output   : %s" % out)
    print("           %d points kept, %d dropped (%.3f%%)"
          % (kept, n_src - kept, 100.0 * (n_src - kept) / max(n_src, 1)))

    hout = laspy.open(out).header
    print("           extent %.1f x %.1f x %.1f m"
          % (hout.maxs[0] - hout.mins[0], hout.maxs[1] - hout.mins[1],
             hout.maxs[2] - hout.mins[2]))

    json.dump(stamp({
        "source": src,
        "output": out,
        "points_in": int(n_src),
        "points_out": int(kept),
        "points_dropped": int(n_src - kept),
        "fraction_dropped": float((n_src - kept) / max(n_src, 1)),
        "parameters": {"cell": cell, "min_points": args.min_points,
                       "close_m": args.close, "fill_holes": not args.no_fill_holes,
                       "dilate_cells": args.dilate,
                       "largest_component_only": not args.keep_all_components},
        "steps": steps,
        "retained_cells": int(mask.sum()),
        "retained_area_m2": float(int(mask.sum()) * cell * cell),
        "mask": mask_to_record(mask, cell, x0, y0),
    }, "trim_cloud.py"), open(out_stem + ".trim.json", "w"))
    print("           record  %s" % (out_stem + ".trim.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
