#!/usr/bin/env python3
"""
Cleanup LAS point cloud by spatial grid and keyframe clustering.

This script removes temporal ghosting / duplication artifacts in a LAS file
produced by SPADE-PGO by keeping only the dominant keyframe cluster per spatial cell.

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
"""

import argparse
import os
import numpy as np
import laspy
from collections import defaultdict
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Cleanup LAS by spatial grid and keyframe clustering")

    p.add_argument("-i", "--input", required=True, help="Input LAS file")
    p.add_argument("-o", "--output", default=None, help="Output LAS file")
    p.add_argument("-g", "--grid-size", type=float, default=10.0, help="Grid size in meters")
    p.add_argument("-w", "--kf-window", type=int, default=40, help="Keyframe clustering window")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    return p.parse_args()


from collections import defaultdict

def debug_keyframe_drone_mapping(keyframes, drone_ids, verbose=True):
    """
    Build and print mapping: keyframe_id -> set(drone_ids)
    """
    kf_to_drones = defaultdict(set)

    for kf, dr in zip(keyframes, drone_ids):
        kf_to_drones[int(kf)].add(int(dr))

    multi_drone_kfs = {kf: ds for kf, ds in kf_to_drones.items() if len(ds) > 1}

    print("\n=== Keyframe → Drone ID mapping summary ===")
    print(f"Total unique keyframes: {len(kf_to_drones)}")

    drone_counts = defaultdict(int)
    for ds in kf_to_drones.values():
        for d in ds:
            drone_counts[d] += 1

    print("Keyframes per drone:")
    for d in sorted(drone_counts):
        print(f"  Drone {d}: {drone_counts[d]} keyframes")

    if multi_drone_kfs:
        print("\nWARNING: Keyframes associated with MULTIPLE drone IDs!")
        print(f"Count: {len(multi_drone_kfs)}")

        if verbose:
            print("Examples (up to 20):")
            for kf in list(multi_drone_kfs.keys())[:20]:
                print(f"  Keyframe {kf} → drones {sorted(multi_drone_kfs[kf])}")
    else:
        print("\nOK: Each keyframe maps to exactly ONE drone_id")

    print("=========================================\n")

    return kf_to_drones



class KeyframeCluster:
    """Container for one keyframe cluster."""
    def __init__(self, kf, drone_id):
        self.keyframes = {kf}
        self.drone_id = drone_id

    def median(self):
        return int(np.median(list(self.keyframes)))

    def can_accept(self, kf, drone_id, window):
        if drone_id != self.drone_id:
            return False
        return abs(kf - self.median()) <= window

    def add(self, kf, drone_id):
        self.keyframes.add(kf)


def cluster_keyframes(kf_counts, kf_to_drone, window):
    """
    Cluster keyframes by temporal proximity, enforcing one keyframe per drone per cluster.

    Args:
        kf_counts: dict {kf_id: point_count}
        kf_to_drone: dict {kf_id: drone_id}
        window: max |kf - median| to merge

    Returns:
        list of KeyframeCluster
    """
    sorted_kfs = sorted(kf_counts.items(), key=lambda x: x[0], reverse=True)
    clusters = []

    for kf, _ in sorted_kfs:
        drone_id = kf_to_drone[kf]
        assigned = False

        for cluster in clusters:
            if cluster.can_accept(kf, drone_id, window):
                cluster.add(kf, drone_id)
                assigned = True
                break

        if not assigned:
            clusters.append(KeyframeCluster(kf, drone_id))

    return clusters


def main():
    args = parse_args()

    input_path = args.input
    output_path = args.output or os.path.splitext(input_path)[0] + "_cleaned.las"

    grid = args.grid_size
    window = args.kf_window

    print(f"Loading LAS: {input_path}")
    las = laspy.read(input_path)

    for dim in ("keyframe", "drone_id"):
        if dim not in las.point_format.extra_dimension_names:
            raise RuntimeError(f"LAS file missing required '{dim}' dimension")

    xyz = np.vstack((las.x, las.y, las.z)).T
    keyframes = las["keyframe"]
    drone_ids = las["drone_id"]
    
    # kf_to_drones = debug_keyframe_drone_mapping(keyframes, drone_ids)

    gx = np.floor(xyz[:, 0] / grid).astype(np.int32)
    gy = np.floor(xyz[:, 1] / grid).astype(np.int32)

    cells = defaultdict(list)
    for i in range(len(xyz)):
        cells[(gx[i], gy[i])].append(i)

    keep_mask = np.zeros(len(xyz), dtype=bool)

    print(f"Processing {len(cells)} grid cells")

    # for cell_idx, indices in cells.items():
    for cell_idx, indices in tqdm(
        cells.items(),
        total=len(cells),
        desc="Filtering point cloud by cell",
        unit="cell"):

        kfs = keyframes[indices]
        drones = drone_ids[indices]

        # Count points per keyframe
        unique_kfs, counts = np.unique(kfs, return_counts=True)
        kf_counts = dict(zip(unique_kfs.tolist(), counts.tolist()))

        if len(kf_counts) == 1:
            keep_mask[indices] = True
            continue

        # Map keyframe -> drone_id (unique per keyframe in LAS)
        kf_to_drone = {}
        for kf, dr in zip(kfs, drones):
            kf_to_drone[int(kf)] = int(dr)

        clusters = cluster_keyframes(kf_counts, kf_to_drone, window)

        def cluster_size(cluster):
            return sum(kf_counts[k] for k in cluster.keyframes)

        dominant = max(clusters, key=cluster_size)

        kept_in_cell = 0
        for i in indices:
            if keyframes[i] in dominant.keyframes:
                keep_mask[i] = True
                kept_in_cell += 1

        total_in_cell = len(indices)
        pct = 100.0 * kept_in_cell / total_in_cell

        if args.verbose:
            print(
                f"Cell {cell_idx}: "
                f"{len(kf_counts)} kfs → "
                f"{len(clusters)} clusters → "
                f"keep {sorted(dominant.keyframes)} "
                f"(drone {dominant.drone_id}). "
                f"Kept {kept_in_cell} / {total_in_cell} points "
                f"({pct:.1f}%)"
            )

    kept = np.count_nonzero(keep_mask)
    print(f"Keeping {kept:,} / {len(xyz):,} points ({100*kept/len(xyz):.1f}%)")

    # Create output LAS with filtered points
    out = laspy.LasData(las.header)

    # IMPORTANT: shrink point record first
    out.points = las.points[keep_mask]

    out.write(output_path)
    print(f"Saved cleaned LAS: {output_path}")



if __name__ == "__main__":
    main()
