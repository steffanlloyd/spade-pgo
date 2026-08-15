#!/usr/bin/env python3
"""
GNSS residual for a named trajectory CSV, rather than the one sanity_check picks by convention.

Needed because assemble.py writes <output>.traj.csv for the OPTIMISED poses and
<output>.traj_odom.csv for the odometry-only poses, beside every product -- so the pair sitting
next to <id>_map_odom.las is byte-identical to the pair next to <id>_map.las. Running
sanity_check on the odometry cloud therefore measures the optimised trajectory, not the
odometry one. Filling drift_odom_m and drift_optimised_m needs both measured against the same
external reference, which is what this does.

    traj_gnss.py TRAJ.csv --bags DIR --crs EPSG
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sanity_check as sc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--bags", required=True)
    ap.add_argument("--crs", default="EPSG:5972")
    a = ap.parse_args()

    import pyproj
    traj = sc.load_traj(a.traj)
    if traj is None:
        sys.exit("no rows in %s" % a.traj)
    sc.check_gnss(traj, a.bags, pyproj.CRS.from_user_input(a.crs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
