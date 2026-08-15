#!/usr/bin/env python3
"""
Measure GNSS quality per bag, from a fixed topic priority.

The release statistics were first built by taking whichever NavSatFix topic appeared first in
each bag. That is not comparable across rows: our bags carry both a raw receiver fix and, on the
mavros platforms, an EKF-fused solution whose covariance is the filter's rather than the
receiver's. This script always prefers the raw fix, and says which topic it used, so a column of
sigma_h means the same thing in every row.

    gnss_probe.py BAG [BAG ...] [--json OUT]

Priority for position quality -- raw receiver fix first, fused only as a fallback:

    /septentrio_gnss/navsatfix          Campaign B, raw Septentrio
    /mavros/global_position/raw/fix     Campaigns A and C, raw GPS through mavros
    /mavros/global_position/global      fused EKF output; reported with fused=true

Priority for fix type and satellite count, which NavSatFix cannot carry:

    /mavros/gpsstatus/gps1/raw          mavros_msgs/GPSRAW
    /septentrio_gnss/gpsfix             gps_common/GPSFix

Campaign A has neither, so its satellite counts and fix-type shares are genuinely unavailable
rather than merely uncomputed -- the recording predates our carrying a status topic.

sigma_h is sqrt((cov_ee + cov_nn) / 2), the convention the published table already used; the
two components are equal in every mavros bag we have, so this reproduces the earlier figures
exactly while remaining meaningful if a receiver ever reports them separately.
"""

import argparse
import json
import os
import sys

import numpy as np

from provenance import stamp

FIX_TOPICS = ["/septentrio_gnss/navsatfix",
              "/mavros/global_position/raw/fix",
              "/mavros/global_position/global"]
FUSED = {"/mavros/global_position/global"}
STATUS_TOPICS = ["/mavros/gpsstatus/gps1/raw", "/septentrio_gnss/gpsfix"]
LIDAR_TOPICS = ["/livox/lidar", "/livox/lidar_192_168_1_154"]

# mavros_msgs/GPSRAW.fix_type
GPSRAW = {0: "no_fix", 1: "no_fix", 2: "2d", 3: "3d", 4: "dgps", 5: "rtk_float", 6: "rtk_fixed"}
# gps_common/GPSFix.status.status
GPSFIX = {-1: "no_fix", 0: "3d", 1: "sbas", 2: "gbas"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--json", default=None,
                    help="Write all records here (JSON: tool, git_commit, bags[])")
    return ap.parse_args(argv)


def probe(path):
    import rosbag

    rec = {"bag": os.path.abspath(path), "name": os.path.basename(path)}
    with rosbag.Bag(path) as b:
        info = b.get_type_and_topic_info().topics
        rec["duration_s"] = round(b.get_end_time() - b.get_start_time(), 1)

        rec["n_lidar_scans"] = sum(info[t].message_count for t in LIDAR_TOPICS if t in info)

        fix_topic = next((t for t in FIX_TOPICS if t in info), None)
        rec["gnss_topic"] = fix_topic
        rec["gnss_fused"] = fix_topic in FUSED if fix_topic else None
        if fix_topic:
            sig, nofix = [], 0
            for _, m, _ in b.read_messages(topics=[fix_topic]):
                if m.status.status < 0:
                    nofix += 1
                    continue
                c = m.position_covariance
                sig.append(np.sqrt((c[0] + c[4]) / 2.0))
            n = len(sig) + nofix
            rec["n_fixes"] = n
            rec["gnss_rate_hz"] = round(n / rec["duration_s"], 2) if rec["duration_s"] else None
            rec["gnss_median_sigma_h_m"] = round(float(np.median(sig)), 3) if sig else None
            rec["gnss_pct_nofix"] = round(100.0 * nofix / n, 2) if n else None

        st_topic = next((t for t in STATUS_TOPICS if t in info), None)
        rec["status_topic"] = st_topic
        if st_topic:
            sats, kinds = [], {}
            gpsraw = st_topic == "/mavros/gpsstatus/gps1/raw"
            for _, m, _ in b.read_messages(topics=[st_topic]):
                if gpsraw:
                    k = GPSRAW.get(m.fix_type, "unknown")
                    sats.append(m.satellites_visible)
                else:
                    k = GPSFIX.get(m.status.status, "unknown")
                    sats.append(m.status.satellites_used)
                kinds[k] = kinds.get(k, 0) + 1
            tot = sum(kinds.values())
            rec["gnss_median_sats"] = int(np.median(sats)) if sats else None
            rec["fix_type_pct"] = {k: round(100.0 * v / tot, 2) for k, v in sorted(kinds.items())}
            rec["gnss_pct_rtk_fixed"] = rec["fix_type_pct"].get("rtk_fixed", 0.0)
            rec["gnss_pct_rtk_float"] = rec["fix_type_pct"].get("rtk_float", 0.0)
            rec["gnss_pct_3d_or_dgps"] = round(
                sum(v for k, v in rec["fix_type_pct"].items()
                    if k in ("3d", "dgps", "sbas", "gbas", "rtk_float", "rtk_fixed")), 2)
        else:
            # No status topic in the recording -- not zero, unknown.
            for k in ("gnss_median_sats", "gnss_pct_rtk_fixed",
                      "gnss_pct_rtk_float", "gnss_pct_3d_or_dgps"):
                rec[k] = None
    return rec


def main(argv=None):
    args = parse_args(argv)
    out = []
    print("%-40s %-32s %6s %8s %7s %6s %s"
          % ("bag", "topic", "fixes", "sigma_h", "nofix%", "sats", "fix types"))
    for p in args.bags:
        if not os.path.exists(p):
            print("missing: %s" % p, file=sys.stderr)
            continue
        r = probe(p)
        out.append(r)
        print("%-40s %-32s %6s %8s %7s %6s %s"
              % (r["name"][:40],
                 (r.get("gnss_topic") or "-") + ("*" if r.get("gnss_fused") else ""),
                 r.get("n_fixes", "-"),
                 r.get("gnss_median_sigma_h_m", "-"),
                 r.get("gnss_pct_nofix", "-"),
                 r.get("gnss_median_sats") if r.get("gnss_median_sats") is not None else "n/a",
                 r.get("fix_type_pct") or "no status topic"))
    print("\n* = EKF-fused topic, covariance is the filter's not the receiver's")
    if args.json:
        json.dump(stamp({"bags": out}, "gnss_probe.py"), open(args.json, "w"), indent=2)
        print("record  %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
