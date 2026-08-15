#!/usr/bin/env python3
"""
Which version of this code produced a file.

The release clouds were post-processed by scripts that were not committed at the time, so their
records named no code version and 52 of them had to be stamped by hand afterwards. Every record
written from here on states its own provenance.

    from provenance import stamp
    json.dump(stamp(rec, "trim_cloud.py"), open(out, "w"), indent=2)
"""

import datetime
import os
import subprocess


def git_commit(path=None):
    """
    HEAD of the checkout containing this script, or None.

    spade-pgo is a submodule: its .git is a file pointing into the parent repo's .git/modules,
    so rev-parse fails whenever only ros1_ws/ is bind-mounted into the container. SPADE_PGO_COMMIT
    lets the caller supply the hash rather than have us silently record nothing.
    """
    env = os.environ.get("SPADE_PGO_COMMIT")
    if env:
        return env.strip()
    path = path or os.path.dirname(os.path.abspath(__file__))
    try:
        # safe.directory: the checkout is bind-mounted from the host and owned by a different
        # uid inside the container, which git otherwise refuses to read.
        out = subprocess.check_output(
            ["git", "-c", "safe.directory=*", "-C", path, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(
            ["git", "-c", "safe.directory=*", "-C", path, "diff", "--quiet", "HEAD"],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return out + ("-dirty" if dirty else "")
    except Exception:
        return None


def stamp(rec, script):
    """Add tool, commit and timestamp to a record dict, in place."""
    rec["tool"] = "spade_pgo/scripts/" + script
    rec["git_commit"] = git_commit()
    rec["generated_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return rec
