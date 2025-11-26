
import os, re, argparse
import numpy as np
import open3d as o3d
from numpy import linalg as LA
import laspy
from laspy import ExtraBytesParams
from pypcd import pypcd

INTENSITY_COLOR_MAX = 200.0  # same as earlier parameter

# --- ARGUMENTS ---
parser = argparse.ArgumentParser(description="Assemble point clouds into LAS and visualize.")
parser.add_argument("-i", "--input", required=True, help="Input data directory")
parser.add_argument("-o", "--output", required=False, help="Output LAS file path")
parser.add_argument("-s", "--start-frame", type=int, default=None, help="Start keyframe index (inclusive)")
parser.add_argument("-e", "--end-frame", type=int, default=None, help="End keyframe index (inclusive)")
parser.add_argument("-ir", "--ignore-range", type=str, default=None,
                    help="Ignore keyframe ranges, e.g., '100-150' or multiple like '50-60,100-150'")
parser.add_argument("-v", "--visualize", action="store_true",
                    help="Enable visualization of merged point cloud")
args = parser.parse_args()

data_dir = args.input if args.input.endswith('/') else args.input + '/'
scan_dir = os.path.join(data_dir, "scans")

output_path = args.output if args.output else os.path.join(data_dir, "map.las")
start_frame = args.start_frame
end_frame = args.end_frame

# --- PARAMETERS ---
NODE_SKIP = 1
NEAR_REMOVAL = True
NEAR_THRESH = 2.0  # meters
POSE_IS_WORLD_FROM_LIDAR = True
T_body_lidar = np.eye(4)

def parse_kf_index(filename: str) -> int:
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


ignore_ranges = parse_ignore_ranges(args.ignore_range)

def is_ignored(idx: int) -> bool:
    for (a, b) in ignore_ranges:
        if a <= idx <= b:
            return True
    return False

# --- Load scan file list ---
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

# --- Apply frame filtering ---
kf_files = [(idx, fname) for (idx, fname) in kf_files
            if (start_frame is None or idx >= start_frame) and
               (end_frame is None or idx <= end_frame) and
               not is_ignored(idx)]

if not kf_files:
    raise RuntimeError("No parsable keyframe indices in scans/ filenames.")

kf_files.sort(key=lambda x: x[0])

# --- Load poses ---
poses = []
offset_x = 0.0
offset_y = 0.0

with open(os.path.join(data_dir, "optimized_poses.txt"), "r") as f:
    for line in f:
        s = line.strip()
        if not s:
            continue
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
        if len(nums) != 12:
            continue
        M = np.array([float(x) for x in nums], dtype=np.float64).reshape(3, 4)
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = M
        poses.append(T)

num_poses = len(poses)
if num_poses == 0:
    raise RuntimeError("No poses parsed from optimized_poses.txt")

usable = [(idx, fname) for (idx, fname) in kf_files if 0 <= idx < num_poses]
if not usable:
    raise RuntimeError("No scans have a matching pose; check indices / filenames.")

print(f"Loaded {len(usable)} scans and {num_poses} poses. Using offsets ({offset_x}, {offset_y}).")
print(f"Output LAS will be saved to: {output_path}")
if start_frame or end_frame:
    print(f"Frame filter applied: start={start_frame}, end={end_frame}")

def pose_for_idx(idx: int) -> np.ndarray:
    T = poses[idx]
    if POSE_IS_WORLD_FROM_LIDAR:
        return T @ T_body_lidar
    else:
        return np.linalg.inv(T) @ T_body_lidar

def o3d_transform_copy(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    q = o3d.geometry.PointCloud(pcd)
    q.transform(T)
    return q

# --- Build global arrays ---
kf_chunks = []  # per-point keyframe indices
xyz_chunks = []
int_chunks = []
labels = []

processed = 0
for idx, fname in usable:
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
    intens = pc.pc_data['intensity'].astype(np.float32) if 'intensity' in pc.pc_data.dtype.names else np.zeros((xyz_local.shape[0],), dtype=np.float32)

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

    # Per-point keyframe index (uint16)
    n_pts = xyz.shape[0]
    kf_chunks.append(np.full((n_pts, 1), idx, dtype=np.uint16))

    processed += 1

if not xyz_chunks:
    raise RuntimeError("No points assembled; check filters and indices.")

np_xyz_all = np.vstack(xyz_chunks)
np_intensity_all = np.vstack(int_chunks)
np_xyz_all[:, 0] += offset_x
np_xyz_all[:, 1] += offset_y
np_kf_all = np.vstack(kf_chunks)  # shape (N, 1), dtype uint16

# --- Save LAS ---
header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001, 0.001, 0.001])
header.offsets = np.array([0.0, 0.0, 0.0])
extra_dim = ExtraBytesParams(name="keyframe", type=np.uint16)
header.add_extra_dim(extra_dim)

las = laspy.LasData(header)
las.x = np_xyz_all[:, 0]
las.y = np_xyz_all[:, 1]
las.z = np_xyz_all[:, 2]
las.intensity = np.clip(np_intensity_all[:, 0], 0, 65535).astype(np.uint16)
las["keyframe"] = np_kf_all[:, 0]  # already uint16
las.write(output_path)
print(f"LAS file saved: {output_path}  (points: {np_xyz_all.shape[0]})")

# --- Visualization ---
if args.visualize:
    print("Visualizing merged point cloud...")
    # Create Open3D point cloud from np_xyz_all
    pcd_global = o3d.geometry.PointCloud()
    pcd_global.points = o3d.utility.Vector3dVector(np_xyz_all)
    
    # Optional: color by intensity
    norm_intensity = np.clip(np_intensity_all[:, 0] / INTENSITY_COLOR_MAX, 0, 1)
    colors = np.stack([norm_intensity, norm_intensity, norm_intensity], axis=1)
    pcd_global.colors = o3d.utility.Vector3dVector(colors)

    # Ensure DISPLAY is set for SSH sessions
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"  # fallback to default X display

    o3d.visualization.draw_geometries([pcd_global])
