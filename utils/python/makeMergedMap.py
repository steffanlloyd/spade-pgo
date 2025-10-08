import os, re, copy, argparse
import numpy as np
import open3d as o3d
from numpy import linalg as LA
import laspy
# import pypcd  # using a py3-compatible fork
from pypcd import pypcd  # <- not needed if you use `import pypcd`

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
args = parser.parse_args()

data_dir = args.input if args.input.endswith('/') else args.input + '/'
scan_dir = os.path.join(data_dir, "scans")

# --- PARAMETERS ---
NODE_SKIP = 1
INTENSITY_COLOR_MAX = 200.0
NEAR_REMOVAL = True
NEAR_THRESH = 2.0  # meters

# Pose convention toggle:
# If your saved poses are T_world_from_lidar (common), keep True.
# If they are T_lidar_from_world, set False (we will invert before applying).
POSE_IS_WORLD_FROM_LIDAR = True

# Optional fixed extrinsic between LiDAR and the pose's reference frame.
# If not needed, keep as identity.
T_body_lidar = np.eye(4)  # fill with your calibrated extrinsic if needed


def parse_kf_index(filename:str) -> int:
    """
    Extracts the integer index from names like '000123.pcd', 'kf_000123.pcd', etc.
    Returns -1 if not found.
    """
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else -1


# --- Load scan file list and map to their keyframe index ---
all_files = [f for f in os.listdir(scan_dir) if f.lower().endswith(".pcd")]
if not all_files:
    raise FileNotFoundError(f"No .pcd files in {scan_dir}")

# Build list of (kf_idx, filename) and sort by kf_idx
kf_files = []
for f in all_files:
    idx = parse_kf_index(f)
    if idx >= 0:
        kf_files.append((idx, f))
    else:
        print(f"WARNING: couldn't parse index from {f}; skipping.")

if not kf_files:
    raise RuntimeError("No parsable keyframe indices in scans/ filenames.")

kf_files.sort(key=lambda x: x[0])  # numeric order

# --- Load poses (KITTI 3x4 per line), and offsets if present ---
poses = []
offset_x = 0.0
offset_y = 0.0
with open(os.path.join(data_dir, "optimized_poses.txt"), "r") as f:
    for line in f:
        s = line.strip()
        if not s:
            continue
        # offsets line? (OPTIONAL)
        if "OFFSET_X" in s.upper() or "OFFSET_Y" in s.upper():
            parts = s.replace(":", " ").replace(",", " ").split()
            for i, tok in enumerate(parts):
                T = tok.upper()
                if T == "OFFSET_X" and i + 1 < len(parts):
                    offset_x = float(parts[i + 1])
                elif T == "OFFSET_Y" and i + 1 < len(parts):
                    offset_y = float(parts[i + 1])
            continue

        # pose line: 12 numbers
        nums = s.split()
        if len(nums) != 12:
            # ignore junk lines safely
            continue
        M = np.array([float(x) for x in nums], dtype=np.float64).reshape(3, 4)
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = M
        poses.append(T)

num_poses = len(poses)
if num_poses == 0:
    raise RuntimeError("No poses parsed from optimized_poses.txt")

print(f"Loaded {len(kf_files)} scans and {num_poses} poses. Using offsets ({offset_x}, {offset_y}).")

# Truncate or filter to matches only
usable = [(idx, fname) for (idx, fname) in kf_files if 0 <= idx < num_poses]
missing_pose = [idx for (idx, _) in kf_files if not (0 <= idx < num_poses)]
if missing_pose:
    print(f"WARNING: {len(missing_pose)} scans have no matching pose index (e.g., {missing_pose[:5]}). They will be skipped.")

if not usable:
    raise RuntimeError("No scans have a matching pose; check indices / filenames.")

# Sanity: check first rotation looks proper
R0 = poses[usable[0][0]][:3,:3]
print("R0 orthonormal?:", np.allclose(R0 @ R0.T, np.eye(3), atol=1e-3), "det(R0)=", np.linalg.det(R0))

def pose_for_idx(idx:int) -> np.ndarray:
    T = poses[idx]
    # Compose with extrinsic if needed:
    # world_from_lidar = world_from_body @ body_from_lidar
    if POSE_IS_WORLD_FROM_LIDAR:
        return T @ T_body_lidar
    else:
        # if saved as lidar_from_world, invert; also compose extrinsic consistently
        return np.linalg.inv(T) @ T_body_lidar

def o3d_transform_copy(pcd:o3d.geometry.PointCloud, T:np.ndarray) -> o3d.geometry.PointCloud:
    q = o3d.geometry.PointCloud(pcd)  # copy
    q.transform(T)                    # in-place
    return q

# --- Build global arrays dynamically ---
xyz_chunks = []
int_chunks = []

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

    # Load geometry with Open3D
    pcd = o3d.io.read_point_cloud(scan_path)
    if len(pcd.points) == 0:
        print(f"WARNING: empty point cloud {fname}; skipping.")
        processed += 1
        continue

    # Local copy before transform for near-removal
    xyz_local = np.asarray(pcd.points, dtype=np.float32)

    # Load intensities with pypcd (Open3D ignores intensity)
    pc = pypcd.PointCloud.from_path(scan_path)
    if 'intensity' in pc.pc_data.dtype.names:
        intens = pc.pc_data['intensity'].astype(np.float32)
    else:
        # fabricate zeros if missing
        intens = np.zeros((xyz_local.shape[0],), dtype=np.float32)

    # Range filter (optional)
    if NEAR_REMOVAL:
        ranges = LA.norm(xyz_local, axis=1)
        eff = np.where(ranges > NEAR_THRESH)[0]
        if eff.size == 0:
            processed += 1
            continue
        pcd = pcd.select_by_index(eff)
        intens = intens[eff]

    # Transform to world
    T_w_l = pose_for_idx(idx)
    pcd_w = o3d_transform_copy(pcd, T_w_l)
    xyz = np.asarray(pcd_w.points, dtype=np.float32)

    # Accumulate
    xyz_chunks.append(xyz)
    int_chunks.append(intens.reshape(-1, 1))
    processed += 1

if not xyz_chunks:
    raise RuntimeError("No points assembled; check filters and indices.")

np_xyz_all = np.vstack(xyz_chunks)
np_intensity_all = np.vstack(int_chunks)

# Apply planar offsets (if you prefer header offsets, skip this and set header.offsets instead)
np_xyz_all[:, 0] += offset_x
np_xyz_all[:, 1] += offset_y

# --- Save LAS ---
las_file_path = os.path.join(data_dir, f"map.las")
header = laspy.LasHeader(point_format=3, version="1.2")
# choose precision; adjust as needed
header.scales  = np.array([0.001, 0.001, 0.001])  # 1 mm resolution
header.offsets = np.array([0.0, 0.0, 0.0])

las = laspy.LasData(header)
las.x = np_xyz_all[:, 0]
las.y = np_xyz_all[:, 1]
las.z = np_xyz_all[:, 2]
las.intensity = np.clip(np_intensity_all[:, 0], 0, 65535).astype(np.uint16)
las.write(las_file_path)
print(f"LAS file saved: {las_file_path}  (points: {np_xyz_all.shape[0]})")
