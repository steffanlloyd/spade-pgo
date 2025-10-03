import os
import sys
import time
import copy
from io import StringIO
import argparse
import pypcd # for the install, use this command: python3.x (use your python ver) -m pip install --user git+https://github.com/DanielPollithy/pypcd.git
from pypcd import pypcd
import numpy as np
from numpy import linalg as LA
import open3d as o3d
import re

import laspy  # Import the laspy library to handle LAS format

from pypcdMyUtils import *

script_dir = os.path.dirname(os.path.realpath(__file__))
jet_table = np.load(os.path.join(script_dir, 'jet_table.npy'))
bone_table = np.load(os.path.join(script_dir, 'bone_table.npy'))

color_table = jet_table
color_table_len = color_table.shape[0]
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input directory", required=True)
args = parser.parse_args()

##########################
# User only consider this block
##########################
data_dir = args.input
# Make sure data_dir ends with a '/'
if not data_dir.endswith('/'):
    data_dir += '/'

node_skip = 1

num_points_in_a_scan = 100000  # for reservation (save faster) // e.g., use 150000 for 128 ray lidars, 100000 for 64 ray lidars, 30000 for 16 ray lidars, if error occured, use the larger value.

is_live_vis = False  # recommend to use false 
is_o3d_vis = False
intensity_color_max = 200

is_near_removal = True
thres_near_removal = 2  # meter (to remove platform-myself structure ghost points)

##########################

scan_dir = os.path.join(data_dir, "scans")
scan_files = os.listdir(scan_dir) 
scan_files = [f for f in os.listdir(scan_dir) if f.lower().endswith(".pcd")]
scan_files.sort()
scan_idx_range_to_stack = [0, len(scan_files)-1]

poses = []
f = open(data_dir + "optimized_poses.txt", 'r')
offset_x = 0
offset_y = 0
while True:
    line = f.readline()
    if not line: break
    if line.startswith("OFFSET_X:"):
        tokens = line.split()
        # tokens[1] and tokens[3] should contain the offset values.
        offset_x = float(tokens[1])
        offset_y = float(tokens[3])
        print("Read offsets:", offset_x, offset_y)
        continue
    pose_SE3 = np.asarray([float(i) for i in line.split()])
    pose_SE3 = np.vstack((np.reshape(pose_SE3, (3, 4)), np.asarray([0, 0, 0, 1])))
    poses.append(pose_SE3)
f.close()

assert (scan_idx_range_to_stack[1] > scan_idx_range_to_stack[0])
print("Merging scans from", scan_idx_range_to_stack[0], "to", scan_idx_range_to_stack[1])

if(is_live_vis):
    vis = o3d.visualization.Visualizer() 
    vis.create_window('Map', visible=True) 

nodes_count = 0
pcd_combined_for_vis = o3d.geometry.PointCloud()
pcd_combined_for_save = None

# manually reserve memory for fast write  
num_all_points_expected = int(num_points_in_a_scan * np.round((scan_idx_range_to_stack[1] - scan_idx_range_to_stack[0]) / node_skip))

np_xyz_all = np.empty([num_all_points_expected, 3])
np_intensity_all = np.empty([num_all_points_expected, 1])
curr_count = 0

for node_idx in range(len(scan_files)):
    if(node_idx < scan_idx_range_to_stack[0] or node_idx >= scan_idx_range_to_stack[1]):
        continue

    nodes_count += 1
    if(nodes_count % node_skip != 0): 
        if(node_idx != scan_idx_range_to_stack[0]): # to ensure the vis init 
            continue

    scan_pose = poses[node_idx]

    scan_path = os.path.join(scan_dir, f"{node_idx+1:06d}.pcd") # scan names use 1-based index
    if not os.path.exists(scan_path):
        print(f"Warning: cannot find scan file for index {node_idx} in PCD format. Skipping frame.")
        continue
    
    print(f"Reading keyframe scan index {node_idx} from {scan_path}. Current point count: {curr_count}.")

    # Sanity checks
    # Check rotation validity and det ~ +1 (proper rotation)
    R = scan_pose[:3,:3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
        print(f"Warning! R is not orthonormal, det(R0)={np.linalg.det(R)}")

    # Check that sequential pose jumps are reasonable (not meters->kilometers)
    if node_idx > 1:
        tim1 = poses[node_idx-1][:3,3]; ti = poses[node_idx][:3,3]
        dist = np.linalg.norm(ti - tim1)
        if dist > 10.0:  # 10 meters
            print(f"Warning: sequential pose motion is very large (m): {dist}")

    # scan_path = os.path.join(scan_dir, scan_files[node_idx])
    scan_pcd = o3d.io.read_point_cloud(scan_path)
    scan_xyz_local = copy.deepcopy(np.asarray(scan_pcd.points))

    scan_pypcd_with_intensity = pypcd.PointCloud.from_path(scan_path)
    scan_intensity = scan_pypcd_with_intensity.pc_data['intensity']
    scan_intensity_colors_idx = np.round((color_table_len - 1) * np.minimum(1, np.maximum(0, scan_intensity / intensity_color_max)))
    scan_intensity_colors = color_table[scan_intensity_colors_idx.astype(int)]

    scan_pcd_global = scan_pcd.transform(scan_pose)  # global coord, note that this is not deepcopy
    scan_pcd_global.colors = o3d.utility.Vector3dVector(scan_intensity_colors)
    scan_xyz = np.asarray(scan_pcd_global.points)

    scan_intensity = np.expand_dims(scan_intensity, axis=1) 
    scan_ranges = LA.norm(scan_xyz_local, axis=1)

    if(is_near_removal):
        eff_idxes = np.where(scan_ranges > thres_near_removal)
        scan_xyz = scan_xyz[eff_idxes[0], :]
        scan_intensity = scan_intensity[eff_idxes[0], :]

        scan_pcd_global = scan_pcd_global.select_by_index(eff_idxes[0])

    if(is_o3d_vis):
        pcd_combined_for_vis += scan_pcd_global  # open3d pointcloud class append is fast 

    if is_live_vis:
        if(node_idx == scan_idx_range_to_stack[0]):  # to ensure the vis init 
            vis.add_geometry(pcd_combined_for_vis) 

        vis.update_geometry(pcd_combined_for_vis)
        vis.poll_events()
        vis.update_renderer()

    # save 
    np_xyz_all[curr_count:curr_count + scan_xyz.shape[0], :] = scan_xyz
    np_intensity_all[curr_count:curr_count + scan_xyz.shape[0], :] = scan_intensity

    curr_count += scan_xyz.shape[0]


if(is_o3d_vis):
    print("draw the merged map.")
    o3d.visualization.draw_geometries([pcd_combined_for_vis])


# save LAS file
np_xyz_all = np_xyz_all[0:curr_count, :]
np_intensity_all = np_intensity_all[0:curr_count, :]

np_xyz_all[:, 0] += offset_x
np_xyz_all[:, 1] += offset_y

# Create a laspy file to save data in LAS format
las_file_path = data_dir + "map_" + str(scan_idx_range_to_stack[0]) + "_to_" + str(scan_idx_range_to_stack[1]) + "_with_intensity.las"

# Create a Laspy header and LAS file
header = laspy.LasHeader(point_format=3, version='1.2')  # Using version '1.2' and point format 3
las = laspy.create(point_format=3, file_version='1.2')

# Set the coordinates and intensity values
las.x = np_xyz_all[:, 0]
las.y = np_xyz_all[:, 1]
las.z = np_xyz_all[:, 2]
las.intensity = np_intensity_all[:, 0].astype(np.uint16)  # Ensure intensity is an unsigned 16-bit integer

# Write to the file
las.write(las_file_path)

print(f"LAS file saved: {las_file_path}")

