# SPADE_SLAM

- A real-time LiDAR SLAM package that integrates LIO and ScanContext. 
    - **Any LIO algorithm** for odometry
    - **ScanContext** for coarse global localization that can deal with big drifts (i.e., place recognition as kidnapped robot problem without initial pose)
    - and iSAM2 of GTSAM is used for pose-graph optimization. 

## Dependencies
- ROS (geometry_msgs, nav_msgs, sensor_msgs, roscpp, rospy, std_msgs, tf)
- PCL
- OpenMP
- GTSAM
- GeographicLib
- laspy

## Features 
1. A strong place recognition and loop closing 
2. A modular implementation 
3. Complete GPS integration
4. Implementation of GeographicLib for easy conversion to various global coordinate systems
5. Easy conversion of point cloud data into the UTM coordinate system

## Running the SLAM

To run the optimization, launch both the pose graph optimizer and fastlio. After building successfully, launch the graphslam launch file:
```bash
roslaunch spade graphslam_headsens.launch
```
This file will use the parameters in `spade/config/pgo_headsens.yaml`. The config parameters are well-documented in this config file. Key parameters to tune are to properly specify the incoming topics (imu and gps). The rest are often good with the defaults, however the GNSS covariances may need to be adjusted if the algorithm is relying too much/too little on GNSS (depends on the accuracy of the covariance of the GNSS measurements from your sensor).

Once the node is running, RViz should start. Play back your rosbag, and the mapping will proceed.

Then, just play back your rosbag.

## Assembling the bag

To assemble the bag, run the python script (within the docker) after the run
```
python3 ~/ros1_ws/src/vertliner-spade/SC-PGO/utils/python/makeMergedMap.py -i ~/save/pointclouds
```
A map file called `map.las` will be saved in the same folder as the input (`~/save/pointclouds`).

## Docker

A docker setup to run this more easily is available at [https://github.com/steffanlloyd/spade-pgo-docker](https://github.com/steffanlloyd/spade-pgo-docker)