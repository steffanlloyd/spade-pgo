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

## How to use? 
- First, install the above mentioned dependencies. 
- Second, read the desired launch file (e.g. fastlio_mid360.launch) and familiarize with algorithm's parameters (explanations are provided within the file). Also modify the "save_directory" parameter to point to a directory of your preference where all the  output data will be saved.
- To run the package (*of course you also need to run your LIO module):
```
    mkdir -p ~/my_ws/src
    cd ~/my_ws/src
    git clone git@gitlab.eclipse.org:eclipse-research-labs/spade-project/opencall-1/olympian/vertliner-spade.git
    cd vertliner-spade
    git checkout devSLAM
    cd ../..
    catkin_make
    source devel/setup.bash
    roslaunch spade_pgo fastlio_mid360.launch 
```
- After acquiring the data, you must see within your "save_directory" a "Scans" folder and the three following files "odom_poses.txt", "optimized_poses.txt", and "times.txt". Within the utils/python folder you will find the "makeMergedMapLas.py" file which will help you register the individual scans to the optimized poses and merge them into a single las file.
