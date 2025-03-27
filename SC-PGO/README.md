# SC-A-LOAM

## What is SC-A-LOAM? 
- A real-time LiDAR SLAM package that integrates A-LOAM and ScanContext. 
    - **A-LOAM** for odometry (i.e., consecutive motion estimation)
    - **ScanContext** for coarse global localization that can deal with big drifts (i.e., place recognition as kidnapped robot problem without initial pose)
    - and iSAM2 of GTSAM is used for pose-graph optimization. 
- This package aims to show ScanContext's handy applicability. 
    - The only things a user should do is just to include `Scancontext.h`, call `makeAndSaveScancontextAndKeys` and `detectLoopClosureID`. 

## Dependencies
- ROS (geometry_msgs, nav_msgs, sensor_msgs, roscpp, rospy, rosbag, std_msgs, image_transport, cv_bridge, tf)
- PCL
- OpenCV
- Ceres
- OpenMP
- GTSAM
- GeographicLib
- laspy

## Features 
1.  A strong place recognition and loop closing 
    - We integrated ScanContext as a loop detector into A-LOAM, and ISAM2-based pose-graph optimization is followed. (see https://youtu.be/okML_zNadhY?t=313 to enjoy the drift-closing moment)
2. A modular implementation 
    - The only difference from A-LOAM is the addition of the `laserPosegraphOptimization.cpp` file. In the new file, we subscribe the point cloud topic and odometry topic (as a result of A-LOAM, published from `laserMapping.cpp`). That is, our implementation is generic to any front-end odometry methods. Thus, our pose-graph optimization module (i.e., `laserPosegraphOptimization.cpp`) can easily be integrated with any odometry algorithms such as non-LOAM family or even other sensors (e.g., visual odometry).  
    - <p align="center"><img src="picture/anypipe.png" width=800></p>
3. Complete GPS integration
4. Implementation of GeographicLib for easy conversion to various global coordinate systems

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
    roslaunch aloam_velodyne fastlio_mid360.launch #(or another launch file of your preference) 
```
- After acquiring the data, you must see within your "save_directory" a "Scans" folder and the three following files "odom_poses.txt", "optimized_poses.txt", and "times.txt". Within the utils/python folder you will find the "makeMergedMapLas.py" file which will help you register the individual scans to the optimized poses and merge them into a single las file.
