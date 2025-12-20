cmake_minimum_required(VERSION 2.8.3)
project(spade_pgo)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  tf
)

#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTSAM REQUIRED QUIET)
find_package(small_gicp REQUIRED)


catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs tf
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS include
)

include_directories(
  include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  /usr/include/GeographicLib
)

add_executable(spade_pgo_node 
  src/spade_pgo_node.cpp
  src/DataBuffer.cpp
  src/geometry.cpp
  src/LoopClosureManager.cpp
  src/NearKFDetector.cpp
  src/PoseGraphManager.cpp
  src/ros_helpers.cpp
  src/Visualizer.cpp
  include/scancontext/Scancontext.cpp
)
target_compile_options(spade_pgo_node 
  PRIVATE ${OpenMP_CXX_FLAGS}
)
target_link_libraries(spade_pgo_node 
  ${catkin_LIBRARIES} ${PCL_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  gtsam
  Geographic
  small_gicp
)




