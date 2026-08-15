#pragma once

#include <memory>
#include <vector>
#include <string>
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include <std_msgs/ColorRGBA.h>
#include <Eigen/Geometry>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>

#include "spade_pgo/PoseGraphManager.hpp"
#include "spade_pgo/PGOParams.hpp"

namespace spade_pgo {
// Forward declaration
class LoopClosureManager;

class Visualizer {
public:
    Visualizer(
        std::shared_ptr<PoseGraphManager> graph_manager,
        ros::NodeHandle& nh, 
        std::shared_ptr<PGOParams> params);
    ~Visualizer();

    // void publishLoopClosureMarkers();
    void publishGPSPathAndMarkers();
    void publishPath();
    void publishMap(int frame_skip = 1);
    void publishSensorCloud(const sensor_msgs::PointCloud2ConstPtr& cloud);
    void publishLCClouds(const pcl::PointCloud<PointType>::Ptr& cloud_curr,
                         const pcl::PointCloud<PointType>::Ptr& cloud_prev,
                         int kf_curr=-1, int kf_prev=-1);
    void publishLCMarkers();

    void setLoopClosureManager(std::shared_ptr<LoopClosureManager> lc_manager);

    static visualization_msgs::MarkerArray createMarkerArray(
        const std::vector<geometry_msgs::Point>& ptsA,
        const std::vector<geometry_msgs::Point>& ptsB,
        const Eigen::Vector4d& color = Eigen::Vector4d(1.0, 0.0, 0.0, 1.0),
        const std::string& frame_id = "/map",
        double line_width = 1);

    std::shared_ptr<PGOParams> params;

private:
    std::shared_ptr<PoseGraphManager> graph_manager_;
    ros::NodeHandle nh_;
    std::weak_ptr<LoopClosureManager> loop_closure_manager_;

    ros::Publisher publisher_path_;
    ros::Publisher publisher_odometry_;
    ros::Publisher publisher_lc_markers_;
    ros::Publisher publisher_gps_markers_;
    ros::Publisher publisher_gps_path_;
    ros::Publisher publisher_map_;
    ros::Publisher publisher_sensor_cloud_;
    ros::Publisher publisher_lc_cloud_curr_;
    ros::Publisher publisher_lc_cloud_prev_;

    tf::TransformBroadcaster tf_broadcaster_;

    // Leaf size for the map publish; the octree is built per call, as it holds the cloud.
    double map_voxel_size_;

    std::string lc_pcd_directory_;
};

}  // namespace spade_pgo