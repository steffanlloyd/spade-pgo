#pragma once

#include <atomic>
#include <string>
#include <Eigen/Geometry>

struct PGOParams{
    struct sc{
        bool enabled;
        double distance_threshold; // m
        double max_radius; // m
        double voxel_size; // m
    } sc;
    struct icp{
        std::string algorithm; // icp, gicp, ndt, small_gicp, or small_vgicp
        double voxel_size; // m
        int num_kf_accumulate_past;
        int num_kf_accumulate_now;
        double fitness_threshold; // average squared distances between matches
        double min_inlier_ratio; // minimum ratio of inliers to total correspondences (0 = disabled)
        double max_correspondence_distance; // m
        int max_iterations; // max ICP iterations
        double max_height_above_ground; // m, filter points above this height relative to min z (0 = disabled)
        double max_radius_from_keyframe; // m, filter points beyond this distance from keyframe (0 = disabled)
        bool zero_init_z; // seed ICP with the keyframes at equal height
        bool save_pointclouds;
        bool publish_pointclouds;
    } icp;
    struct graph{
        double prior_noise_rot; // rad
        double prior_noise_lin; // m
        double kf_gap_lin; // m
        double kf_gap_rot; // rad
        double lio_noise_rot; // rad
        double lio_noise_lin; // m
        double orientation_noise; // rad
        bool use_gnss_altitude;
        double gnss_min_initialization_distance;
        double gnss_time_delta; // s, max keyframe-to-fix timestamp gap for a GNSS match
        double gps_noise_threshold;
        double gps_noise_scale;
        double gps_noise_z_scale;
        double loop_closure_noise_scale;
    } graph;
    struct near_kf{
        bool enabled;
        double distance_threshold;
        double min_consecutive_kf_distance;
        int min_kf_seperation;
    } near_kf;
    struct visualize{
        double voxel_size;
    } visualize;
    struct ros{
        std::string gps_topic;          // Empty string "" disables GNSS
        std::string orientation_topic;  // Empty string "" disables external orientation
        std::string orientation_msg_type; // "odometry" or "quaternion"
        std::string pointcloud_topic;
        std::string lio_odometry_topic;
        std::string save_directory;
    } ros;
    struct extrinsics{
        Eigen::Isometry3d T_body_lidar = Eigen::Isometry3d::Identity();
    } extrinsics;
    
    // Helper methods to check if features are enabled.
    // Backed by atomics rather than the topic strings: the orchestrator can swap topics
    // per session via ReinitSession while the process_pg thread is reading these.
    bool useGNSS() const { return use_gnss_.load(std::memory_order_relaxed); }
    bool useExternalOrientation() const { return use_orientation_.load(std::memory_order_relaxed); }

    /// Recompute the feature flags from the current topic strings. Call after changing them.
    void refreshFeatureFlags() {
        use_gnss_.store(!ros.gps_topic.empty(), std::memory_order_relaxed);
        use_orientation_.store(!ros.orientation_topic.empty(), std::memory_order_relaxed);
    }

private:
    std::atomic<bool> use_gnss_{false};
    std::atomic<bool> use_orientation_{false};
};
