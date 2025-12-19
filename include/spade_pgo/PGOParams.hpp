#pragma once

#include <string>

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
        double max_correspondence_distance; // m
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
    
    // Helper methods to check if features are enabled
    bool useGNSS() const { return !ros.gps_topic.empty(); }
    bool useExternalOrientation() const { return !ros.orientation_topic.empty(); }
};
