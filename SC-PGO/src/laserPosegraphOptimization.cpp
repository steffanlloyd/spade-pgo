#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>

#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/NavSatFix.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <eigen3/Eigen/Dense>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <GeographicLib/UTMUPS.hpp>
#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LocalCartesian.hpp>
#include <GeographicLib/Geoid.hpp>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

#include "scancontext/Scancontext.h"


using namespace gtsam;

using std::cout;
using std::endl;

double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0; // large value means must add the first given frame.
double rotationAccumulated = 1000000.0; // large value means must add the first given frame.

bool isNowKeyFrame = false; 

Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init 
Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero 
gtsam::Pose3 T_odom_to_gps;

std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<sensor_msgs::NavSatFix::ConstPtr> gpsBuf;
std::queue<std::pair<int, int> > loopClosureCandidateBuf;
std::vector<std::pair<int, int>> loopClosureIdsTested;
std::vector<std::pair<int, int>> loopClosureIdsAdded;

// SL (Q): what is mBuf a mutex for?
std::mutex mBuf;
std::mutex mBufGPS;
std::mutex mKF;

double timeLaserOdometry = 0.0;
double timeLaser = 0.0;

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudMapAfterPGO(new pcl::PointCloud<PointType>());

std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds; 
std::vector<Pose6D> keyframePoses;
std::vector<Pose6D> keyframePosesUpdated;
std::vector<gtsam::GPSFactor> keyframeGpsFactor;
std::vector<double> keyframeTimes;
int recentIdxUpdated = 0;

gtsam::NonlinearFactorGraph gtSAMgraph;
bool gtSAMgraphMade = false;
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;

noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
noiseModel::Base::shared_ptr robustGPSNoise;

pcl::VoxelGrid<PointType> downSizeFilterScancontext;
SCManager scManager;
double scDistThres, scMaximumRadius;

pcl::VoxelGrid<PointType> downSizeFilterICP;
std::mutex mtxICP;
std::mutex mtxPosegraph;
std::mutex mtxRecentPose;

pcl::PointCloud<PointType>::Ptr laserCloudMapPGO(new pcl::PointCloud<PointType>());
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;
double mapVizFilterSize;
bool laserCloudMapPGORedraw = true;    

nav_msgs::Odometry::ConstPtr currGPS, lastGPS;
bool hasGPSforThisKF = false;
bool gpsOffsetInitialized = false; 

double gpsAltitudeInitOffset = 0.0;
double recentOptimizedX = 0.0;
double recentOptimizedY = 0.0;
double recentOptimizedZ = 0.0;
bool useGPS;
bool useGPSElevation;
double gpsDistThr;
double gpsCovThr;
double gpsScale;
double gpsScaleZ;
double filterSC;
u_int8_t thrInitGps = 20;
bool gpsInit = false;
bool firstGPS = true;
GeographicLib::LocalCartesian geoConverter;

//ICP params (loop closing)
std::string loopClosureMethod;
int numHistKeyframesIcpOld, numHistKeyframesIcpCurr;
double loopIcpFitnessScoreThreshold;
double voxelizationLeafSizeICP;
double loopNoiseScore;
double loopClosureNoiseScale;
double ICPMaxCorrespondenceDistance;
int ICPRANSACIterations;
bool ICPSavePointclouds;
bool ICPPublishPointclouds;

bool useScanControlLoopClosure;
bool useNearKFLoopClosure;
int nearKFProcessedId = 0;
int lcMinStepSeperation;
double lcDistanceThreshold;
double lcDistanceThreshold2;
double consecutiveLCMinDistance;
double consecutiveLCMinDistance2;

bool triggerExtraGraphOptimization = false;

std::string gpsTopic;

ros::Publisher pubMapAftPGO, pubOdomAftPGO, pubPathAftPGO;   
ros::Publisher pubLoopScanLocal, pubLoopSubmapLocal;
ros::Publisher pubOdomRepubVerifier;
ros::Publisher pubMarker;

std::string save_directory;
std::string pgKITTIformat, pgScansDirectory;
std::string odomKITTIformat;
std::fstream pgTimeSaveStream;

std::string padZeros(int val, int num_digits = 6) {
  std::ostringstream out;
  out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
  return out.str();
}

// Function to compute Euclidean distance (squared) between two poses
double euclideanDistance2(const Pose6D& p1, const Pose6D& p2) {
    return  (p1.x - p2.x) * (p1.x - p2.x) +
            (p1.y - p2.y) * (p1.y - p2.y) +
            (p1.z - p2.z) * (p1.z - p2.z);
}

// Function to compute Euclidean distance between two poses
double euclideanDistance(const Pose6D& p1, const Pose6D& p2) {
    return std::sqrt(euclideanDistance2(p1, p2));
}

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6DtoGTSAMPose3

void saveOdometryVerticesKITTIformat(std::string _filename)
{
    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);
    for(const auto& _pose6d: keyframePoses) {
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);
        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

void saveOptimizedVerticesKITTIformat(gtsam::Values _estimates, std::string _filename)
{
    using namespace gtsam;

    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);

    for(const auto& key_value: _estimates) {
        auto p = dynamic_cast<const GenericValue<Pose3>*>(&key_value.value);
        if (!p) continue;

        const Pose3& pose = p->value();

        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

// SL (Q): How do I use this?
nav_msgs::Odometry::ConstPtr navSatFixToUTMOdometry(const sensor_msgs::NavSatFix::ConstPtr& nav_sat_fix)
{
    // For UTM
    int zone;
    bool northp;
    double x_utm, y_utm;
    double gamma;
    double scale;

    // Convert from WGS84 lat/lon to UTM using GeographicLib’s UTM/UPS converter
    GeographicLib::UTMUPS::Forward(nav_sat_fix->latitude, nav_sat_fix->longitude, zone, northp, x_utm, y_utm, gamma, scale);

    // Create an Odometry message
    nav_msgs::Odometry odom;
    odom.header.frame_id = "map";   // Set the frame ID
    odom.header.stamp = nav_sat_fix->header.stamp; // Use the timestamp from NavSatFix

    // Set the position in the Odometry message
    odom.pose.pose.position.x = x_utm;
    odom.pose.pose.position.y = y_utm;
    odom.pose.pose.position.z = nav_sat_fix->altitude;  // Use altitude for Z in the UTM frame

    // Set an identity orientation if orientation information is not available
    odom.pose.pose.orientation.w = 1.0;

    // Copy covariances from NavSatFix
    // Assuming the covariances in NavSatFix are already in the correct units
    odom.pose.covariance[0] = nav_sat_fix->position_covariance[0];  // Variance in X (easting)
    odom.pose.covariance[7] = nav_sat_fix->position_covariance[4];  // Variance in Y (northing)
    odom.pose.covariance[14] = nav_sat_fix->position_covariance[8]; // Variance in Z (altitude)

    // Set all other covariances to zero (you can modify this if you have specific correlations or uncertainties)
    for (size_t i = 0; i < odom.pose.covariance.size(); ++i) {
        if (i != 0 && i != 7 && i != 14) {
            odom.pose.covariance[i] = 0.0;
        }
    }

    // Create a shared pointer to the Odometry object
    nav_msgs::Odometry::Ptr odom_ptr = boost::make_shared<nav_msgs::Odometry>(odom);

    // Return a ConstPtr from that
    return nav_msgs::Odometry::ConstPtr(odom_ptr);
}

// SL (Q): What if this "first" navsatfix has error? Is the whole map shifted, or is that fixed afterwards by the graph somehow?
nav_msgs::Odometry::ConstPtr navSatFixToLCOdometry(const sensor_msgs::NavSatFix::ConstPtr& nav_sat_fix)
{
    if (firstGPS)
    {
        geoConverter.Reset(nav_sat_fix->latitude, nav_sat_fix->longitude, nav_sat_fix->altitude);
        firstGPS = false;
    }

    // Create an Odometry message
    nav_msgs::Odometry odom;   
    odom.header.frame_id = "map";   // Set the frame ID
    odom.header.stamp = nav_sat_fix->header.stamp; // Use the timestamp from NavSatFix

    // Set the position in the Odometry message
    geoConverter.Forward(nav_sat_fix->latitude, nav_sat_fix->longitude, nav_sat_fix->altitude, odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z);

    // Set an identity orientation if orientation information is not available
    odom.pose.pose.orientation.w = 1.0;

    // Copy covariances from NavSatFix
    // Assuming the covariances in NavSatFix are already in the correct units
    odom.pose.covariance[0] = nav_sat_fix->position_covariance[0];  // Variance in X (easting)
    odom.pose.covariance[7] = nav_sat_fix->position_covariance[4];  // Variance in Y (northing)
    odom.pose.covariance[14] = nav_sat_fix->position_covariance[8]; // Variance in Z (altitude)

    // Set all other covariances to zero (you can modify this if you have specific correlations or uncertainties)
    for (size_t i = 0; i < odom.pose.covariance.size(); ++i) {
        if (i != 0 && i != 7 && i != 14) {
            odom.pose.covariance[i] = 0.0;
        }
    }

    // Create a shared pointer to the Odometry object
    nav_msgs::Odometry::Ptr odom_ptr = boost::make_shared<nav_msgs::Odometry>(odom);

    //geoConverter.Reverse ... for acquiring LLA

    // Return a ConstPtr from that
    return nav_msgs::Odometry::ConstPtr(odom_ptr);
}

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(_laserOdometry);
	mBuf.unlock();
} // laserOdometryHandler

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &_laserCloudFullRes)
{
	mBuf.lock();
	fullResBuf.push(_laserCloudFullRes);
	mBuf.unlock();
} // laserCloudFullResHandler

void gpsHandler(const sensor_msgs::NavSatFix::ConstPtr &_gps)
{
    if(useGPS) {
        mBufGPS.lock();
        gpsBuf.push(_gps);  
        mBufGPS.unlock();
    }
} // gpsHandler

void initNoises( void )
{
    priorNoise =
    noiseModel::Diagonal::Variances(
            (Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8)
                    .finished());  // rad*rad, meter*meter

    gtsam::Vector odomNoiseVector6(6);
    // odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );
} // initNoises

// SL: Why using roll pitch yaw instead of quaternion?
Pose6D getOdom(nav_msgs::Odometry::ConstPtr _odom)
{
    auto tx = _odom->pose.pose.position.x;
    auto ty = _odom->pose.pose.position.y;
    auto tz = _odom->pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw}; 
} // getOdom

Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2)
{
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
    Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
    Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
    Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
    // std::cout << "delta : " << dx << ", " << dy << ", " << dz << ", " << droll << ", " << dpitch << ", " << dyaw << std::endl;

    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
} // SE3Diff

pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);
    
    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}


void publish_lc_markers()
{
    visualization_msgs::MarkerArray markerArray;

    for (size_t i = 0; i < loopClosureIdsAdded.size(); ++i){
        // Assume pubMarker is a ros::Publisher advertising on, e.g., "/loop_closure_marker"
        visualization_msgs::Marker marker;
        marker.header.frame_id = "camera_init"; // or the frame that your keyframe positions are in
        marker.header.stamp = ros::Time::now();
        marker.ns = "loop_closures";
        marker.id = i; // You can assign different IDs if you want to publish multiple markers
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;

        // Set the line width
        marker.scale.x = 0.3; // Adjust this value for the desired line thickness

        // Set the color (red in this example)
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        geometry_msgs::Point pointA, pointB;
        // Fill in the coordinates of the first keyframe (e.g., from keyframePosesUpdated[prev_idx])
        pointA.x = keyframePosesUpdated[loopClosureIdsAdded[i].first].x;
        pointA.y = keyframePosesUpdated[loopClosureIdsAdded[i].first].y;
        pointA.z = keyframePosesUpdated[loopClosureIdsAdded[i].first].z;

        // Fill in the coordinates of the current keyframe (e.g., from keyframePosesUpdated[curr_idx])
        pointB.x = keyframePosesUpdated[loopClosureIdsAdded[i].second].x;
        pointB.y = keyframePosesUpdated[loopClosureIdsAdded[i].second].y;
        pointB.z = keyframePosesUpdated[loopClosureIdsAdded[i].second].z;

        // For LINE_LIST, add each pair of points consecutively
        marker.points.push_back(pointA);
        marker.points.push_back(pointB);
        
        markerArray.markers.push_back(marker);
    }
    // Publish it
    pubMarker.publish(markerArray);
} // publish_lc_markers()

void pubPath( void )
{
    // pub odom and path 
    nav_msgs::Odometry odomAftPGO;
    nav_msgs::Path pathAftPGO;
    pathAftPGO.header.frame_id = "camera_init";
    mKF.lock(); 
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()) - 1; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)

    // SL: Loop through all elements of keyframePosesUpdated, transform them into a 
    // a list of geometry_msgs::PoseStamped, and store into a an array in a nav::messages/Path
    // message
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    {
        const Pose6D& pose_est = keyframePosesUpdated.at(node_idx); // updated poses
        // const gtsam::Pose3& pose_est = isamCurrentEstimate.at<gtsam::Pose3>(node_idx);

        nav_msgs::Odometry odomAftPGOthis;
        odomAftPGOthis.header.frame_id = "camera_init";
        odomAftPGOthis.child_frame_id = "/aft_pgo";
        odomAftPGOthis.header.stamp = ros::Time().fromSec(keyframeTimes.at(node_idx));
        odomAftPGOthis.pose.pose.position.x = pose_est.x;
        odomAftPGOthis.pose.pose.position.y = pose_est.y;
        odomAftPGOthis.pose.pose.position.z = pose_est.z;
        odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);
        odomAftPGO = odomAftPGOthis;

        geometry_msgs::PoseStamped poseStampAftPGO;
        poseStampAftPGO.header = odomAftPGOthis.header;
        poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

        pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
        pathAftPGO.header.frame_id = "camera_init";
        pathAftPGO.poses.push_back(poseStampAftPGO);
    }
    mKF.unlock(); 

    // SL: Publish the most recent odometry pose
    pubOdomAftPGO.publish(odomAftPGO); // last pose
    // SL: Publish the last pose 
    pubPathAftPGO.publish(pathAftPGO); // poses 

    // SL: Updates the transform
    // Only send a transform if the timestamp has changed.
    static tf::TransformBroadcaster br;
    static ros::Time last_tf_time;
    if (odomAftPGO.header.stamp != last_tf_time)
    {
        last_tf_time = odomAftPGO.header.stamp;
        
        tf::Transform transform;
        tf::Quaternion q;
        transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
        q.setW(odomAftPGO.pose.pose.orientation.w);
        q.setX(odomAftPGO.pose.pose.orientation.x);
        q.setY(odomAftPGO.pose.pose.orientation.y);
        q.setZ(odomAftPGO.pose.pose.orientation.z);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp, "camera_init", "/aft_pgo"));
    }

    publish_lc_markers();
} // pubPath

// Updates the keyFramesPoses with the isam estimate
void updatePoses(void)
{
    mKF.lock(); 
    for (int node_idx=0; node_idx < int(isamCurrentEstimate.size()); node_idx++)
    {
        Pose6D& p =keyframePosesUpdated[node_idx];
        p.x = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().x();
        p.y = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().y();
        p.z = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().z();
        p.roll = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().roll();
        p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
        p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().yaw();
    }
    mKF.unlock();

    mtxRecentPose.lock();
    const gtsam::Pose3& lastOptimizedPose = isamCurrentEstimate.at<gtsam::Pose3>(int(isamCurrentEstimate.size())-1);
    recentOptimizedX = lastOptimizedPose.translation().x();
    recentOptimizedY = lastOptimizedPose.translation().y();
    recentOptimizedZ = lastOptimizedPose.translation().z();

    recentIdxUpdated = int(keyframePosesUpdated.size()) - 1;

    mtxRecentPose.unlock();
} // updatePoses

void runISAM2opt(void)
{
    // Add all new graph factors to the ISAM optimizer
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    if (triggerExtraGraphOptimization == true) {
        for(int i=0; i<5; i++) isam->update();
    }
    triggerExtraGraphOptimization = false;

    // SL: Clears the graph since all the factors have already been added to ISAM optimizer
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    updatePoses();
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, gtsam::Pose3 transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType *pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(
                                    transformIn.translation().x(), transformIn.translation().y(), transformIn.translation().z(), 
                                    transformIn.rotation().roll(), transformIn.rotation().pitch(), transformIn.rotation().yaw() );
    
    int numberOfCores = 8; // TODO move to yaml 
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        pointFrom = &cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom->intensity;
    }
    return cloudOut;
} // transformPointCloud

void loopFindNearKeyframesCloud( pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& submap_size)
{
    // extract and stacking near keyframes (in global coord)
    nearKeyframes->clear();   
    for (int i = -submap_size; i <= submap_size; ++i) {
        int keyNear = key + i;
        // if (keyNear < 0 || keyNear >= int(keyframeLaserClouds.size()) )
        //     continue;
        if (keyNear < 0 || keyNear >= recentIdxUpdated )
            continue;

        mKF.lock(); 
        *nearKeyframes += * local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[keyNear]);
        mKF.unlock(); 
    }

    if (nearKeyframes->empty())
        return;

    // downsample near keyframes. Don't do this for small_gicp, since it can do
    // this more efficiently within the algorithm itself.
    if(loopClosureMethod != "small_gicp" && loopClosureMethod != "small_vgicp"){
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());  
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }
} // loopFindNearKeyframesCloud


// Helper function
std::shared_ptr<small_gicp::PointCloud> pclToEigen(const pcl::PointCloud<PointType>::Ptr& pcl_cloud) {
    std::vector<Eigen::Vector3d> points;
    points.reserve(pcl_cloud->points.size());  // Reserve space for efficiency

    for (const auto& point : pcl_cloud->points) {
        points.emplace_back(point.x, point.y, point.z);
    }

    return std::make_shared<small_gicp::PointCloud>(points);
}
pcl::PointCloud<pcl::PointXYZ>::Ptr eigenToPcl(const std::shared_ptr<small_gicp::PointCloud> eigen_cloud) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl_cloud->resize(eigen_cloud->size());
    for (size_t i = 0; i<eigen_cloud->size(); ++i)
    {
        pcl_cloud->points[i].x = eigen_cloud->points[i][0];
        pcl_cloud->points[i].y = eigen_cloud->points[i][1];
        pcl_cloud->points[i].z = eigen_cloud->points[i][2];
    }
    return pcl_cloud;
}

// SL: Compute the relative pose between two key frame indicies 
// SL: Modified to return the noise model as well   
std::optional<std::pair<gtsam::Pose3, gtsam::noiseModel::Base::shared_ptr>> doICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx )
{
    // SL: Define point clouds
    pcl::PointCloud<PointType>::Ptr currKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());     

    // Assemble point clouds from frames around potential key frames indicies
    loopFindNearKeyframesCloud(currKeyframeCloud, _curr_kf_idx, numHistKeyframesIcpCurr); // use same root of loop kf idx 
    loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, numHistKeyframesIcpOld);   

    // loop verification
    if(ICPPublishPointclouds){
        sensor_msgs::PointCloud2 currKeyframeCloudMsg;   
        pcl::toROSMsg(*currKeyframeCloud, currKeyframeCloudMsg);
        currKeyframeCloudMsg.header.frame_id = "camera_init";     
        pubLoopScanLocal.publish(currKeyframeCloudMsg);
    
        sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
        pcl::toROSMsg(*targetKeyframeCloud, targetKeyframeCloudMsg);
        targetKeyframeCloudMsg.header.frame_id = "camera_init";
        pubLoopSubmapLocal.publish(targetKeyframeCloudMsg);
    }

    // Save the two point clouds to PCD files
    // Save the two point clouds using filenames that reflect their keyframe indices
    if(ICPSavePointclouds){
        std::stringstream curr_filename, loop_filename;
        curr_filename << save_directory << "loop_closure_pcd/" << "current_kf_" << _curr_kf_idx << "-" << _loop_kf_idx << ".pcd";
        loop_filename << save_directory << "loop_closure_pcd/" << "loop_kf_" << _curr_kf_idx << "-" << _loop_kf_idx << ".pcd";
    
        pcl::io::savePCDFileASCII(curr_filename.str(), *currKeyframeCloud);
        pcl::io::savePCDFileASCII(loop_filename.str(), *targetKeyframeCloud);
    }

    bool isConverged;
    float fitnessScore;
    Eigen::Isometry3d correctionTransform;
    
    // Get the current time before ICP computation
    ros::Time icp_start_time = ros::Time::now();
    pcl::PointCloud<PointType>::Ptr reg_result(new pcl::PointCloud<PointType>());

    
    if(loopClosureMethod == "gicp"){

        pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
        gicp.setMaxCorrespondenceDistance(ICPMaxCorrespondenceDistance);   // smaller threshold for fine
        gicp.setMaximumIterations(40);
        gicp.setTransformationEpsilon(1e-7);
        gicp.setEuclideanFitnessEpsilon(1e-7);
        gicp.setRANSACIterations(ICPRANSACIterations); // SL (Q): Why no ransac? Could help!
    
        gicp.setInputSource(currKeyframeCloud);
        gicp.setInputTarget(targetKeyframeCloud);
    
        gicp.align(*reg_result);

        isConverged = gicp.hasConverged();
        fitnessScore = gicp.getFitnessScore();
        correctionTransform = gicp.getFinalTransformation().cast<double>();

    }else if(loopClosureMethod == "icp"){

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(ICPMaxCorrespondenceDistance);   // smaller threshold for fine
        icp.setMaximumIterations(40);
        icp.setTransformationEpsilon(1e-7);
        icp.setEuclideanFitnessEpsilon(1e-7);
        icp.setRANSACIterations(ICPRANSACIterations); // SL (Q): Why no ransac? Could help!
    
        icp.setInputSource(currKeyframeCloud);
        icp.setInputTarget(targetKeyframeCloud);
    
        icp.align(*reg_result);

        isConverged = icp.hasConverged();
        fitnessScore = icp.getFitnessScore();
        correctionTransform = icp.getFinalTransformation().cast<double>();

    }else if(loopClosureMethod == "ndt"){

        pcl::NormalDistributionsTransform<PointType, PointType> ndt;
        ndt.setResolution(0.6); // Adjust resolution (in meters) as needed
        ndt.setMaxCorrespondenceDistance(ICPMaxCorrespondenceDistance);   // Maximum correspondence distance
        ndt.setMaximumIterations(40);
        ndt.setTransformationEpsilon(1e-7);
        ndt.setEuclideanFitnessEpsilon(1e-7);
    
        ndt.setInputSource(currKeyframeCloud);
        ndt.setInputTarget(targetKeyframeCloud);
    
        ndt.align(*reg_result);
        
        isConverged = ndt.hasConverged();
        fitnessScore = ndt.getFitnessScore();
        correctionTransform = ndt.getFinalTransformation().cast<double>();

    }else if(loopClosureMethod == "small_gicp" || loopClosureMethod == "small_vgicp"){
        
        auto source = pclToEigen(currKeyframeCloud);
        auto target = pclToEigen(targetKeyframeCloud); 
        small_gicp::RegistrationSetting settings;
        settings.num_threads = 6;                    // Number of threads to be used
        settings.max_correspondence_distance = ICPMaxCorrespondenceDistance;  // Maximum correspondence distance between points (e.g., triming threshold)
        settings.voxel_resolution = 1.0;
        settings.max_iterations = 20;
        settings.downsampling_resolution = voxelizationLeafSizeICP;
        if (loopClosureMethod == "small_gicp") settings.type = small_gicp::RegistrationSetting::RegistrationType::GICP;
        else if(loopClosureMethod == "small_vgicp") settings.type = small_gicp::RegistrationSetting::RegistrationType::VGICP;
        else ROS_ERROR("Invalid small_gicp method: %s", loopClosureMethod.c_str());

        Eigen::Isometry3d init_transform = Eigen::Isometry3d::Identity();
        small_gicp::RegistrationResult result = small_gicp::align(target->points, source->points, init_transform, settings);

        isConverged = result.converged;
        fitnessScore = result.error / result.num_inliers;
        correctionTransform = result.T_target_source;

    }else{
        ROS_ERROR("Invalid loopClosureMethod: %s. Expected \"icp\", \"gicp\", or \"ndt\". Skipping loop closure.", loopClosureMethod.c_str());
        return std::nullopt;  // Trigger error and exit the function early
    }

    if (!isConverged || fitnessScore > loopIcpFitnessScoreThreshold) {
        ROS_WARN("[Loop Closure] ICP fitness test failed (%.4g > %.4g). Not adding loop closure between %d and %d. ICP runtime: %.3g ms.", fitnessScore, loopIcpFitnessScoreThreshold, _loop_kf_idx, _curr_kf_idx, (ros::Time::now() - icp_start_time).toSec()*1e3);
        return std::nullopt;
    } else {
        ROS_INFO("[Loop Closure] ICP fitness test passed (%.4g < %.4g). Adding loop closure between %d and %d. ICP runtime: %.3g ms.", fitnessScore, loopIcpFitnessScoreThreshold, _loop_kf_idx, _curr_kf_idx, (ros::Time::now() - icp_start_time).toSec()*1e3);
    }

    // Get pose transformation (need to take inverse)
    gtsam::Pose3 betweenPose = Pose3(correctionTransform.matrix()).inverse();

    // Compute the noise model proportional to the fitness score squared
    gtsam::Vector robustNoiseVector6(6);
    robustNoiseVector6 << fitnessScore, fitnessScore, fitnessScore, fitnessScore, fitnessScore, fitnessScore;
    robustNoiseVector6 *= loopClosureNoiseScale;
    auto loopNoise = gtsam::noiseModel::Robust::Create(
                        gtsam::noiseModel::mEstimator::Cauchy::Create(1),
                        gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6)
                    );

    // Return both the relative pose and the noise model
    return std::make_pair(betweenPose, loopNoise);
} // doICPVirtualRelative

void process_pg()
{
    // SL: Start infinite loop
    while(1)
    {
        // SL: While odometry buffer and laser scan buffer (full res) are not empty.
		while ( !odometryBuf.empty() && !fullResBuf.empty() )
        {
            //
            // pop and check keyframe is or not  
            // 
            // SL: Discard odometry measurements older than the first laser scan
			mBuf.lock();       
            while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
                odometryBuf.pop();
            if (odometryBuf.empty())
            {
                // If the odometry buffer is now empty, break out of loop and wait more
                mBuf.unlock();
                break;
            }

            // Time equal check
            timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
            timeLaser = fullResBuf.front()->header.stamp.toSec();
            // TODO
            // SL (Q): What is to do? what is being checked here?

            // SL: Convert the full res laser scan queue item into a point cloud, store it in thisKeyFrame
            laserCloudFullRes->clear(); // SL (Q): Why are we clearing the full laser cloud? I don't think this variable is actually even used besides this one line.
            pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*fullResBuf.front(), *thisKeyFrame);
            fullResBuf.pop();

            // SL: Convert odometry queue item corresponding to a position and RPY, store in pose_curr
            Pose6D pose_curr = getOdom(odometryBuf.front());
            odometryBuf.pop();

            // SL: Look for a GPS message corresponding to key frame (as determined be eps)
            // SL: Discard all other prior GPS points
            // SL: Store 
            // SL (Q): eps should be a ROS parameter
            double eps = 0.1; // find a gps topic arrived within eps second 
            mBufGPS.lock();
            while (!gpsBuf.empty()) {
                auto thisGPS = gpsBuf.front();
                auto thisGPSTime = thisGPS->header.stamp.toSec();
                if( abs(thisGPSTime - timeLaserOdometry) < eps ) {
                    currGPS = navSatFixToLCOdometry(thisGPS);
                    hasGPSforThisKF = true; 
                    break;
                } else {
                    hasGPSforThisKF = false;
                }
                gpsBuf.pop();
            }
            mBufGPS.unlock(); 
            mBuf.unlock();

            //
            // Early reject by counting local delta movement (for equi-spreated kf drop)
            // SL: Decide if should assign this measurement as a key frame, by checking
            // how much distance has been travelled since the previous keyframe.
            // SL (Q): If it's not a key frame, is the laser scan discarded? Shouldn't it be merged, somehow?
            // SL (Q): What is the downside of just having more keyframes?
            // 
            odom_pose_prev = odom_pose_curr;
            odom_pose_curr = pose_curr;
            // SL (Q): Seems like the logic to calculate the distance between poses should be in the diffTransformation function
            // Then you could have a dedicated function that returns cartesian distance and angular distance
            Pose6D dtf = diffTransformation(odom_pose_prev, odom_pose_curr); // dtf means delta_transform

            double delta_translation = sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z); // note: absolute value. 
            translationAccumulated += delta_translation;
            rotationAccumulated += (dtf.roll + dtf.pitch + dtf.yaw); // sum just naive approach.  

            // Break out of loop if distance travelled is insufficient. Otherwise, reset accumulated travel.
            if( translationAccumulated < keyframeMeterGap && rotationAccumulated < keyframeRadGap ) continue;
            
            translationAccumulated = 0.0; // reset 
            rotationAccumulated = 0.0; // reset 
            
            // Save data and Add consecutive node 
            //
            // SL: Downsample input point cloud from thisKeyFrame according to voxel
            pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
            downSizeFilterScancontext.setInputCloud(thisKeyFrame);
            downSizeFilterScancontext.filter(*thisKeyFrameDS);

            mKF.lock(); 
            // SL: Store downsampled data to vectors of laser clouds, poses, and timestamps
            keyframeLaserClouds.push_back(thisKeyFrameDS);
            keyframePoses.push_back(pose_curr);
            keyframePosesUpdated.push_back(pose_curr); // init
            keyframeTimes.push_back(timeLaserOdometry);

            // SL (Q): What does this line do?
            scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);

            // SL (Q): laserCloudMapPGORedraw doesn't seem to be referenced anywhere else in code. If this line
            // doesn't do anything, just remove.
            laserCloudMapPGORedraw = true;
            mKF.unlock(); 

            // SL: Get indicies of keyframes (current and previous)
            const int prev_node_idx = keyframePoses.size() - 2; 
            const int curr_node_idx = keyframePoses.size() - 1; // becuase cpp starts with 0 (actually this index could be any number, but for simple implementation, we follow sequential indexing)

            if( ! gtSAMgraphMade /* prior node */) {
                // SL: If graph is not made, initialize it with the origin pose
                const int init_node_idx = 0; 
                gtsam::Pose3 poseOrigin = Pose6DtoGTSAMPose3(keyframePoses.at(init_node_idx));
                // auto poseOrigin = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

                mtxPosegraph.lock();  
                {
                    // prior factor 
                    gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(init_node_idx, poseOrigin, priorNoise));
                    initialEstimate.insert(init_node_idx, poseOrigin);
                    // runISAM2opt();          
                }   
                mtxPosegraph.unlock();
                gtSAMgraphMade = true; 
            } 
            // SL (Q): Seems like this should just be if, not else if?
            else if (keyframePoses.size() > 1 ) { // == keyframePoses.size() > 1 
                // SL: If keyframe sizes are large enough, add between factors based on LIO
                gtsam::Pose3 poseFrom = Pose6DtoGTSAMPose3(keyframePoses.at(prev_node_idx));
                gtsam::Pose3 poseTo = Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx));

                mtxPosegraph.lock();
                {
                    // odom factor
                    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, poseFrom.between(poseTo), odomNoise));
    
                    // gps factor 

                    // Check the covariances of GPS (x,y)
                    bool addGps = false;

                    if(currGPS != nullptr)
                        addGps = (currGPS->pose.covariance[0] < gpsCovThr && currGPS->pose.covariance[7] < gpsCovThr) ? 1 : 0;
                    
                    if(hasGPSforThisKF && addGps) {

                        // double curr_altitude_offseted = currGPS->altitude - gpsAltitudeInitOffset;
                        //Just for the first frame -> Initialize
                        if(lastGPS == nullptr) 
                            lastGPS = currGPS;
                        
                        //Add gps correction after few meters. TODO: Create a function for distance
                        if(sqrt(( currGPS->pose.pose.position.x - lastGPS->pose.pose.position.x ) * ( currGPS->pose.pose.position.x - lastGPS->pose.pose.position.x ) 
                            + ( currGPS->pose.pose.position.y - lastGPS->pose.pose.position.y ) * ( currGPS->pose.pose.position.y - lastGPS->pose.pose.position.y )
                            + ( currGPS->pose.pose.position.z - lastGPS->pose.pose.position.z ) * ( currGPS->pose.pose.position.z - lastGPS->pose.pose.position.z )) > gpsDistThr)
                        {
                            float gps_noise_z;
                            // gps_counter++;
                            gtsam::Point3 gpsConstraint;
                            // SL: Explain what happens if altitude not used?
                            if(useGPSElevation) {
                                gpsConstraint << currGPS->pose.pose.position.x, currGPS->pose.pose.position.y, currGPS->pose.pose.position.z;
                                gps_noise_z = currGPS->pose.covariance[14] * gpsScale * gpsScaleZ;
                            }
                            else 
                            {
                                mtxRecentPose.lock(); //Protect recentOptimizedZ
                                gpsConstraint << currGPS->pose.pose.position.x, currGPS->pose.pose.position.y, recentOptimizedZ;
                                mtxRecentPose.unlock();
                                gps_noise_z = 1e8;
                                // SL (Q): GPS NOISE Z should be infinite if we don't know it
                                // SL (Q): How do you know for sure that recentOptimizedZ has been populated, and isn't the default value (0)? 
                                // SL (Q): Maybe safer to just set z to 0, then set gps_noise_z = 1e10 or something.
                            }
                            gtsam::Vector gps_noise(3);
                            
                            // Build gps noise model and factor
                            gps_noise << currGPS->pose.covariance[0] * gpsScale, currGPS->pose.covariance[7] * gpsScale, gps_noise_z;
                            noiseModel::Diagonal::shared_ptr gps_noise_Model =
                                    noiseModel::Diagonal::Variances(gps_noise);
                            gtsam::GPSFactor curr_gps_factor = gtsam::GPSFactor(curr_node_idx, gpsConstraint, gps_noise_Model);
                           
                            // SL: Don't add gps measurements until we've reached at least thrInitGPS valid GPS measurements (default 20)
                            // If we're below the threshold, just make the factor and add it to the keyframeGpsFactor vector
                            if  (keyframeGpsFactor.size() < thrInitGps)
                            {
                                // If we get reliable origin point, we adjust the weight of this gps factor to fix the map origin 
                                if(keyframeGpsFactor.size() == 0) 
                                {
                                    // SL (Q): why is gps_noise being redefined? Why are we suddenly scaled by 1e-6?
                                    // If this is the correct logic, why not:
                                    // gps_noise *= 1e-6;
                                    gps_noise << currGPS->pose.covariance[0] * 1e-6, currGPS->pose.covariance[7] * 1e-6, gps_noise_z * 1e-6;
                                    gps_noise_Model =
                                    noiseModel::Diagonal::Variances(gps_noise);
                                    curr_gps_factor = gtsam::GPSFactor(curr_node_idx, gpsConstraint, gps_noise_Model);
                                }
                                keyframeGpsFactor.push_back(curr_gps_factor);
                                ROS_DEBUG("Accumulated gps factor: %ld", keyframeGpsFactor.size());
                            }
                            // Otherwise, if there are enough GPS points, but the GPS measurements have not been initialized, go
                            // through all current keyframe's and add them to the graph
                            else if(!gpsInit)
                            {
                                ROS_INFO("Initialize GNSS transform!");
                                for (size_t i = 0; i < keyframeGpsFactor.size(); ++i) {
                                    gtsam::GPSFactor gpsFactor = keyframeGpsFactor.at(i);
                                    gtSAMgraph.add(gpsFactor);
                                }
                                gpsInit = true;
                                triggerExtraGraphOptimization = true;
                            }
                            // Otherwise, just add the current factor
                            else
                            {
                                gtSAMgraph.add(curr_gps_factor);
                                ROS_DEBUG("GPS factor added at node: %d", curr_node_idx);
                                triggerExtraGraphOptimization = true;
                             }
                            lastGPS = currGPS;
                        } // End GPS measurement valid block (travelled enough distance)
                    } // End has GPS valid block 

                    // SL (Q): Suggested refactor for block above (makes code more readable and logic clearer).
                    // odometryDistance would check for nullptr's.
                    //
                    // if (hasGPSforThisKF &&
                    //     currGPS != nullptr &&
                    //     (currGPS->pose.covariance[0] < gpsCovThr && currGPS->pose.covariance[7] < gpsCovThr) &&
                    //     odometryDistance(lastGPS, currGPS) > gpsDistThr)
                    // { 
                    //     lastGPS = currGPS;
                    //
                    //     gtsam::Point3 gpsConstraint(currGPS->pose.pose.position.x, currGPS->pose.pose.position.y, currGPS->pose.pose.position.z);
                    //     gtsam::Vector gps_noise(3);
                    //     gps_noise << currGPS->pose.covariance[0] * gpsScale, currGPS->pose.covariance[7] * gpsScale, currGPS->pose.covariance[14] * gpsScale;
                    //     if(!useGPSElevation){
                    //         gps_noise(2) = 0.01; // OR 1e10
                    //         mtxRecentPose.lock(); //Protect recentOptimizedZ
                    //         gpsConstraint(2) = recentOptimizedZ; // or just 0?
                    //         mtxRecentPose.unlock();
                    //     }
                    //     if(keyframeGPSFactor.size() == 0) gps_noise *= 1e-6;
                    //     noiseModel::Diagonal::shared_ptr gps_noise_Model = noiseModel::Diagonal::Variances(gps_noise);
                    //     gtsam::GPSFactor curr_gps_factor = gtsam::GPSFactor(curr_node_idx, gpsConstraint, gps_noise_Model);
                    //
                    //     if(gpsInit){
                    //         gtSAMgraph.add(curr_gps_factor);
                    //         ROS_INFO("GPS factor added at node: %d", curr_node_idx);
                    //         triggerExtraGraphOptimization = true;
                    //     }else if (keyframeGpsFactor.size() < thrInitGPS){
                    //         keyframeGpsFactor.push_back(curr_gps_factor);
                    //         ROS_INFO("Accumulated gps factor: %d", keyframeGpsFactor.size());
                    //     }else{
                    //         ROS_INFO("Initialize GNSS transform!");
                    //         for (int i = 0; i < keyframeGpsFactor.size(); ++i) {
                    //             gtSAMgraph.add(keyframeGpsFactor.at(i));
                    //         }
                    //         gpsInit = true;
                    //         triggerExtraGraphOptimization = true;
                    //     }
                    // }

                    // Insert the LIO pose estimate as an initial estimate to the graph
                    initialEstimate.insert(curr_node_idx, poseTo);                
                    // runISAM2opt();
                }
                mtxPosegraph.unlock();

                // if(curr_node_idx % 100 == 0)
                    // ROS_INFO("Added posegraph odom node: %d", curr_node_idx);
            } // End if statement that there are more than 1 keyframe, and graph is valid

            // if want to print the current graph, use gtSAMgraph.print("\nFactor Graph:\n");

            // save utility 
            // SL: Saves the full resolution scan key frame to a file.
            std::string curr_node_idx_str = padZeros(curr_node_idx);
            pcl::io::savePCDFileBinary(pgScansDirectory + curr_node_idx_str + ".pcd", *thisKeyFrame); // scan 
            pgTimeSaveStream << timeLaser << std::endl; // path 
        } // End of while loop (while odometry buffer and full res scan buffer are both not empty)

        // ps. 
        // scan context detector is running in another thread (in constant Hz, e.g., 1 Hz)
        // pub path and point cloud in another thread

        // wait (must required for running the while loop)
        // SL (Q): Sleep duration should be a ROS parameter
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_pg

void submitLoopClosureCandidate(int id1, int id2)
{
    // Check if have already tested this pair
    for (const auto& p : loopClosureIdsTested){
        if ((p.first == id1 && p.second == id2) ||
            (p.first == id2 && p.second == id1)){
            ROS_DEBUG("Already detected loop between %d and %d", id1, id2);
            return;
        }
    }

    // Push to ICP buffer and add to tested list
    mBuf.lock();
    loopClosureCandidateBuf.emplace(id1, id2);
    mBuf.unlock();
    loopClosureIdsTested.emplace_back(id1, id2);
}

// SL: Use ScanContext to identify potential loop closure keyframe candidates, push them into the loopClosureCandidateBuf queue.
void performSCLoopClosure(void)
{
    if( int(keyframePoses.size()) < scManager.NUM_EXCLUDE_RECENT) // do not try too early 
        return;

    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
    int SCclosestHistoryFrameID = detectResult.first;

    // Exit if not found anything
    if( SCclosestHistoryFrameID == -1 ) return;

    // Otherwise, add to buffer
    submitLoopClosureCandidate(SCclosestHistoryFrameID, keyframePoses.size() - 1);

} // performSCLoopClosure

// Add keyframes based on nearby keyframes
int lastDetectedLC_id1 = -1;
void performNearKFClosure(){
    
    while(nearKFProcessedId < recentIdxUpdated-1){
        // Increment the frame we're checking
        nearKFProcessedId++;

        if(lastDetectedLC_id1 >= 0 &&
            euclideanDistance2(keyframePosesUpdated[nearKFProcessedId], keyframePosesUpdated[lastDetectedLC_id1]) < consecutiveLCMinDistance2){
             // We're still too close to a previously detected loop closure
             // Skip
             continue;
         }

        // We're not too close a loop closure (current). Now check all previous frames for nearby loop closures
        int lastDetectedLC_id2 = -1;
        for (int j = 0; j < recentIdxUpdated - lcMinStepSeperation; ++j) {
            // If we've already detected a nearby KF (past) for this current keyframe, check that the current frame isn't super close to that one.
            if(lastDetectedLC_id2 >= 0 &&
               euclideanDistance2(keyframePosesUpdated[j], keyframePosesUpdated[lastDetectedLC_id2]) < consecutiveLCMinDistance2){
                // We're still too close to the previous LC
                // Skip
                continue;
            }

            // Check if we're within tolerance
            double distance2 = euclideanDistance2(keyframePosesUpdated[j], keyframePosesUpdated[nearKFProcessedId]);
            if (distance2 < lcDistanceThreshold2) {
                submitLoopClosureCandidate(j, nearKFProcessedId);
                lastDetectedLC_id1 = nearKFProcessedId;
                lastDetectedLC_id2 = j;
                ROS_INFO("Added candidate loop closure pairs %d / %d, distance %.3g m < %.3g m", j, nearKFProcessedId, sqrt(distance2), lcDistanceThreshold);
            }
        }
    }
}

// SL: Identify keyframes that have potential loop closures at a given rate
void process_lcd(void)
{
    // SL (Q): Loop closure frequency should be a ROS param
    float loopClosureFrequency = 0.4; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        // SL (Q): I'm curious how well ScanContext does at identifying loop closures in the forest context. It's design for an urban environment. It matches the highest point in the point cloud in an area. In an urban environment, this works well because features have distinct upper roofs. But, in our context the top of the forest is rarely meausrements, so it may not be reliable. Instead we're just always surrounded by forests. Maybe better to just rely on the LIO position, and detect candidate loop closures as estimated close keyframes? Or use some other metric? E.g. just calculate the distance between each pair of key frames, then find pairs that are nearby as candidates.
        if(useScanControlLoopClosure) performSCLoopClosure();

        // Try LC based on naive KF distance
        if(useNearKFLoopClosure) performNearKFClosure();
    }
} // process_lcd

// Compute exact graph factors from identified loop closure frames
void process_icp(void)
{
    while(1)
    {
		while ( !loopClosureCandidateBuf.empty() )
        {
            if( loopClosureCandidateBuf.size() > 30 ) {
                ROS_WARN("%ld loop closure candidates waiting... Adjust settings to produce fewer loop closure candidates", loopClosureCandidateBuf.size());
            }

            mBuf.lock(); 
            std::pair<int, int> loop_idx_pair = loopClosureCandidateBuf.front();

            // SL: We can't process loop closures until the pose has been updated by the graph.
            // Skip any that are not updated.
            if(loop_idx_pair.first>=recentIdxUpdated || loop_idx_pair.second>=recentIdxUpdated){
                mBuf.unlock(); 
                continue;
            }else{
                loopClosureCandidateBuf.pop();
                mBuf.unlock();
            }

            // get BetweenFactor between poses (or std::nullopt if doesn't pass test)
            const int prev_node_idx = loop_idx_pair.first;
            const int curr_node_idx = loop_idx_pair.second;
            
            auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx);

            // SL: Add between factor to graph if valid
            if(relative_pose_optional) {
                auto [relative_pose, loopNoise] = relative_pose_optional.value();

                mtxPosegraph.lock();
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relative_pose, loopNoise));
                triggerExtraGraphOptimization = true;
                mtxPosegraph.unlock();

                // Add to list of added loop closures
                loopClosureIdsAdded.push_back(std::pair<int, int>(prev_node_idx, curr_node_idx));
            } 
        }

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_icp

void process_viz_path(void)
{
    // SL (Q): The visualization path rate should be a ROS parameter
    float hz = 10.0; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        if(recentIdxUpdated > 1) {
            pubPath();
        }
    }
}

void process_isam(void)
{
    float hz = 1; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        if( gtSAMgraphMade ) {
            mtxPosegraph.lock();
            runISAM2opt();
            // cout << "running isam2 optimization ..." << endl;
            mtxPosegraph.unlock();

            // Save poses and vertices
            saveOptimizedVerticesKITTIformat(isamCurrentEstimate, pgKITTIformat); // pose
            saveOdometryVerticesKITTIformat(odomKITTIformat); // pose
        }
    }
}

// SL: Publishes the global map, optimized from the pose graph
void pubMap(void)
{
    int SKIP_FRAMES = 2; // sparse map visulalization to save computations

    laserCloudMapPGO->clear();

    mKF.lock(); 
    // SL: Loop through all elements of keyframePosesUpdates, add the local cloud to the global map cloud
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx+=SKIP_FRAMES) {
        *laserCloudMapPGO += *local2global(keyframeLaserClouds[node_idx], keyframePosesUpdated[node_idx]);
    }
    mKF.unlock(); 

    // SL: Downsample cloud
    downSizeFilterMapPGO.setInputCloud(laserCloudMapPGO);
    downSizeFilterMapPGO.filter(*laserCloudMapPGO);

    // SL: Publish message
    sensor_msgs::PointCloud2 laserCloudMapPGOMsg;
    pcl::toROSMsg(*laserCloudMapPGO, laserCloudMapPGOMsg);
    laserCloudMapPGOMsg.header.frame_id = "camera_init";
    pubMapAftPGO.publish(laserCloudMapPGOMsg);
}

// SL: Publish visualization map at a given frequency
void process_viz_map(void)
{
    // SL (Q): vizmapFrequency should be a ROS parameter
    float vizmapFrequency = 0.1; // 0.1 means run onces every 10s
    ros::Rate rate(vizmapFrequency);
    while (ros::ok()) {
        rate.sleep();
        if(recentIdxUpdated > 1) {
            pubMap();
        }
    }
} // pointcloud_viz


int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserPGO");
	ros::NodeHandle nh;

    // SL (Q): For all the graph noise parameters, it makes more sense for the "human"
    // readable parameter (the one inputted as the parameter to ROS) be the standard
    // deviation, rather than the variance. So it is better to have the input variances be input
    // standard deviations instead, then square them before you give it to the pose graph.
    // Just makes it easier to process mentally (2x standard deviation = 2x weight, whereas 2x variance
    // only gives sqrt(2) more weight).)
	nh.param<std::string>("save_directory", save_directory, "/"); // pose assignment every k m move 
    pgKITTIformat = save_directory + "optimized_poses.txt";
    odomKITTIformat = save_directory + "odom_poses.txt";
    pgTimeSaveStream = std::fstream(save_directory + "times.txt", std::fstream::out); 
    pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);
    pgScansDirectory = save_directory + "Scans/";
    (void)system((std::string("exec rm -r ") + pgScansDirectory).c_str());
    (void)system((std::string("mkdir -p ") + pgScansDirectory).c_str());

	nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 2.0); // pose assignment every k m move 
	nh.param<double>("keyframe_deg_gap", keyframeDegGap, 10.0); // pose assignment every k deg rot 
    keyframeRadGap = deg2rad(keyframeDegGap);

	nh.param<double>("sc_dist_thres", scDistThres, 0.2);  
	nh.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor 
    nh.param<double>("sc_leaf_size_down", filterSC, 0.4); // Scan Context point cloud downsampling

    //GPS Params
    nh.param<std::string>("gps_topic", gpsTopic, "/mavros/global_position/global"); // Scaling factor to be multiplied with gps Covariance 
    nh.param<bool>("use_gps", useGPS, false); // Covariance threshold for gps measurement to be taken into account (before applying scaling factor)
    nh.param<bool>("use_gps_elevation", useGPSElevation, false); // Scaling factor to be multiplied with gps Covariance 
    nh.param<double>("gps_dist_thr", gpsDistThr, 5.0); // Distance between gps measurements to be implemented in the graph
    nh.param<double>("gps_cov_thr", gpsCovThr, 4.0); // Covariance threshold for gps measurement to be taken into account (before applying scaling factor)
    nh.param<double>("gps_scale", gpsScale, 1.0); // Scaling factor to be multiplied with gps Covariance 
    nh.param<double>("gps_scale_z", gpsScaleZ, 1e2); // Scaling factor to be multiplied with gps Covariance (on z axis only, in addition to the gps_scale)

    //ICP Loop Params
    nh.param<std::string>("loop_closure_method", loopClosureMethod, "small_gicp"); // icp, gicp, or ndt

    nh.param<int>("icp_num_keyframes_old", numHistKeyframesIcpOld, 10); // Number of keyframes point cloud to be included in the submap for icp alignment (old keyframe)
    nh.param<int>("icp_num_keyframes_curr", numHistKeyframesIcpCurr, 3); // Number of keyframes point cloud to be included in the submap for icp alignment (current keyframe)
    nh.param<double>("icp_loop_fitness_score_thr", loopIcpFitnessScoreThreshold, 0.3); // ICP's loop fitness score threshold to accept closing a loop (i.e. registration was successful)
    nh.param<double>("icp_leaf_size_down", voxelizationLeafSizeICP, 0.2); // ICP's downsampling   
    nh.param<double>("loop_noise_score", loopNoiseScore, 0.8); // Loop Noise variance
    nh.param<double>("loop_closure_noise_scale", loopClosureNoiseScale, 1); // Loop Noise variance
    nh.param<double>("icp_max_correspondence_distance", ICPMaxCorrespondenceDistance, 2); // Loop Noise variance
    nh.param<int>("icp_ransac_iterations", ICPRANSACIterations, 10); // Loop Noise variance
    nh.param<bool>("icp_save_pointclouds", ICPSavePointclouds, false); // Loop Noise variance
    nh.param<bool>("icp_publish_pointclouds", ICPPublishPointclouds, false); // Loop Noise variance

    nh.param<bool>("use_scan_control_loop_closure", useScanControlLoopClosure, true);
    nh.param<bool>("use_near_kf_loop_closure", useNearKFLoopClosure, true);
    nh.param<int>("lc_min_step_seperation", lcMinStepSeperation, 20);
    nh.param<double>("near_kf_threshold", lcDistanceThreshold, 10); // Loop Noise variance
    lcDistanceThreshold2 = lcDistanceThreshold*lcDistanceThreshold;
    nh.param<double>("consecutive_lc_min_distance", consecutiveLCMinDistance, 5); // Loop Noise variance
    consecutiveLCMinDistance2 = consecutiveLCMinDistance*consecutiveLCMinDistance;
    

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);   
    initNoises();
   
    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

    downSizeFilterScancontext.setLeafSize(filterSC, filterSC, filterSC);
    downSizeFilterICP.setLeafSize(voxelizationLeafSizeICP, voxelizationLeafSizeICP, voxelizationLeafSizeICP);

	nh.param<double>("mapviz_filter_size", mapVizFilterSize, 0.4); // pose assignment every k frames 
    downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);

    // SL (Q): Should rename this topic listener to something more generic
	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_registered_local", 100, laserCloudFullResHandler);
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, laserOdometryHandler);
	ros::Subscriber subGPS = nh.subscribe<sensor_msgs::NavSatFix>(gpsTopic, 100, gpsHandler);

	pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
	pubOdomRepubVerifier = nh.advertise<nav_msgs::Odometry>("/repub_odom", 100);
	pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
	pubMapAftPGO = nh.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map", 100);

	pubLoopScanLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local", 100);
	pubLoopSubmapLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_submap_local", 100);

    // Make loop closure marker publisher
    pubMarker = nh.advertise<visualization_msgs::MarkerArray>("loop_closure_markers", 10);


	std::thread posegraph_slam {process_pg}; // pose graph construction
	std::thread lc_detection {process_lcd}; // loop closure detection 
	std::thread icp_calculation {process_icp}; // loop constraint calculation via icp 
    // SL (Q): Should define ROS parameters to decide whether to run isam2 as you described below, or the other option.
    // I'm not 100% sure I understand what you mean by "uncomment this and comment all the above runisam2opt when node is added".
	std::thread isam_update {process_isam}; // if you want to call less isam2 run (for saving redundant computations and no real-time visulization is required), uncommment this and comment all the above runisam2opt when node is added. 

    // SL (Q): Visualization update frequency should be a ros parameter. And then we can add logic that if that
    // parameter is negative (e.g. -1), that means to not run it at all.
	std::thread viz_map {process_viz_map}; // visualization - map (low frequency because it is heavy)
	std::thread viz_path {process_viz_path}; // visualization - path (high frequency)

 	ros::spin();

	return 0;
}

