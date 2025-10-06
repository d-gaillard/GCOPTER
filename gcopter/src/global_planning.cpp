#include "misc/visualizer.hpp"
#include "gcopter/trajectory.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/voxel_map.hpp"
#include "gcopter/sfc_gen.hpp"

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

struct Config
{
    std::string mapTopic;
    std::string targetTopic;
    double dilateRadius;
    double voxelWidth;
    std::vector<double> mapBound;
    double timeoutRRT;
    double maxVelMag;
    double maxBdrMag;
    double maxTiltAngle;
    double minThrust;
    double maxThrust;
    double vehicleMass;
    double gravAcc;
    double horizDrag;
    double vertDrag;
    double parasDrag;
    double speedEps;
    double weightT;
    std::vector<double> chiVec;
    double smoothingEps;
    int integralIntervs;
    double relCostTol;

    Config(rclcpp::Node::SharedPtr node)
    {
        node->declare_parameter("MapTopic", mapTopic);
        node->declare_parameter("TargetTopic", targetTopic);
        node->declare_parameter("DilateRadius", dilateRadius);
        node->declare_parameter("VoxelWidth", voxelWidth);
        node->declare_parameter("MapBound", mapBound);
        node->declare_parameter("TimeoutRRT", timeoutRRT);
        node->declare_parameter("MaxVelMag", maxVelMag);
        node->declare_parameter("MaxBdrMag", maxBdrMag);
        node->declare_parameter("MaxTiltAngle", maxTiltAngle);
        node->declare_parameter("MinThrust", minThrust);
        node->declare_parameter("MaxThrust", maxThrust);
        node->declare_parameter("VehicleMass", vehicleMass);
        node->declare_parameter("GravAcc", gravAcc);
        node->declare_parameter("HorizDrag", horizDrag);
        node->declare_parameter("VertDrag", vertDrag);
        node->declare_parameter("ParasDrag", parasDrag);
        node->declare_parameter("SpeedEps", speedEps);
        node->declare_parameter("WeightT", weightT);
        node->declare_parameter("ChiVec", chiVec);
        node->declare_parameter("SmoothingEps", smoothingEps);
        node->declare_parameter("IntegralIntervs", integralIntervs);
        node->declare_parameter("RelCostTol", relCostTol);
        
        node->get_parameter("MapTopic", mapTopic);
        node->get_parameter("TargetTopic", targetTopic);
        node->get_parameter("DilateRadius", dilateRadius);
        node->get_parameter("VoxelWidth", voxelWidth);
        node->get_parameter("MapBound", mapBound);
        node->get_parameter("TimeoutRRT", timeoutRRT);
        node->get_parameter("MaxVelMag", maxVelMag);
        node->get_parameter("MaxBdrMag", maxBdrMag);
        node->get_parameter("MaxTiltAngle", maxTiltAngle);
        node->get_parameter("MinThrust", minThrust);
        node->get_parameter("MaxThrust", maxThrust);
        node->get_parameter("VehicleMass", vehicleMass);
        node->get_parameter("GravAcc", gravAcc);
        node->get_parameter("HorizDrag", horizDrag);
        node->get_parameter("VertDrag", vertDrag);
        node->get_parameter("ParasDrag", parasDrag);
        node->get_parameter("SpeedEps", speedEps);
        node->get_parameter("WeightT", weightT);
        node->get_parameter("ChiVec", chiVec);
        node->get_parameter("SmoothingEps", smoothingEps);
        node->get_parameter("IntegralIntervs", integralIntervs);
        node->get_parameter("RelCostTol", relCostTol);
    }
};

class GlobalPlanner : public rclcpp::Node
{
public:
    GlobalPlanner(const Config &conf);
    
private:
    void mapCallBack(const sensor_msgs::msg::PointCloud2 & msg);
    void targetCallBack(const geometry_msgs::msg::PoseStamped & msg);

    void plan();
    void process();

    Config config;

    bool mapInitialized;
    voxel_map::VoxelMap voxelMap;
    Visualizer visualizer;
    std::vector<Eigen::Vector3d> startGoal;
    
    Trajectory<5> traj;
    double trajStamp;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mapSub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr targetSub;

};


GlobalPlanner::GlobalPlanner(const Config &conf);
    : Node("global_planning_node")
    config(conf),
    mapInitialized(false),
    visualizer(*this) //TODO: fix the initializer
{
    const Eigen::Vector3i xyz((config.mapBound[1] - config.mapBound[0]) / config.voxelWidth,
        (config.mapBound[3] - config.mapBound[2]) / config.voxelWidth,
        (config.mapBound[5] - config.mapBound[4]) / config.voxelWidth);

    const Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2], config.mapBound[4]);

    voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

    mapSub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        config.mapTopic, 1, std::bind(&GlobalPlanner::mapCallBack, this, std::placeholders::_1));

    targetSub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        config.targetTopic, 1, std::bind(&GlobalPlanner::targetCallBack, this, std::placeholders::_1));

};

void GlobalPlanner::mapCallBack(const sensor_msgs::msg::PointCloud2 & msg)
{
  if (!mapInitialized)
  {
      size_t cur = 0;
      const size_t total = msg->data.size() / msg->point_step;
      float *fdata = (float *)(&msg->data[0]);
      for (size_t i = 0; i < total; i++)
      {
          cur = msg->point_step / sizeof(float) * i;

          if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
              std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
              std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2]))
          {
              continue;
          }
          voxelMap.setOccupied(Eigen::Vector3d(fdata[cur + 0],
                                              fdata[cur + 1],
                                              fdata[cur + 2]));
      }

      voxelMap.dilate(std::ceil(config.dilateRadius / voxelMap.getScale()));

      mapInitialized = true;
    }
};

void GlobalPlanner::targetCallBack(const geometry_msgs::msg::PoseStamped & msg)
{
    if (mapInitialized)
    {
        if (startGoal.size() >= 2)
        {
            startGoal.clear();
        }
        const double zGoal = config.mapBound[4] + config.dilateRadius +
            fabs(msg->pose.orientation.z) *
                (config.mapBound[5] - config.mapBound[4] - 2 * config.dilateRadius);
        const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
        if (voxelMap.query(goal) == 0)
        {
            visualizer.visualizeStartGoal(goal, 0.5, startGoal.size());
            startGoal.emplace_back(goal);
        }
        else
        {
            ROS_WARN("Infeasible Position Selected !!!\n");
        }

        plan();
    }
    return;
};

void GlobalPlanner::plan()
{
    if (startGoal.size() == 2)
    {
        std::vector<Eigen::Vector3d> route;
        sfc_gen::planPath<voxel_map::VoxelMap>(startGoal[0],
                                                startGoal[1],
                                                voxelMap.getOrigin(),
                                                voxelMap.getCorner(),
                                                &voxelMap, 0.01,
                                                route);
        std::vector<Eigen::MatrixX4d> hPolys;
        std::vector<Eigen::Vector3d> pc;
        voxelMap.getSurf(pc);

        sfc_gen::convexCover(route,
                                pc,
                                voxelMap.getOrigin(),
                                voxelMap.getCorner(),
                                7.0,
                                3.0,
                                hPolys);
        sfc_gen::shortCut(hPolys);

        if (route.size() > 1)
        {
            visualizer.visualizePolytope(hPolys);

            Eigen::Matrix3d iniState;
            Eigen::Matrix3d finState;
            iniState << route.front(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
            finState << route.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

            gcopter::GCOPTER_PolytopeSFC gcopter;

            // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
            // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
            // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
            //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
            // initialize some constraint parameters
            Eigen::VectorXd magnitudeBounds(5);
            Eigen::VectorXd penaltyWeights(5);
            Eigen::VectorXd physicalParams(6);
            magnitudeBounds(0) = config.maxVelMag;
            magnitudeBounds(1) = config.maxBdrMag;
            magnitudeBounds(2) = config.maxTiltAngle;
            magnitudeBounds(3) = config.minThrust;
            magnitudeBounds(4) = config.maxThrust;
            penaltyWeights(0) = (config.chiVec)[0];
            penaltyWeights(1) = (config.chiVec)[1];
            penaltyWeights(2) = (config.chiVec)[2];
            penaltyWeights(3) = (config.chiVec)[3];
            penaltyWeights(4) = (config.chiVec)[4];
            physicalParams(0) = config.vehicleMass;
            physicalParams(1) = config.gravAcc;
            physicalParams(2) = config.horizDrag;
            physicalParams(3) = config.vertDrag;
            physicalParams(4) = config.parasDrag;
            physicalParams(5) = config.speedEps;
            const int quadratureRes = config.integralIntervs;

            traj.clear();

            if (!gcopter.setup(config.weightT,
                                iniState, finState,
                                hPolys, INFINITY,
                                config.smoothingEps,
                                quadratureRes,
                                magnitudeBounds,
                                penaltyWeights,
                                physicalParams))
            {
                return;
            }

            if (std::isinf(gcopter.optimize(traj, config.relCostTol)))
            {
                return;
            }

            if (traj.getPieceNum() > 0)
            {
                trajStamp = ros::Time::now().toSec();
                visualizer.visualize(traj, route);
            }
        }
    }
};

void GlobalPlanner::process()
{
    Eigen::VectorXd physicalParams(6);
    physicalParams(0) = config.vehicleMass;
    physicalParams(1) = config.gravAcc;
    physicalParams(2) = config.horizDrag;
    physicalParams(3) = config.vertDrag;
    physicalParams(4) = config.parasDrag;
    physicalParams(5) = config.speedEps;

    flatness::FlatnessMap flatmap;
    flatmap.reset(physicalParams(0), physicalParams(1), physicalParams(2),
                    physicalParams(3), physicalParams(4), physicalParams(5));

    if (traj.getPieceNum() > 0)
    {
        const double delta = ros::Time::now().toSec() - trajStamp;
        if (delta > 0.0 && delta < traj.getTotalDuration())
        {
            double thr;
            Eigen::Vector4d quat;
            Eigen::Vector3d omg;

            flatmap.forward(traj.getVel(delta),
                            traj.getAcc(delta),
                            traj.getJer(delta),
                            0.0, 0.0,
                            thr, quat, omg);
            double speed = traj.getVel(delta).norm();
            double bodyratemag = omg.norm();
            double tiltangle = acos(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)));
            std_msgs::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
            speedMsg.data = speed;
            thrMsg.data = thr;
            tiltMsg.data = tiltangle;
            bdrMsg.data = bodyratemag;
            visualizer.speedPub.publish(speedMsg);
            visualizer.thrPub.publish(thrMsg);
            visualizer.tiltPub.publish(tiltMsg);
            visualizer.bdrPub.publish(bdrMsg);

            visualizer.visualizeSphere(traj.getPos(delta),
                                        config.dilateRadius);
        }
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    Config config;
    auto planner_node = std::make_shared<GlobalPlanner>(config);

    rclcpp::spin(planner_node);
    rclcpp::shutdown();
    return 0;
}