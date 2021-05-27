#include <boost/foreach.hpp>
#include <Eigen/Dense>
#include <chrono>

// ROS
#include <ros/ros.h>
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/CameraInfo.h"
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32MultiArray.h>

// PCL
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>


typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

// pcl::visualization::PCLVisualizer::Ptr simpleVis (void)
// {
//   // --------------------------------------------
//   // -----Open 3D viewer and add point cloud-----
//   // --------------------------------------------
//   pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

//   viewer->setBackgroundColor (255, 255, 255);
//   viewer->addCoordinateSystem (1.0);
//   viewer->initCameraParameters ();
 
//   viewer->setCameraPosition(0.0, 0.0, 0.0, 0.0, -1.0, 0.0);
//   viewer->removeCoordinateSystem();
//   return (viewer);
// }

// pcl::visualization::PCLVisualizer::Ptr viewer = simpleVis();


class segmentPC
{
    ros::Publisher pub;
    ros::Subscriber sub;
    PointCloud stored_pc;
    bool receive_msg = false;

    public:
    segmentPC(ros::NodeHandle *nh)
    {
        sub = nh->subscribe<PointCloud>("/points_masked", 1, &segmentPC::sub_callback, this);
        pub = nh->advertise<std_msgs::Float32MultiArray> ("/obb_array", 1);
        printf("Init completed!");

    }

    void sub_callback(const PointCloud::ConstPtr& msg)
    {
        // std::cout <<"received" << std::endl;
        pcl::copyPointCloud(*msg, stored_pc);
        receive_msg = true;
    }

    void loop()
    {
        if (!receive_msg)
        {
            std::cout<<"No data received"<<std::endl;
            return;
        }
        double downsample_voxel_size = 0.002;
    //     double distance_thresh = 1.0;
    //     double plane_inliers_thresh = 0.02;
        double euclidean_clustering_tol = 0.01;
        auto start = std::chrono::high_resolution_clock::now();

        // viewer->spinOnce (50);
        // viewer->removeAllPointClouds();
        // viewer->removeAllShapes();

        PointCloud::Ptr cur_cloud (new PointCloud);
        pcl::copyPointCloud(stored_pc, *cur_cloud);

        // downsample point cloud 
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud (cur_cloud);
        double leafsize = downsample_voxel_size;
        sor.setLeafSize (leafsize, leafsize, leafsize);
        sor.filter (*cur_cloud);   


        // std::cout << cur_cloud->points.size() << std::endl;
        // Creating the KdTree object for the search method of the extraction
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
        tree->setInputCloud (cur_cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
        ec.setClusterTolerance (euclidean_clustering_tol); // in m
        ec.setMinClusterSize (50);
        ec.setMaxClusterSize (25000);
        ec.setSearchMethod (tree);
        ec.setInputCloud (cur_cloud);
        
        ec.extract (cluster_indices);

        int counter = 0;
        // std::cout <<'y' << std:: endl;
        for(auto it: cluster_indices)
        {
            
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
            int len_it = 0;
            for (const auto& idx : it.indices)
            {
                len_it ++;
                cloud_cluster->push_back ((*cur_cloud)[idx]);
                // dst_cloud->points[idx].r = 0;
                // dst_cloud->points[idx].g = 255;
                // dst_cloud->points[idx].b = 0;

            }
            // std::cout << "cluster indices len: " << len_it << std::endl;
            // get bbox of point cloud
            pcl::PointXYZRGB minpoint;
            pcl::PointXYZRGB maxpoint;
            pcl::PointXYZRGB position;
            Eigen::Matrix3f rot_matrix;
            pcl::MomentOfInertiaEstimation<pcl::PointXYZRGB> bbox_maker;
            bbox_maker.setInputCloud(cloud_cluster);
            bbox_maker.compute();

            pcl::PointIndices::Ptr ind_ptr (new pcl::PointIndices);
            ind_ptr->indices = it.indices;
            ind_ptr->header = it.header;

            bbox_maker.getOBB(minpoint, maxpoint, position, rot_matrix);

            // create array to send OBB data
            std::vector<float> rot_m_vec(rot_matrix.size());
            Eigen::Map<Eigen::Matrix3f>(rot_m_vec.data(), rot_matrix.rows(), rot_matrix.cols()) = rot_matrix;

            std::vector<float> points_array;
            points_array.emplace_back(minpoint.x);
            points_array.emplace_back(minpoint.y);
            points_array.emplace_back(minpoint.z);

            points_array.emplace_back(maxpoint.x);
            points_array.emplace_back(maxpoint.y);
            points_array.emplace_back(maxpoint.z);

            points_array.emplace_back(position.x);
            points_array.emplace_back(position.y);
            points_array.emplace_back(position.z);

            std::vector<float> results (points_array);

            results.insert(results.end(), rot_m_vec.begin(), rot_m_vec.end());

            std_msgs::Float32MultiArray out_array;

            for (auto el: results)
            {
                out_array.data.push_back(el);
            }

            // pub.publish(out_array);

            Eigen::Vector3f major_vector, middle_vector, minor_vector;
	        Eigen::Vector3f mass_center;

            bbox_maker.getEigenVectors(major_vector, middle_vector, minor_vector);
	        bbox_maker.getMassCenter(mass_center);

            Eigen::Vector3f dims(maxpoint.x - minpoint.x, maxpoint.y - minpoint.y, maxpoint.z - minpoint.z);
            
            Eigen::VectorXf vec_joined(major_vector.size() + middle_vector.size() + mass_center.size() + dims.size());
            vec_joined << major_vector, middle_vector, mass_center, dims;
            
            std_msgs::Float32MultiArray out_array1;
            for(int i=0; i<vec_joined.size(); i++)
                out_array1.data.push_back((vec_joined)[i]);

            pub.publish(out_array1);
                // cout << (*vec)[i];
            // for (auto el: vec_joined.data())
            // {
            //     out_array1.data.push_back(el);
            // }
            
            // std::vector<float> out_array1;
            // out_array1.insert(out_array1.end(), major_vector.begin(), major_vector.end());

            // Eigen::Vector3f position_v(position.x, position.y, position.z);
            // Eigen::Quaternionf quat(rot_matrix);
            // viewer->addCube(position_v, quat, maxpoint.x - minpoint.x, maxpoint.y - minpoint.y, maxpoint.z - minpoint.z, "OBB_" + std::to_string(counter));
            // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "OBB_" + std::to_string(counter));

            // pcl::PointXYZRGB center(mass_center(0), mass_center(1), mass_center(2));
            // pcl::PointXYZRGB x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
            // pcl::PointXYZRGB y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
            // pcl::PointXYZRGB z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
            // viewer->addLine(center, x_axis, 1.0d, 0.0f, 0.0f, "major eigen vector " + std::to_string(counter));//main ingredient
            // viewer->addLine(center, y_axis, 0.0f, 1.0d, 0.0f, "middle eigen vector " + std::to_string(counter));
            // viewer->addLine(center, z_axis, 0.0f, 0.0f, 1.0d, "minor eigen vector " + std::to_string(counter));
            break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds> (end-start).count();

        std::cout << "FPS: " << 1000 / double(dur) << std::endl;

        // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cur_cloud);
        // viewer->addPointCloud<pcl::PointXYZRGB> (cur_cloud, rgb, "sample cloud");  

    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "sub_pcl");
    ros::NodeHandle nh;
    segmentPC node(&nh);

    ros::Rate loop_rate(50); //50 Hz
    while(nh.ok())
    {
        node.loop();
        ros::spinOnce();
        loop_rate.sleep();

    }
}

