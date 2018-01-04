#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

class Dropbox:
    def __init__(self, color, arm, x, y, z):
        self.color = color
        self.arm = arm
        self.position = []
        self.position.append(x)
        self.position.append(y)
        self.position.append(z)

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 code:

    # Convert ROS msg to PCL data
    cloud_filtered = ros_to_pcl(pcl_msg)
   
   # Voxel Grid Downsampling
    vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # Statistical Outlier Filtering
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(5) # orig 50

    # Set threshold scale factor
    x = -0.5 # orig 1.0

    # Any point with a mean distance larger than (global_mean_distance + x * global_std_dev) is considered an outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Perform filtering (outliers are filtered out)
    cloud_filtered = outlier_filter.filter()

    # PassThrough Filter 1
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    # PassThrough Filter 2
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.43
    axis_max = 0.43
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

     # RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False) # table
    extracted_outliers = cloud_filtered.extract(inliers, negative=True) # tabletop objects

    # Statistical Outlier Filtering
    outlier_filter = extracted_outliers.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(90) # orig 50

    # Set threshold scale factor
    x = 0.5 # orig 1.0

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Perform filtering (outliers are filtered out)
    cloud_filtered = outlier_filter.filter()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()

    ec = white_cloud.make_EuclideanClusterExtraction()
    # Tune these parameters
    ec.set_ClusterTolerance(0.05)  # 0.001
    ec.set_MinClusterSize(10)      # 10
    ec.set_MaxClusterSize(3000)    # 250

    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract() # a list of lists, one for each object

    # Create final point cloud where points of different lists have different colors
    cluster_colors = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, index in enumerate(indices):
            color_cluster_point_list.append([white_cloud[index][0], \
                                             white_cloud[index][1], \
                                             white_cloud[index][2], \
                                             rgb_to_float(cluster_colors[j])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cloud_table = pcl_to_ros(extracted_inliers) # table
    ros_cloud_objects = pcl_to_ros(extracted_outliers) # objects
    ros_cluster_cloud = pcl_to_ros(cluster_cloud) # colored objects

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

# Exercise-3 code: 

    # Classify the clusters!
    detected_objects_labels = []
    detected_objects = []

    # Loop through each detected cluster one at a time
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = extracted_outliers.extract(pts_list)

        # Convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # See src/sensor_stick/src/sensor_stick/capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)


    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Initialize variables and "constants"
    scene_index = 3
    yaml_dict_list = []

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_list_param = rospy.get_param('/dropbox')
    boxes = {}
    for box in dropbox_list_param:
        boxes[box['group']] = Dropbox(box['group'], box['name'], box['position'][0], box['position'][1], box['position'][2]) 
    boxes['unknown'] = Dropbox('unknown', 'unknown', 0, 0, 0) 

    # loop through pick list
    for pick_object in object_list_param:
        pick_object_name = pick_object['name']
        pick_object_group = pick_object['group']

        # loop through detected objects to look for the pick list object
        for detected_object in object_list:

            # has such an object been detected? NOTE: misclassified ones will be included
            if pick_object_name == detected_object.label:

                # set test_scene_num
                test_scene_num = Int32()
                test_scene_num.data = scene_index

                # set arm_name
                arm_name = String()
                arm_name.data = boxes.get(pick_object_group, 'unknown').arm

                # set object_name
                object_name = String()
                object_name.data = pick_object_name

                # set pick_pose with object centroid
                points_arr = ros_to_pcl(detected_object.cloud).to_array()
                centroid = np.mean(points_arr, axis=0)[:3]
                pick_pose = Pose()
                pick_pose.position.x = np.asscalar(centroid[0])
                pick_pose.position.y = np.asscalar(centroid[1])
                pick_pose.position.z = np.asscalar(centroid[2])
                pick_pose.orientation.x = 0
                pick_pose.orientation.y = 0
                pick_pose.orientation.z = 0
                pick_pose.orientation.w = 0

                # set place_pose
                place_pose = Pose()
                dropbox = boxes.get(pick_object_group, 'unknown')
                place_pose.position.x = dropbox.position[0]
                place_pose.position.y = dropbox.position[1]
                place_pose.position.z = dropbox.position[2]
                place_pose.orientation.x = 0
                place_pose.orientation.y = 0
                place_pose.orientation.z = 0
                place_pose.orientation.w = 0

                yaml_dict_list.append(make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose))

        # Wait for 'pick_place_routine' service to come up
#       rospy.wait_for_service('pick_place_routine')
#
#        try:
#            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
#
#            # TODO: Insert your message variables to be sent as a service request
#            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)
#
#            print ("Response: ",resp.success)
#
#        except rospy.ServiceException, e:
#            print "Service call failed: %s"%e

    # Output your request parameters into output yaml file
    send_to_yaml('output_'+str(scene_index)+'.yaml', yaml_dict_list)

if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('perception', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber('/pr2/world/points', pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher('/pcl_objects', PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher('/pcl_table', PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher('/pcl_cluster', PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher('/object_markers', Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher('/detected_objects', DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb')) # NOTE: Run script in the dir containing model.sav
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

     # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
   
