import open3d as o3d
import numpy as np
import scipy.io as sio
import re
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
from urllib.request import urlretrieve
import sys
from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

# Clear terminal
from os import system; system('clear')

""" Inputs """
""" Load multiway DGR config file """ 
config = get_config()
pair_registration_method = config.pair_registration_method   # 'icp' | 'dgr'
voxel_size  = config.multiway_voxel_size
showPlots   = config.showPlots
pcd_folder  =  ' '.join(config.pcd_folder)
# pcd_folder  =  config.pcd_folder
firstNfiles = 6 #config.firstNfiles
verbose     = config.verbose

max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

# Point clouds file formats
pointcloud_ext = ['.ply', '.PLY', '.pcd', '.PCD']

""" Load weights and instantiate DGR """ 

if config.weights is None:
    config.weights = "ResUNetBN2C-feat32-3dmatch-v0.05.pth"
dgr = DeepGlobalRegistration(config)

""" Natural sorting """
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

""" Load point clouds """
def load_point_clouds(folder, pcd_extension, voxel_size=0.0):
    pcds = []
    files = sorted_alphanumeric(os.listdir(folder))

    if firstNfiles == 0:
        filesNum = len(files)
    else:
        filesNum = firstNfiles

    for index, filename in enumerate(files[0:filesNum]):
        ext = os.path.splitext(filename)[1] 
        if ext.lower() not in pcd_extension:
            continue
        pcdFile = os.path.join(folder,filename)
        pcd = o3d.io.read_point_cloud(pcdFile)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds

""" Pairwise registration """
def pairwise_registration(source, target, pair_registration_method ='icp'):
    
    if pair_registration_method == 'icp':
        print("Apply Pairwise point-to-plane ICP")
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_coarse, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        icp_fine = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_fine,
            icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        transformation = icp_fine.transformation
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance_fine,
            icp_fine.transformation)
    else:
        print("Apply Pairwise DGR")
        transformation, rmse = dgr.register(source, target)
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance_fine,
            transformation)

    return transformation, information

""" Full registration """
def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine, pair_registration_method):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_pairwise, information_pairwise = pairwise_registration(
                pcds[source_id], pcds[target_id], pair_registration_method)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_pairwise, odometry)    
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_pairwise,
                                                             information_pairwise,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_pairwise,
                                                             information_pairwise,
                                                             uncertain=True))
    return pose_graph


""" Multiway registration """
def main():

    """ Load point clouds and visualize """
    pcds_down = load_point_clouds(pcd_folder, pointcloud_ext, voxel_size)
    if showPlots:
        o3d.visualization.draw_geometries(pcds_down,
                                    zoom=0.7,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])

    """ Full registration """
    print("Full registration ...")
    if verbose:
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            pose_graph = full_registration(pcds_down,
                                        max_correspondence_distance_coarse,
                                        max_correspondence_distance_fine, pair_registration_method)
    else:
        pose_graph = full_registration(pcds_down,
                            max_correspondence_distance_coarse,
                            max_correspondence_distance_fine, pair_registration_method)



    """ Open3D uses the function global_optimization to perform pose graph optimization. """
    """ Pose graph """
    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    if verbose:
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)
    else:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)



    """ Visualize optimization
    The transformed point clouds are listed and visualized using draw_geometries. """
    print("Transform points and display")
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    
    if showPlots:
        o3d.visualization.draw_geometries(pcds_down,
                                        zoom=0.7,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])


    """ Make a combined point cloud.
    Visualize the multiway total point clouds """
    pcds = load_point_clouds(pcd_folder, pointcloud_ext, voxel_size)
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)

    if showPlots:
        o3d.visualization.draw_geometries([pcd_combined_down],
                                        zoom=0.7,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])

    """ Save MAT file of the transformations """
    allTransformations = np.empty([4,4, len(pcds_down)], np.float64)
    for point_id in range(len(pcds_down)):
        allTransformations[:,:, point_id] = pose_graph.nodes[point_id].pose

    # Save to MATLAB MAT file
    sio.savemat('transMat.mat', {'tforms': allTransformations})


""" Main function callback multiway total point clouds """
if __name__ == "__main__":
    main()