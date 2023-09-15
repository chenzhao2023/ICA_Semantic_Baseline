import os
import argparse
import cv2
import numpy as np
import SimpleITK as sitk
import pickle

from tqdm import tqdm

import networkx as nx
import six

from skimage.measure import regionprops
from radiomics import featureextractor


def convert_graph_2_tree(g):
    def _is_branch_exist(branch_name):
        for node in g.nodes():
            if g.nodes()[node]['data'].vessel_class == branch_name:
                return True
        return False

    tree = {"features": [], 'labels': 'LMA', 'node_index': 0, 'children': [ 
                        {"features": [], 'labels': 'LAD1', 'node_index': 0, 'children': [ 
                                        {"features": [], 'labels': 'LAD2', 'node_index': 0, 'children': [ 
                                            {"features": [], 'labels': 'LAD3', 'node_index': 0, 'children': []}, 
                                            {"features": [], 'labels': 'D2', 'node_index': 0, 'children': []}, 
                                        ]},
                                        {"features": [], 'labels': 'D1', 'node_index': 0, 'children': []} 
                        ]},
                        {"features": [], 'labels': 'LCX1', 'node_index': 0, 'children': [
                                        {"features": [], 'labels': 'LCX2', 'node_index': 0, 'children': [
                                            {"features": [], 'labels': 'LCX3', 'node_index': 0, 'children': []},
                                            {"features": [], 'labels': 'OM2', 'node_index': 0, 'children': []},
                                        ]},
                                        {"features": [], 'labels': 'OM1', 'node_index': 0, 'children': []},
                        ]}
    ]} 

    if not _is_branch_exist("D2"):
        tree['children'][0]['children'][0]['children'] = [tree['children'][0]['children'][0]['children'][0]]

    if not _is_branch_exist("D1"):
        tree['children'][0]['children'] = [tree['children'][0]['children'][0]]

    if not _is_branch_exist("OM2"):
        tree['children'][1]['children'][0]['children'] = [tree['children'][1]['children'][0]['children'][0]]
    
    if not _is_branch_exist("OM1"):
        tree['children'][1]['children'] = [tree['children'][1]['children'][0]]

    # if no lad3
    lad2_end = True
    for node in g.nodes():
        if g.nodes()[node]['data'].vessel_class in ['LAD3', 'D2']:
            lad2_end = False
    if lad2_end:
        tree['children'][0]['children'][0]['children'] = []
    
    # if no lad2
    lad1_end = True
    for node in g.nodes():
        if g.nodes()[node]['data'].vessel_class in ['LAD2', 'D1']:
            lad1_end = False
    if lad1_end:
        tree['children'][0]['children'] = []

    # if no lcx3
    lcx2_end = True
    for node in g.nodes():
        if g.nodes()[node]['data'].vessel_class in ['LCX3', 'OM2']:
            lcx2_end = False
    if lcx2_end:
        tree['children'][1]['children'][0]['children'] = []
    
    # if no lad2
    lcx1_end = True
    for node in g.nodes():
        if g.nodes()[node]['data'].vessel_class in ['LCX2', 'OM1']:
            lcx1_end = False
    if lcx1_end:
        tree['children'][1]['children'] = []


    return tree

PARENT_MAPPING = {"LAD1": "LMA", "LAD2": "LAD1", "LAD3": "LAD2",
                  "LCX1": "LMA", "LCX2": "LCX1", "LCX3": "LCX2", 
                  "D1": "LAD1", "D2": "LAD2",
                  "OM1": "LCX1", "OM2": "LCX2"}

LABEL_MAPPING_ONE_HOT = {"LMA": [1,0,0,0,0],
                 "LAD": [0,1,0,0,0],
                 "LCX": [0,0,1,0,0],
                 "D":   [0,0,0,1,0],
                 "OM":  [0,0,0,0,1]}


def assign_info(g, tree):
    for node in g.nodes():
        assert len(g.nodes()[node]['data'].features) > 0
        if g.nodes()[node]['data'].vessel_class == 'LMA':
            tree["features"] = g.nodes()[node]['data'].features
            tree["labels"] = LABEL_MAPPING_ONE_HOT["LMA"]
            tree["node_index"] = node
        elif g.nodes()[node]['data'].vessel_class == 'LAD1':
            assert tree['children'][0]['labels'] == "LAD1"
            tree['children'][0]['labels'] = LABEL_MAPPING_ONE_HOT["LAD"]
            tree['children'][0]["features"] = g.nodes()[node]['data'].features
            tree['children'][0]["node_index"] = node
        elif g.nodes()[node]['data'].vessel_class == 'LAD2':
            assert tree['children'][0]['children'][0]['labels'] == "LAD2"
            tree['children'][0]['children'][0]["labels"] = LABEL_MAPPING_ONE_HOT["LAD"]
            tree['children'][0]['children'][0]["features"] = g.nodes()[node]['data'].features
            tree['children'][0]['children'][0]["node_index"] = node
        elif g.nodes()[node]['data'].vessel_class == 'LAD3':
            assert tree['children'][0]['children'][0]['children'][0]['labels'] == "LAD3"
            tree['children'][0]['children'][0]['children'][0]["labels"] = LABEL_MAPPING_ONE_HOT["LAD"]
            tree['children'][0]['children'][0]['children'][0]["features"] = g.nodes()[node]['data'].features
            tree['children'][0]['children'][0]['children'][0]["node_index"] = node
        elif g.nodes()[node]['data'].vessel_class == 'LCX1':
            assert tree['children'][1]['labels'] == "LCX1"
            tree['children'][1]['labels'] = LABEL_MAPPING_ONE_HOT["LCX"]
            tree['children'][1]["features"] = g.nodes()[node]['data'].features
            tree['children'][1]["node_index"] = node
        elif g.nodes()[node]['data'].vessel_class == 'LCX2':
            assert tree['children'][1]['children'][0]['labels'] == "LCX2"
            tree['children'][1]['children'][0]["labels"] = LABEL_MAPPING_ONE_HOT["LCX"]
            tree['children'][1]['children'][0]["features"] = g.nodes()[node]['data'].features
            tree['children'][1]['children'][0]["node_index"] = node
        elif g.nodes()[node]['data'].vessel_class == 'LCX3':
            assert tree['children'][1]['children'][0]['children'][0]["labels"] == "LCX3"
            tree['children'][1]['children'][0]['children'][0]["labels"] = LABEL_MAPPING_ONE_HOT["LCX"]
            tree['children'][1]['children'][0]['children'][0]["features"] = g.nodes()[node]['data'].features
            tree['children'][1]['children'][0]['children'][0]["node_index"] = node
        elif g.nodes()[node]['data'].vessel_class == 'D1':
            assert tree['children'][0]['children'][1]['labels'] == "D1"
            tree['children'][0]['children'][1]["labels"] = LABEL_MAPPING_ONE_HOT["D"]
            tree['children'][0]['children'][1]["features"] = g.nodes()[node]['data'].features
            tree['children'][0]['children'][1]["node_index"] = node
        elif g.nodes()[node]['data'].vessel_class == 'D2':
            assert tree['children'][0]['children'][0]['children'][1]['labels'] == "D2"
            tree['children'][0]['children'][0]['children'][1]["labels"] = LABEL_MAPPING_ONE_HOT["D"]
            tree['children'][0]['children'][0]['children'][1]["features"] = g.nodes()[node]['data'].features
            tree['children'][0]['children'][0]['children'][1]["node_index"] = node
        elif g.nodes()[node]['data'].vessel_class == 'OM1':
            assert tree['children'][1]['children'][1]['labels'] == "OM1"
            tree['children'][1]['children'][1]['labels'] = LABEL_MAPPING_ONE_HOT["OM"]
            tree['children'][1]['children'][1]['features'] = g.nodes()[node]['data'].features
            tree['children'][1]['children'][1]['node_index'] = node
        elif g.nodes()[node]['data'].vessel_class == 'OM2':
            assert tree['children'][1]['children'][0]['children'][1]['labels'] == "OM2"
            tree['children'][1]['children'][0]['children'][1]['labels'] = LABEL_MAPPING_ONE_HOT["OM"]
            tree['children'][1]['children'][0]['children'][1]['features'] = g.nodes()[node]['data'].features
            tree['children'][1]['children'][0]['children'][1]['node_index'] = node
    return tree


def extract_pos_feature(vessel_obj, binary_image, original_image):
    features = []
    feature_dict = {}
    # feature 1: number of vessel pixels
    feature_dict['n_pixels'] = np.sum(vessel_obj.vessel_mask == 1)
    features.append(np.sum(vessel_obj.vessel_mask == 1))
    # feature 2: length of centerlines
    feature_dict['n_centerline'] = np.sum(vessel_obj.vessel_centerline == 1)
    features.append(np.sum(vessel_obj.vessel_centerline == 1))
    # feature 5, 6: mean and stand deviation of the intensities within the centerline in grayscale image_x
    # vessel_centerline_roi = original_image * vessel_obj.vessel_centerline
    # features.append(np.mean(vessel_centerline_roi.flatten()[vessel_centerline_roi.flatten() > 0]))  # mean
    # features.append(np.std(vessel_centerline_roi.flatten()[vessel_centerline_roi.flatten() > 0]))  # std
    # feature 7-10: absolute position of vessel in whole binary image_x
    binary_properties = regionprops(binary_image, original_image)
    binary_center_of_mass = binary_properties[0].centroid
    binary_weighted_center_of_mass = binary_properties[0].weighted_centroid

    vessel_properties = regionprops(vessel_obj.vessel_mask, original_image)
    vessel_center_of_mass = vessel_properties[0].centroid
    vessel_weighted_center_of_mass = vessel_properties[0].weighted_centroid

    feature_dict["x_center"] = vessel_center_of_mass[0] / binary_center_of_mass[0]
    feature_dict["y_center"] = vessel_center_of_mass[1] / binary_center_of_mass[1]
    feature_dict["weighted_x_center"] = vessel_weighted_center_of_mass[0] / binary_weighted_center_of_mass[0]
    feature_dict["weighted_y_center"] = vessel_weighted_center_of_mass[1] / binary_weighted_center_of_mass[1]
    features.append(vessel_center_of_mass[0] / binary_center_of_mass[0])
    features.append(vessel_center_of_mass[1] / binary_center_of_mass[1])
    features.append(vessel_weighted_center_of_mass[0] / binary_weighted_center_of_mass[0])
    features.append(vessel_weighted_center_of_mass[1] / binary_weighted_center_of_mass[1])

    # feature 11-18: absolute position of start and end point to center of binary image_x
    if vessel_obj.node1.x < vessel_obj.node2.x:
        features.append(vessel_obj.node1.x / binary_center_of_mass[0])
        features.append(vessel_obj.node1.y / binary_center_of_mass[1])
        features.append(vessel_obj.node1.x / binary_weighted_center_of_mass[0])
        features.append(vessel_obj.node1.y / binary_weighted_center_of_mass[1])
        features.append(vessel_obj.node2.x / binary_center_of_mass[0])
        features.append(vessel_obj.node2.y / binary_center_of_mass[1])
        features.append(vessel_obj.node2.x / binary_weighted_center_of_mass[0])
        features.append(vessel_obj.node2.y / binary_weighted_center_of_mass[1])

        feature_dict["p1_x_center"] = vessel_obj.node1.x / binary_center_of_mass[0]
        feature_dict["p1_y_center"] = vessel_obj.node1.x / binary_center_of_mass[0]
        feature_dict["p1_x_weighted_center"] = vessel_obj.node1.x / binary_center_of_mass[0]
        feature_dict["p1_y_weighted_center"] = vessel_obj.node1.x / binary_center_of_mass[0]
        feature_dict["p2_x_center"] = vessel_obj.node2.x / binary_center_of_mass[0]
        feature_dict["p2_y_center"] = vessel_obj.node2.y / binary_center_of_mass[1]
        feature_dict["p2_x_weighted_center"] = vessel_obj.node2.x / binary_weighted_center_of_mass[0]
        feature_dict["p2_y_weighted_center"] = vessel_obj.node2.y / binary_weighted_center_of_mass[1]
    else:
        features.append(vessel_obj.node2.x / binary_center_of_mass[0])
        features.append(vessel_obj.node2.y / binary_center_of_mass[1])
        features.append(vessel_obj.node2.x / binary_weighted_center_of_mass[0])
        features.append(vessel_obj.node2.y / binary_weighted_center_of_mass[1])
        features.append(vessel_obj.node1.x / binary_center_of_mass[0])
        features.append(vessel_obj.node1.y / binary_center_of_mass[1])
        features.append(vessel_obj.node1.x / binary_weighted_center_of_mass[0])
        features.append(vessel_obj.node1.y / binary_weighted_center_of_mass[1])

        feature_dict["p1_x_center"] = vessel_obj.node2.x / binary_center_of_mass[0]
        feature_dict["p1_y_center"] = vessel_obj.node2.y / binary_center_of_mass[1]
        feature_dict["p1_x_weighted_center"] = vessel_obj.node2.x / binary_weighted_center_of_mass[0]
        feature_dict["p1_y_weighted_center"] = vessel_obj.node2.y / binary_weighted_center_of_mass[1]
        feature_dict["p2_x_center"] = vessel_obj.node1.x / binary_center_of_mass[0]
        feature_dict["p2_y_center"] = vessel_obj.node1.y / binary_center_of_mass[1]
        feature_dict["p2_x_weighted_center"] = vessel_obj.node1.x / binary_weighted_center_of_mass[0]
        feature_dict["p2_y_weighted_center"] = vessel_obj.node1.y / binary_weighted_center_of_mass[1]

    # feature 19-26 : absolute position of start and end point to center of vessel segment image_x
    if vessel_obj.node1.x < vessel_obj.node2.x:
        features.append(vessel_obj.node1.x / vessel_center_of_mass[0])
        features.append(vessel_obj.node1.y / vessel_center_of_mass[1])
        features.append(vessel_obj.node1.x / vessel_weighted_center_of_mass[0])
        features.append(vessel_obj.node1.y / vessel_weighted_center_of_mass[1])
        features.append(vessel_obj.node2.x / vessel_center_of_mass[0])
        features.append(vessel_obj.node2.y / vessel_center_of_mass[1])
        features.append(vessel_obj.node2.x / vessel_weighted_center_of_mass[0])
        features.append(vessel_obj.node2.y / vessel_weighted_center_of_mass[1])

        feature_dict["p1_abs_x_center"] = vessel_obj.node1.x / vessel_center_of_mass[0]
        feature_dict["p1_abs_y_center"] = vessel_obj.node1.y / vessel_center_of_mass[1]
        feature_dict["p1_abs_x_weighted_center"] = vessel_obj.node1.x / vessel_weighted_center_of_mass[0]
        feature_dict["p1_abs_y_weighted_center"] = vessel_obj.node1.y / vessel_weighted_center_of_mass[1]
        feature_dict["p2_abs_x_center"] = vessel_obj.node2.x / vessel_center_of_mass[0]
        feature_dict["p2_abs_y_center"] = vessel_obj.node2.y / vessel_center_of_mass[1]
        feature_dict["p2_abs_x_weighted_center"] = vessel_obj.node2.x / vessel_weighted_center_of_mass[0]
        feature_dict["p2_abs_y_weighted_center"] = vessel_obj.node2.y / vessel_weighted_center_of_mass[1]
    else:
        features.append(vessel_obj.node2.x / vessel_center_of_mass[0])
        features.append(vessel_obj.node2.y / vessel_center_of_mass[1])
        features.append(vessel_obj.node2.x / vessel_weighted_center_of_mass[0])
        features.append(vessel_obj.node2.y / vessel_weighted_center_of_mass[1])
        features.append(vessel_obj.node1.x / vessel_center_of_mass[0])
        features.append(vessel_obj.node1.y / vessel_center_of_mass[1])
        features.append(vessel_obj.node1.x / vessel_weighted_center_of_mass[0])
        features.append(vessel_obj.node1.y / vessel_weighted_center_of_mass[1])

        feature_dict["p1_abs_x_center"] = vessel_obj.node2.x / vessel_center_of_mass[0]
        feature_dict["p1_abs_y_center"] = vessel_obj.node2.y / vessel_center_of_mass[1]
        feature_dict["p1_abs_x_weighted_center"] = vessel_obj.node2.x / vessel_weighted_center_of_mass[0]
        feature_dict["p1_abs_y_weighted_center"] = vessel_obj.node2.y / vessel_weighted_center_of_mass[1]
        feature_dict["p2_abs_x_center"] = vessel_obj.node1.x / vessel_center_of_mass[0]
        feature_dict["p2_abs_y_center"] = vessel_obj.node1.y / vessel_center_of_mass[1]
        feature_dict["p2_abs_x_weighted_center"] = vessel_obj.node1.x / vessel_weighted_center_of_mass[0]
        feature_dict["p2_abs_y_weighted_center"] = vessel_obj.node1.y / vessel_weighted_center_of_mass[1]

    # feature 27, 28: degree of two points
    if vessel_obj.node1.x < vessel_obj.node2.x:
        features.append(vessel_obj.node1.degree)
        features.append(vessel_obj.node2.degree)
        feature_dict["p1_degree"] = vessel_obj.node1.degree
        feature_dict["p2_degree"] = vessel_obj.node2.degree

    else:
        features.append(vessel_obj.node2.degree)
        features.append(vessel_obj.node1.degree)
        feature_dict["p1_degree"] = vessel_obj.node2.degree
        feature_dict["p2_degree"] = vessel_obj.node1.degree

    # feature 29, 30, 31, 32: mean, std, min, max of vascular radius
    radius = vessel_obj.vessel_centerline_dist
    radius = radius[radius > 0]
    features.append(np.mean(radius))
    features.append(np.std(radius))
    features.append(np.min(radius))
    features.append(np.max(radius))
    feature_dict["r_mean"] = np.mean(radius)
    feature_dict["r_std"] = np.std(radius)
    feature_dict["r_min"] = np.min(radius)
    feature_dict["r_max"] = np.max(radius)

    return feature_dict


def extract_features(g, binary_image, original_image, step, patch_size):

    def __cart_to_pol(points, a=0, b=0):
        points = np.array(points)
        if len(points.shape) == 1:
            points = np.expand_dims(points, 0)
        rho = np.sqrt((points[:,0]-a)**2 + (points[:,1]-b)**2)
        phi = np.arctan2((points[:,1]-a), (points[:,0]-b))
        return rho, phi

    def __find_segment_by_name(branch_name):
        for node in g.nodes():
            if g.nodes()[node]['data'].vessel_class == branch_name:
                return g.nodes()[node]['data']
        raise ValueError("Cannot find branch {}".format(branch_name))

    def __check_and_move(coord, bound_size=binary_image.shape[0]):
        if coord + patch_size//2 > bound_size:
            coord = bound_size - patch_size//2
        if coord < patch_size//2:
            coord = patch_size//2
        return coord


    binary_properties = regionprops(binary_image, original_image)
    binary_center_of_mass = binary_properties[0].centroid # original of polar coord
    
    # convert graph into tree structured data
    tree = convert_graph_2_tree(g) # tree will be used for tree-lstm models

    for node in g.nodes():
        features = []
        vessel_segment = g.nodes()[node]['data']

        centerline_x, centerline_y = np.where(vessel_segment.vessel_centerline>0)[0], np.where(vessel_segment.vessel_centerline>0)[1]
        points = [(centerline_x[i], centerline_y[i]) for i in range(len(centerline_x))]
        rhos, phis = __cart_to_pol(points, a=binary_center_of_mass[0], b=binary_center_of_mass[1])

        # image features I1,...IN
        patches = []
        for i in np.arange(0, len(points), step=step):
            patch_x = __check_and_move(points[i][0])
            patch_y = __check_and_move(points[i][1])
            patch_im = original_image[patch_x-patch_size//2: patch_x+patch_size//2, 
                                      patch_y-patch_size//2: patch_y+patch_size//2]
            patch_im = patch_im / 255.
            patches.append(patch_im)
        if len(patches) == 1:
            patches.append(patches[0])

        print(f"branch_name {g.nodes()[node]['data'].vessel_class}, # patches = {len(patches)}")

        vessel_segment.patches = patches
        # type II features: SCT2D coordinates of the 1st point, center point and ending point
        for i in [0, len(rhos)//2, len(rhos)-1]:
            features.append(rhos[i])
            features.append(phis[i])
        
        # type III features: directions in SCT2D coordinates: directional vector between start and ending points
        vector = np.array(points[0]) - np.array(points[-1])
        rho, phi = __cart_to_pol(vector, a=binary_center_of_mass[0], b=binary_center_of_mass[1])
        features.append(rho[0])
        features.append(phi[0])

        pos_features = extract_pos_feature(vessel_segment, binary_image, original_image)
        # assert len(features) == 8
        for key in pos_features.keys():
            features.append(pos_features[key])
    
        vessel_segment.features = features

        vessel_segment.vessel_centerline = np.asarray(vessel_segment.vessel_centerline, np.uint8)
        vessel_segment.vessel_mask = None
        vessel_segment.vessel_centerline_dist = None

    tree = assign_info(g, tree)
    return g, tree


def _get_sample_list(txt_file_path="artery/data"):
    dataset0 = []
    dataset1 = []
    with open(os.path.join(txt_file_path, "selected_subjects.txt"), "r") as f:
        for row in f.readlines():
            if row[0].isdigit():
                dataset1.append(row.strip())
            elif row[0].isalpha() and (row.strip() not in ["NJ", "TW"]):
                dataset0.append(row.strip())

    return {"NJ": dataset1, "TW": dataset0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/media/z/data21/artery_semantic_segmentation')
    parser.add_argument('--data_path', type=str, default="gmn_vessel")
    parser.add_argument('--save_path', type=str, default="artery/data")
    parser.add_argument('--project_path', type=str, default="ThinkMatch")
    parser.add_argument('--image_size', type=int, default=512)

    # for patch extraction
    parser.add_argument('--patch_size', type=int, default=12)
    parser.add_argument('--step', type=int, default=12)

    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    sample_list = _get_sample_list()

    FEATURE_DICT_ALL = {}
    for dataset_name, data_path in zip(["TW", "NJ"],
                                       [os.path.join(args.base_path, "gmn_vessel/data/data_tw_semantic/processed"),
                                        os.path.join(args.base_path, "gmn_vessel/data/data_nj_semantic/processed")]):
        selected_sample_names = sample_list[dataset_name]
        for selected_sample_name in tqdm(selected_sample_names):
            print(f"[x] processing {selected_sample_name} ---- ")
            pkl_file_path = os.path.join(args.base_path, data_path, selected_sample_name, f"{selected_sample_name}_g_switch_unique.pkl")

            binary_image = cv2.imread(os.path.join(args.base_path, data_path, selected_sample_name, f"{selected_sample_name}_binary_image.png"), cv2.IMREAD_GRAYSCALE)
            if binary_image.shape[0] != args.image_size:
                binary_image = cv2.resize(binary_image, (args.image_size, args.image_size))

            original_image = cv2.imread(os.path.join(args.base_path, data_path, selected_sample_name, f"{selected_sample_name}.png"), cv2.IMREAD_GRAYSCALE)
            if original_image.shape[0] != args.image_size:
                original_image = cv2.resize(original_image, (args.image_size, args.image_size))
            print(pkl_file_path)
            g = pickle.load(open(pkl_file_path, 'rb'))
            save_path = os.path.join(args.base_path, args.project_path, args.save_path)
            g, tree = extract_features(g, binary_image, original_image, args.step, args.patch_size)

            pickle.dump(tree, open(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}_tree.pkl"), "wb"))
            pickle.dump(g, open(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}.pkl"), "wb"))
            cv2.imwrite(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}_binary_image.png"), binary_image)
            cv2.imwrite(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}.png"), original_image)

            semantic_image = cv2.imread(os.path.join(args.base_path, data_path, selected_sample_name, f"{selected_sample_name}_step12_g_switch_unique_semantic_image.png"))
            cv2.imwrite(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}_semantic.png"), semantic_image)
