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
from glob import glob

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


def extract_features_with_random(g, binary_image, original_image, ban_list):

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

    binary_properties = regionprops(binary_image, original_image)
    binary_center_of_mass = binary_properties[0].centroid # original of polar coord
    
    # convert graph into tree structured data
    tree = convert_graph_2_tree(g) # tree will be used for tree-lstm models

    for node in g.nodes():
        features = []
        vessel_segment = g.nodes()[node]['data']
        
        vessel_segment.vessel_centerline = np.asarray(vessel_segment.vessel_centerline, np.uint8)
        vessel_segment.vessel_mask = None
        vessel_segment.vessel_centerline_dist = None

        centerline_x, centerline_y = np.where(vessel_segment.vessel_centerline>0)[0], np.where(vessel_segment.vessel_centerline>0)[1]
        points = [(centerline_x[i], centerline_y[i]) for i in range(len(centerline_x))]
        rhos, phis = __cart_to_pol(points, a=binary_center_of_mass[0], b=binary_center_of_mass[1])


        # type I features, angle between the directional vector at the end of parent segment 
        # and the directional vector at the start of child segment:
        if vessel_segment.vessel_class == 'LMA':
            features.append(0)
            features.append(0)
        else:
            parent_segment_class = PARENT_MAPPING[vessel_segment.vessel_class]
            parent_vessel_segment = __find_segment_by_name(parent_segment_class)
            centerline_x_p, centerline_y_p = np.where(parent_vessel_segment.vessel_centerline>0)[0], np.where(parent_vessel_segment.vessel_centerline>0)[1]
            parent_points = [(centerline_x_p[i], centerline_y_p[i]) for i in range(len(centerline_x_p))]
            parent_end_vector = np.array(parent_points[-2]) - np.array(parent_points[-1])
            child_start_vector = np.array(points[0]) - np.array(points[1])
            vector = parent_end_vector - child_start_vector
            rho, phi = __cart_to_pol(vector, a=binary_center_of_mass[0], b=binary_center_of_mass[1])
            features.append(rho[0])
            features.append(phi[0])

        # type II features: SCT2D coordinates of the 1st point, center point and ending point
        for i in [0, len(rhos)//2, len(rhos)-1]:
            features.append(rhos[i])
            features.append(phis[i])
        
        # type III features: directions in SCT2D coordinates: directional vector between start and ending points
        vector = np.array(points[0]) - np.array(points[-1])
        rho, phi = __cart_to_pol(vector, a=binary_center_of_mass[0], b=binary_center_of_mass[1])
        features.append(rho[0])
        features.append(phi[0])

        # and tangential direction at the start point
        vector = np.array(points[0]) - np.array(points[1])
        rho, phi = __cart_to_pol(vector, a=binary_center_of_mass[0], b=binary_center_of_mass[1])
        features.append(rho[0])
        features.append(phi[0])
        
        assert len(features) == 12
    
        vessel_segment.features = features

        if vessel_segment.vessel_class in ban_list:
            vessel_segment.features = np.random.rand(12)

    # pickle.dump(g, open(os.path.join(save_path, f"{selected_sample_name}.pkl"), "wb"))
    # cv2.imwrite(os.path.join(save_path, f"{selected_sample_name}_binary_image.png"), binary_image)
    # cv2.imwrite(os.path.join(save_path, f"{selected_sample_name}.png"), original_image)

    tree = assign_info(g, tree)
    return g, tree


def extract_features(g, binary_image, original_image, save_path):

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

    binary_properties = regionprops(binary_image, original_image)
    binary_center_of_mass = binary_properties[0].centroid # original of polar coord
    
    # convert graph into tree structured data
    tree = convert_graph_2_tree(g) # tree will be used for tree-lstm models

    for node in g.nodes():
        features = []
        vessel_segment = g.nodes()[node]['data']
        
        vessel_segment.vessel_centerline = np.asarray(vessel_segment.vessel_centerline, np.uint8)
        vessel_segment.vessel_mask = None
        vessel_segment.vessel_centerline_dist = None

        centerline_x, centerline_y = np.where(vessel_segment.vessel_centerline>0)[0], np.where(vessel_segment.vessel_centerline>0)[1]
        points = [(centerline_x[i], centerline_y[i]) for i in range(len(centerline_x))]
        rhos, phis = __cart_to_pol(points, a=binary_center_of_mass[0], b=binary_center_of_mass[1])


        # type I features, angle between the directional vector at the end of parent segment 
        # and the directional vector at the start of child segment:
        if vessel_segment.vessel_class == 'LMA':
            features.append(0)
            features.append(0)
        else:
            parent_segment_class = PARENT_MAPPING[vessel_segment.vessel_class]
            parent_vessel_segment = __find_segment_by_name(parent_segment_class)
            centerline_x_p, centerline_y_p = np.where(parent_vessel_segment.vessel_centerline>0)[0], np.where(parent_vessel_segment.vessel_centerline>0)[1]
            parent_points = [(centerline_x_p[i], centerline_y_p[i]) for i in range(len(centerline_x_p))]
            parent_end_vector = np.array(parent_points[-2]) - np.array(parent_points[-1])
            child_start_vector = np.array(points[0]) - np.array(points[1])
            vector = parent_end_vector - child_start_vector
            rho, phi = __cart_to_pol(vector, a=binary_center_of_mass[0], b=binary_center_of_mass[1])
            features.append(rho[0])
            features.append(phi[0])

        # type II features: SCT2D coordinates of the 1st point, center point and ending point
        for i in [0, len(rhos)//2, len(rhos)-1]:
            features.append(rhos[i])
            features.append(phis[i])
        
        # type III features: directions in SCT2D coordinates: directional vector between start and ending points
        vector = np.array(points[0]) - np.array(points[-1])
        rho, phi = __cart_to_pol(vector, a=binary_center_of_mass[0], b=binary_center_of_mass[1])
        features.append(rho[0])
        features.append(phi[0])

        # and tangential direction at the start point
        vector = np.array(points[0]) - np.array(points[1])
        rho, phi = __cart_to_pol(vector, a=binary_center_of_mass[0], b=binary_center_of_mass[1])
        features.append(rho[0])
        features.append(phi[0])
        
        assert len(features) == 12
    
        vessel_segment.features = features

    # pickle.dump(g, open(os.path.join(save_path, f"{selected_sample_name}.pkl"), "wb"))
    # cv2.imwrite(os.path.join(save_path, f"{selected_sample_name}_binary_image.png"), binary_image)
    # cv2.imwrite(os.path.join(save_path, f"{selected_sample_name}.png"), original_image)

    tree = assign_info(g, tree)
    return g, tree


def _get_sample_list(data_path):
    subject_files = glob(f"{data_path}/*.pkl")
    subjects = []
    for subject_file in subject_files:
        subject = subject_file[subject_file.rfind("/")+1: subject_file.rfind(".pkl")]
        subjects.append(subject)
    return subjects


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/media/z/data2/artery_semantic_segmentation')
    parser.add_argument('--data_path', type=str, default="/media/z/data2/artery_semantic_segmentation/hnn_hm_june/data/artery_with_feature")
    parser.add_argument('--save_path', type=str, default="data_aaai")
    parser.add_argument('--project_path', type=str, default="pytorch-tree-lstm")
    parser.add_argument('--image_size', type=int, default=512)
    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    sample_list = _get_sample_list(args.data_path)

    FEATURE_DICT_ALL = {}
    for selected_sample_name in tqdm(sample_list):
        print(f"[x] processing {selected_sample_name} ---- ")
        pkl_file_path = os.path.join(args.data_path, f"{selected_sample_name}.pkl")

        binary_image = cv2.imread(os.path.join(args.data_path, f"{selected_sample_name}_binary_image.png"), cv2.IMREAD_GRAYSCALE)
        if binary_image is None:
            continue
        if binary_image.shape[0] != args.image_size:
            binary_image = cv2.resize(binary_image, (args.image_size, args.image_size))

        original_image = cv2.imread(os.path.join(args.data_path, f"{selected_sample_name}.png"), cv2.IMREAD_GRAYSCALE)
        if original_image.shape[0] != args.image_size:
            original_image = cv2.resize(original_image, (args.image_size, args.image_size))
        print(pkl_file_path)
        g = pickle.load(open(pkl_file_path, 'rb'))
        save_path = os.path.join(args.base_path, args.project_path, args.save_path)
        g, tree = extract_features(g, binary_image, original_image, save_path)

        pickle.dump(tree, open(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}_tree.pkl"), "wb"))
        pickle.dump(g, open(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}.pkl"), "wb"))
        cv2.imwrite(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}_binary_image.png"), binary_image)
        cv2.imwrite(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}.png"), original_image)

        # semantic_image = cv2.imread(os.path.join(args.data_path, selected_sample_name, f"{selected_sample_name}_step12_g_switch_unique_semantic_image.png"))
        # cv2.imwrite(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}_semantic.png"), semantic_image)

    # normalize features
    # sorted_keys = sorted(FEATURE_DICT_ALL)
    # for k in sorted_keys:
    #     print(f"k = {k}, min = {np.min(FEATURE_DICT_ALL[k])}, max = {np.max(FEATURE_DICT_ALL[k])}, len = {len(FEATURE_DICT_ALL[k])}")

    # for dataset_name, data_path in zip(["TW", "NJ"],
    #                                    [os.path.join(args.base_path, "gmn_vessel/data/data_tw_semantic/processed"),
    #                                     os.path.join(args.base_path, "gmn_vessel/data/data_nj_semantic/processed")]):
    #     selected_sample_names = sample_list[dataset_name]
    #     for selected_sample_name in tqdm(selected_sample_names):
    #         print(f"[x] processing {selected_sample_name} ---- ")
    #         pkl_file_path = os.path.join(args.base_path, data_path, selected_sample_name, f"{selected_sample_name}_g_switch_unique.pkl")

    #         binary_image = cv2.imread(os.path.join(args.base_path, data_path, selected_sample_name, f"{selected_sample_name}_binary_image.png"), cv2.IMREAD_GRAYSCALE)
    #         if binary_image.shape[0] != args.image_size:
    #             binary_image = cv2.resize(binary_image, (args.image_size, args.image_size))

    #         original_image = cv2.imread(os.path.join(args.base_path, data_path, selected_sample_name, f"{selected_sample_name}.png"), cv2.IMREAD_GRAYSCALE)
    #         if original_image.shape[0] != args.image_size:
    #             original_image = cv2.resize(original_image, (args.image_size, args.image_size))

    #         print(pkl_file_path)
    #         g = pickle.load(open(pkl_file_path, 'rb'))
    #         extract_features(g, binary_image, original_image, 
    #                     save_path=os.path.join(args.base_path, args.project_path, args.save_path))
    #         pickle.dump(g, open(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}.pkl"), "wb"))