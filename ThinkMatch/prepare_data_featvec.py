import os
import torch
import argparse
import cv2
import numpy as np
import SimpleITK as sitk
import pickle
import networkx as nx

from tqdm import tqdm

import six

from skimage.measure import regionprops
from radiomics import featureextractor
from glob import glob
from tqdm import tqdm
from artery.Artery import _load_graph, ARTERY_CATEGORY
from train_artery_ngm_feat import collate_fn


class FeatureExtractor:

    def __init__(self):
        self.__init_config__()

    def __init_config__(self):
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('gldm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeatureClassByName('glszm')
        extractor.enableFeatureClassByName('ngtdm')

        print('Extraction parameters:\n\t', extractor.settings)
        print('Enabled filters:\n\t', extractor.enabledImagetypes)
        print('Enabled features:\n\t', extractor.enabledFeatures)
        self.extractor = extractor

    def extract_pos_feature(self, vessel_obj, binary_image, original_image):
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

        features.append(vessel_obj.vessel_class)
        return feature_dict

    def extract_radiomics_features(self, vessel_obj, original_image):
        # extract features from predicted numpy arrays
        data_x = sitk.GetImageFromArray(original_image)
        data_y = sitk.GetImageFromArray(vessel_obj.vessel_mask)

        # columns_drop = ["diagnostics_Configuration_EnabledImageTypes", "diagnostics_Configuration_Settings",
        #                 "diagnostics_Image-original_Dimensionality", "diagnostics_Image-original_Hash",
        #                 "diagnostics_Image-original_Size", "diagnostics_Image-original_Spacing",
        #                 "diagnostics_Mask-original_BoundingBox", "diagnostics_Mask-original_CenterOfMass",
        #                 "diagnostics_Mask-original_CenterOfMassIndex", "diagnostics_Mask-original_Hash",
        #                 "diagnostics_Mask-original_Size", "diagnostics_Mask-original_Spacing",
        #                 "diagnostics_Versions_Numpy", "diagnostics_Versions_PyRadiomics",
        #                 "diagnostics_Versions_PyWavelet", "diagnostics_Versions_Python",
        #                 "diagnostics_Versions_SimpleITK",
        #                 "diagnostics_Image-original_Maximum",
        #                 "diagnostics_Image-original_Mean",
        #                 "diagnostics_Image-original_Minimum"]

        feature = self.extractor.execute(data_x, data_y, label=1)
        keys = []
        feature_dict = {}
        for key, value in sorted(six.iteritems(feature)):
            if not key.startswith("diagnostics"):
                #print('\t', key, ':', value)
                if type(value) == np.ndarray:
                    keys.append(key)
                    feature_dict[key] = np.float(value)
                else:
                    raise ValueError("CANNOT PARSE VALUE")

        #print("# number of keys {}".format(len(keys)))
        # write csv header
        return feature_dict

    def get_class(self, vessel_obj):
        return vessel_obj.vessel_class


def _get_sample_list():
    dataset0 = []
    dataset1 = []
    with open("selected_subjects.txt", "r") as f:
    #with open("/media/z/data2/artery_semantic_segmentation/gmn_4_semantic_seg/selected_subjects.txt", "r") as f:
        for row in f.readlines():
            if row[0].isdigit():
                dataset1.append(row.strip())
            elif row[0].isalpha() and (row.strip() not in ["NJ", "TW"]):
                dataset0.append(row.strip())

    return {"NJ": dataset1, "TW": dataset0}


def get_feature_names():
    from core.utils.module import VesselSegment, Node
    vessel_obj = VesselSegment(Node(1, 100, 100), Node(3, 120, 120), np.random.rand(128,128))
    vessel_obj.vessel_class = 1

    vessel_mask = np.random.rand(128,128)
    mean = np.mean(vessel_mask)
    vessel_mask[vessel_mask>mean] = 1
    vessel_mask[vessel_mask!=1] = 0
    #print(vessel_mask)
    vessel_mask = np.array(vessel_mask, dtype=np.int8)
    vessel_obj.vessel_mask = vessel_mask
    vessel_obj.vessel_centerline_dist = np.random.rand(128,128)

    binary_image = vessel_mask

    feature_dict = {}
    fa = FeatureExtractor()
    pos_features = fa.extract_pos_feature(vessel_obj, binary_image, np.random.rand(128,128))
    image_features = fa.extract_radiomics_features(vessel_obj, np.random.rand(128,128))
    
    feature_dict.update(image_features)
    feature_dict.update(pos_features)
    sorted_keys = sorted(feature_dict)

    return sorted_keys


def __switch__(g0, g1):
    n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()

    if n0 >= n1:
        return g1, g0
    else:
        return g0, g1

def __build_graphs__(g):
    A = nx.adjacency_matrix(g).todense()
    edge_num = g.number_of_edges()*2 # in network graph, adjmatrix indicates undirected graph, thus, for directed graph, the number of edges are doubled
    node_num = g.number_of_nodes()
    G = np.zeros((node_num, edge_num), dtype=np.float32)
    H = np.zeros((node_num, edge_num), dtype=np.float32)
    edge_idx = 0
    for i in range(node_num):  # iterative graph adjacency matrix
        for j in range(node_num):
            if A[i, j] == 1:
                G[i, edge_idx] = 1
                H[j, edge_idx] = 1
                edge_idx += 1
    return A, G, H, edge_num


def __build_2gm_pair(g0, g1, category, sample_idx):
    n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()
    perm_mat = np.zeros((n0, n1))
    for i in range(n0):
        for j in range(n1):
            if g0.nodes()[i]['data'].vessel_class == g1.nodes()[j]['data'].vessel_class:
                perm_mat[i, j] = 1.0

    A0, G0, H0, e0 = __build_graphs__(g0)
    A1, G1, H1, e1 = __build_graphs__(g1)

    ret_dict = {'ns': [torch.tensor(x) for x in [n0, n1]],
                'es': [torch.tensor(x) for x in [e0, e1]],
                'gs': [g0, g1],
                'gt_perm_mat': perm_mat,
                'Gs': [torch.Tensor(x) for x in [G0, G1]],
                'Hs': [torch.Tensor(x) for x in [H0, H1]],
                'As': [torch.Tensor(x) for x in [A0, A1]],
                'cls': category,
                'id_list': [sample_idx[0], sample_idx[1]]}

    feat0 = np.stack([np.array(g0.nodes()[i]['data'].features) for i in range(n0)], axis=-1).T
    feat1 = np.stack([np.array(g1.nodes()[i]['data'].features) for i in range(n1)], axis=-1).T
    ret_dict['pos_features'] = [torch.Tensor(x) for x in [feat0, feat1]]

    return ret_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/media/z/data21/artery_semantic_segmentation')
    parser.add_argument('--data_path', type=str, default="gmn_vessel")
    parser.add_argument('--save_path', type=str, default="artery/data2")
    parser.add_argument('--project_path', type=str, default="ThinkMatch")
    parser.add_argument('--image_size', type=int, default=512)
    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    sample_list = _get_sample_list()

    FEATURE_DICT_ALL = {}
    fa = FeatureExtractor()

    extract = False
    if extract:
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
                for n in range(len(g.nodes)):
                    vessel_segment = g.nodes()[n]['data']
                    feature_dict = {}
                    pos_features = fa.extract_pos_feature(vessel_segment, binary_image, original_image)
                    image_features = fa.extract_radiomics_features(vessel_segment, original_image)
                    feature_dict.update(pos_features)
                    feature_dict.update(image_features)
                    sorted_keys = sorted(feature_dict)
                    features = []
                    for k in sorted_keys:
                        if k in FEATURE_DICT_ALL.keys():
                            FEATURE_DICT_ALL[k].append(feature_dict[k])
                        else:
                            FEATURE_DICT_ALL[k] = []
                            FEATURE_DICT_ALL[k].append(feature_dict[k])

                        features.append(feature_dict[k])

                    vessel_segment.features = features
                    # new
                    # vessel_segment.vessel_centerline = None
                    vessel_segment.vessel_mask = None
                    vessel_segment.vessel_centerline_dist = None

                pickle.dump(g, open(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}.pkl"), "wb"))
                cv2.imwrite(os.path.join(args.base_path, args.project_path, args.save_path,
                                        f"{selected_sample_name}_binary_image.png"), binary_image)
                cv2.imwrite(os.path.join(args.base_path, args.project_path, args.save_path,
                                        f"{selected_sample_name}.png"), original_image)

                semantic_image = cv2.imread(os.path.join(args.base_path, data_path, selected_sample_name,
                                                        f"{selected_sample_name}_step12_g_switch_unique_semantic_image.png"))
                cv2.imwrite(os.path.join(args.base_path, args.project_path, args.save_path,
                                        f"{selected_sample_name}_semantic.png"), semantic_image)

        # normalize features
        sorted_keys = sorted(FEATURE_DICT_ALL)
        for k in sorted_keys:
            print(f"k = {k}, min = {np.min(FEATURE_DICT_ALL[k])}, max = {np.max(FEATURE_DICT_ALL[k])}, len = {len(FEATURE_DICT_ALL[k])}")

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
                for n in range(len(g.nodes)):
                    vessel_segment = g.nodes()[n]['data']
                    feature_dict = {}
                    pos_features = fa.extract_pos_feature(vessel_segment, binary_image, original_image)
                    image_features = fa.extract_radiomics_features(vessel_segment, original_image)
                    feature_dict.update(pos_features)
                    feature_dict.update(image_features)
                    sorted_keys = sorted(feature_dict)
                    features = []
                    for k in sorted_keys:
                        fea = (feature_dict[k] - np.min(FEATURE_DICT_ALL[k])) / (np.max(FEATURE_DICT_ALL[k]) - np.min(FEATURE_DICT_ALL[k]))
                        features.append(fea)

                    vessel_segment.features = features
                    # new
                    # vessel_segment.vessel_centerline = None
                    vessel_segment.vessel_centerline = np.asarray(vessel_segment.vessel_centerline, np.uint8)
                    vessel_segment.vessel_mask = None
                    vessel_segment.vessel_centerline_dist = None


                pickle.dump(g, open(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}.pkl"), "wb"))
    

    # generate training samples for 2GM
    if not os.path.isdir(os.path.join(args.base_path, args.project_path, "ngm_data")):
        os.makedirs(os.path.join(args.base_path, args.project_path, "ngm_data"))

    # graph_names = glob(os.path.join(args.base_path, args.project_path, args.save_path, "*.pkl"))
    # for i in tqdm(range(len(graph_names))):
    #     for j in range(len(graph_names)):
    #         if i!=j:
    #             data_file_path_0 = graph_names[i][:graph_names[i].rfind("/")]
    #             sample_id_0 = graph_names[i][graph_names[i].rfind("/")+1:graph_names[i].rfind(".pkl")]
    #             category_0 = ARTERY_CATEGORY[0] if sample_id_0.rfind(ARTERY_CATEGORY[0]) != -1 else ARTERY_CATEGORY[1]

    #             data_file_path_1 = graph_names[j][:graph_names[j].rfind("/")]
    #             sample_id_1 = graph_names[j][graph_names[j].rfind("/")+1:graph_names[j].rfind(".pkl")]
    #             category_1 = ARTERY_CATEGORY[0] if sample_id_1.rfind(ARTERY_CATEGORY[0]) != -1 else ARTERY_CATEGORY[1]
    #             if category_0 == category_1:
    #                 image, bin, g0 = _load_graph(data_file_path_0, sample_id_0)
    #                 image, bin, g1 = _load_graph(data_file_path_1, sample_id_1)

    #                 g0, g1 = __switch__(g0, g1)
    #                 # build graph
    #                 gm_data = __build_2gm_pair(g0, g1, category_0, [sample_id_0, sample_id_1])
    #                 gm_data = collate_fn([gm_data])
    #                 pickle.dump(gm_data, 
    #                     open(os.path.join(args.base_path, args.project_path, "ngm_data", f"{sample_id_0}_{sample_id_1}.pkl"), "wb"))

