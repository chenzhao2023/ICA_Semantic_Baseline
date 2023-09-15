import os
import argparse
import cv2
import numpy as np
import SimpleITK as sitk
import pickle

from tqdm import tqdm

import six

from skimage.measure import regionprops
from radiomics import featureextractor


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
    with open("./selected_subjects.txt", "r") as f:
        for row in f.readlines():
            if row[0].isdigit():
                dataset1.append(row.strip())
            elif row[0].isalpha() and (row.strip() not in ["NJ", "TW"]):
                dataset0.append(row.strip())

    return {"NJ": dataset1, "TW": dataset0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/media/z/data2/artery_semantic_segmentation')
    parser.add_argument('--data_path', type=str, default="gmn_vessel")
    parser.add_argument('--save_path', type=str, default="data")
    parser.add_argument('--project_path', type=str, default="ml")
    parser.add_argument('--image_size', type=int, default=512)
    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    sample_list = _get_sample_list()

    FEATURE_DICT_ALL = {}
    fa = FeatureExtractor()

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

            #pickle.dump(g, open(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}.pkl"), "wb"))
            cv2.imwrite(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}_binary_image.png"), binary_image)
            cv2.imwrite(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}.png"), original_image)
            semantic_image = cv2.imread(os.path.join(args.base_path, data_path, selected_sample_name, f"{selected_sample_name}_step12_g_switch_unique_semantic_image.png"))
            cv2.imwrite(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}_semantic.png"), semantic_image)

    # normalize features

    target = open(os.path.join(args.base_path, args.project_path, args.save_path, "feature.csv"), "w")
    head_line = "patient_id,label,view"
    for k in sorted_keys:
        head_line += f",{k}"
    head_line += "\n"
    target.write(head_line)

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


                if selected_sample_name.rfind("LAO")!=-1:
                    data_row = f"{selected_sample_name},{g.nodes()[n]['data'].vessel_class},1"
                else:
                    data_row = f"{selected_sample_name},{g.nodes()[n]['data'].vessel_class},0"
                # for k in sorted_keys:
                #     data_row+= ",{}"

                features = []
                for k in sorted_keys:
                    fea = (feature_dict[k] - np.min(FEATURE_DICT_ALL[k])) / (np.max(FEATURE_DICT_ALL[k]) - np.min(FEATURE_DICT_ALL[k]))
                    data_row+= f",{fea}"
                    features.append(fea)
                data_row += "\n"
                target.write(data_row)
                vessel_segment.features = features

            # pickle.dump(g, open(os.path.join(args.base_path, args.project_path, args.save_path, f"{selected_sample_name}.pkl"), "wb")
    target.flush()
    target.close()