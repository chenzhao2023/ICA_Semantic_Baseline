import networkx as nx
import numpy as np
import cv2
import os
import argparse
import pickle
import matplotlib.pyplot as plt

from artery.Artery import SEMANTIC_MAPPING
from tqdm import tqdm
from artery.data_augmentation import *



def _get_sample_list():
    dataset0 = []
    dataset1 = []
    with open("selected_subjects.txt", "r") as f:
        for row in f.readlines():
            if row[0].isdigit():
                dataset1.append(row.strip())
            elif row[0].isalpha() and (row.strip() not in ["NJ", "TW"]):
                dataset0.append(row.strip())

    return {"NJ": dataset1, "TW": dataset0}



class ArteryAugmentationFlow:
    def __init__(self, g, original_image, params):
        self.g = g
        self.original_image = original_image
        self.params = params
    
    def augment(self):
        # 1. blur image
        if np.random.rand() < self.params.blur:
            self.original_image = gaussian_blur(self.original_image)
            add_vessel_image(self.g, original_image)

        # 2. trim graph
        if np.random.rand() < self.params.trim:
            self.g, removed_nodes = trim_graph(self.g, self.params.trim)
            self.g = merge_graph(self.g, removed_nodes)
            print(f"graph trim, removed nodes = {removed_nodes}")

        # 3. rotate graph
        for n in self.g.nodes:
            if np.random.rand() < self.params.rotate:
                angle = np.random.randint(args.rotate_angle) if np.random.randint(2) else -np.random.randint(args.rotate_angle)
                adjusted_angle = check_rotate(self.g, self.g.nodes[n]['data'].vessel_class, angle)
                print(f"rotate artery: {self.g.nodes[n]['data'].vessel_class}, random = {angle}, adjusted = {adjusted_angle}")
                rotate_artery(self.g, self.g.nodes[n]['data'].vessel_class, adjusted_angle)

        # 4. resize
        if np.random.rand() < self.params.resize:
            factor_offset = np.random.uniform(0, self.params.resize_factor)  
            factor = 1.0+factor_offset if np.random.randint(2) else 1.0-factor_offset
            adjusted_factor = check_resize(self.g, "LMA", factor)
            print(f"raw factor = {factor}, adjusted factor = {adjusted_factor}")
            resize_branch(self.g, "LMA", adjusted_factor)

        return self.g, self.original_image

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/media/z/data21/artery_semantic_segmentation')
    parser.add_argument('--data_path', type=str, default="gmn_vessel")
    parser.add_argument('--save_path', type=str, default="artery/data_augment")
    parser.add_argument('--project_path', type=str, default="ThinkMatch")
    parser.add_argument('--image_size', type=int, default=512)

    # DATA AUGMENTATION PARAMS
    parser.add_argument('--blur', type=float, default=0.2)
    parser.add_argument('--rotate', type=float, default=0.5)
    parser.add_argument('--rotate_angle', type=float, default=20)
    parser.add_argument('--resize', type=float, default=0.4)
    parser.add_argument('--resize_factor', type=float, default=0.1)
    parser.add_argument('--trim', type=float, default=0.3)

    parser.add_argument('--n_augment', type=int, default=64)
    parser.add_argument('--max_try', type=int, default=5)

    args = parser.parse_args()

    # load graph function
    # base_path = args.base_path
    # data_path = args.data_path
    # save_path = args.save_path
    # image_size = args.image_size
    # project_path = args.project_path


    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    sample_list = _get_sample_list()


    for dataset_name, data_path in zip(["TW", "NJ"],
                                    [os.path.join(args.base_path, "gmn_vessel/data/data_tw_semantic/processed"),
                                        os.path.join(args.base_path, "gmn_vessel/data/data_nj_semantic/processed")]):
        selected_sample_names = sample_list[dataset_name]
        for selected_sample_name in tqdm(selected_sample_names):
            for idx in tqdm(range(args.n_augment)):
                if os.path.isfile(f"{args.save_path}/{selected_sample_name}_{idx}.pkl"):
                    continue

                try:
                    print(f"[x] processing {selected_sample_name} ---- ")

                    binary_image = cv2.imread(os.path.join(args.base_path, data_path, selected_sample_name, f"{selected_sample_name}_binary_image.png"), cv2.IMREAD_GRAYSCALE)
                    if binary_image.shape[0] != args.image_size:
                        binary_image = cv2.resize(binary_image, (args.image_size, args.image_size))

                    original_image = cv2.imread(os.path.join(args.base_path, data_path, selected_sample_name, f"{selected_sample_name}.png"), cv2.IMREAD_GRAYSCALE)
                    if original_image.shape[0] != args.image_size:
                        original_image = cv2.resize(original_image, (args.image_size, args.image_size))
                
                    pkl_file_path = os.path.join(args.base_path, data_path, selected_sample_name, f"{selected_sample_name}_g_switch_unique.pkl")
                    g_switch= pickle.load(open(pkl_file_path, 'rb'))

                    # show original image
                    g_switch = add_vessel_image(g_switch, original_image)
                    assign_new_obj_to_node(g_switch)

                    # data augmentataion
                    # semantic_img = visualize_semantic_cv2(g_switch, original_image, SEMANTIC_MAPPING)
                    # plt.imshow(semantic_img)
                    # plt.show()

                    augmentor = ArteryAugmentationFlow(g_switch, original_image, args)
                    g_aug, image = augmentor.augment()
                    # semantic_img = visualize_semantic_cv2(g_aug, image, SEMANTIC_MAPPING)
                    pickle.dump(g_aug, open(f"{args.save_path}/{selected_sample_name}_{idx}.pkl", "wb"))
                    artery_img = visualize_semantic_cv2(g_aug, np.zeros_like(image), SEMANTIC_MAPPING)
                    cv2.imwrite(f"{args.save_path}/{selected_sample_name}_{idx}.png", artery_img)

                    # labeldict = {}
                    # for node_index in g_aug.nodes():
                    #     labeldict[node_index] = f"{g_aug.nodes[node_index]['data'].vessel_class}"
                    # nx.draw(g_aug, labels=labeldict, with_labels=True, node_color="#3498DB", width=2.0, font_size=12, font_color="w", node_size=900)

                    # plt.savefig(f"{args.save_path}/{selected_sample_name}_{idx}_g.png")
                    # plt.close()

                except AssertionError:
                    print(f"[!] {selected_sample_name}, idx = {idx}, cannot augment")
                    continue
                except ValueError:
                    print(f"[!] {selected_sample_name}, idx = {idx}, cannot find overlapped nodes")
                    continue
                except AttributeError:
                    print(f"[!] {selected_sample_name}, idx = {idx}, attribute error")
                    continue

    # base_path = "/media/z/data21/artery_semantic_segmentation"
    # data_path = "gmn_vessel/data/data_tw_semantic/processed"
    # image_size = 512
    # project_path = "ThinkMatch"

    # selected_sample_name = "B01_LCA_LAO"

    # binary_image = cv2.imread(f"{base_path}/{data_path}/{selected_sample_name}/{selected_sample_name}_binary_image.png", cv2.IMREAD_GRAYSCALE)
    # original_image = cv2.imread(f"{base_path}/{data_path}/{selected_sample_name}/{selected_sample_name}.png",  cv2.IMREAD_GRAYSCALE)

    # pkl_file_path = f"{base_path}/{data_path}/{selected_sample_name}/{selected_sample_name}_g_switch_unique.pkl"
    # g_switch= pickle.load(open(pkl_file_path, 'rb'))

    # pkl_file_path = f"{base_path}/{data_path}/{selected_sample_name}/{selected_sample_name}.pkl"
    # g = pickle.load(open(pkl_file_path, 'rb'))


    # # show original image
    # g_switch = add_vessel_image(g_switch, original_image)
    # assign_new_obj_to_node(g_switch)

    # adjusted_angle = check_rotate(g_switch, "LMA", 10)
    # print(adjusted_angle)
    # rotate_artery(g_switch, "LMA", adjusted_angle)