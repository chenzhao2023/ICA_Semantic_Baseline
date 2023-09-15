import numpy as np
import os
import copy
import networkx as nx
import pickle
import cv2

from glob import glob
from tqdm import tqdm


#ARTERY_CATEGORY = ["LAO", "RAO"]
#DATA_FILE_PATH = "/media/z/data21/artery_semantic_segmentation/ThinkMatch/artery/data2"

# AAAI settings
ARTERY_CATEGORY = ["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"]
DATA_FILE_PATH = "/media/z/data2/artery_semantic_segmentation/hnn_hm_june/data/artery_with_feature"

SEMANTIC_MAPPING = {"OTHER": [255, 0, 0], 
                    "LAD": [255, 255, 0], "LCX": [102, 255, 102], "LMA": [0, 102, 255], \
                    "D": [255, 0, 255], "OM": [102, 255, 255],
                    "SEP": [102, 0, 255]}

SUB_BRANCH_CATEGORY = ["LMA", "LAD1", "LAD2", "LAD3", "LCX1", "LCX2", "LCX3", "D1", "D2", "OM1", "OM2", "OM3"]
MAIN_BRANCH_CATEGORY = ["LMA", "LAD", "LCX", "D", "OM"]

def _get_sample_list(data_file_path, category):
    pkl_file_paths = glob(os.path.join(data_file_path, "*.pkl"))
    samples = []
    for pkl_file_path in pkl_file_paths:
        if pkl_file_path.rfind("tree") == -1:
            sample_name = pkl_file_path[pkl_file_path.rfind("/")+1: pkl_file_path.rfind(".pkl")]
            if category == "":
                samples.append(sample_name)
            else:
                if sample_name.rfind(category) !=-1:
                    samples.append(sample_name)
    return samples


def _load_graph(data_file_path, sample_id):
    image = cv2.imread(os.path.join(data_file_path, f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
    binary_image = cv2.imread(os.path.join(data_file_path, f"{sample_id}_binary_image.png"), cv2.IMREAD_GRAYSCALE)
    pkl_file_path = os.path.join(data_file_path, f"{sample_id}.pkl")
    g = pickle.load(open(pkl_file_path, 'rb'))
    return image, binary_image, g


def _load_graph_in_mem(data_file_path, sample_id, dataset=None):
    if dataset is None:
        # load all samples from hd
        sample_list = _get_sample_list(data_file_path, "")
        data = {}
        # this will load all data into memory, for 300 subject, costs about 10GB RAM
        print("Artery._load_graph_in_mem, loading all data")
        for sample_name in tqdm(sample_list):
            image, bin, g = _load_graph(data_file_path, sample_name)
            data[sample_name] = {"image": image, "binary_image": bin, "g": g}
        return data, sample_list
    else:
        return dataset[sample_id]['image'], dataset[sample_id]['binary_image'], dataset[sample_id]['g']


def _build_assign_graph(g0: nx.Graph, g1: nx.Graph, tails0, heads0, tails1, heads1):
    num_nodes0 = g0.number_of_nodes()
    num_nodes1 = g1.number_of_nodes()
    num_edges0 = g0.number_of_edges()
    num_edges1 = g1.number_of_edges()
    num_matches = num_nodes0*num_nodes1

    gidx1 = np.zeros(num_matches, np.int)
    gidx2 = np.zeros(num_matches, np.int)
    for i in range(num_matches):
        gidx1[i] = i / num_nodes1
        gidx2[i] = i % num_nodes1

    feature_per_node = len(g0.nodes()[0]['data'].features)
    node_feaLen = feature_per_node*2
    edge_feaLen = feature_per_node*4
    num_assGraph_nodes = num_matches
    num_assGraph_edges = num_edges0 * num_edges1
    senders = np.zeros(num_assGraph_edges, np.int)
    receivers = np.zeros(num_assGraph_edges, np.int)
    node_features = np.zeros((num_assGraph_nodes, node_feaLen), np.float)
    edge_features = np.zeros((num_assGraph_edges, edge_feaLen), np.float)

    vertex_labels = []
    # assign graph vertex features
    for i in range(num_matches):
        fea_node0 = np.array(g0.nodes()[gidx1[i]]['data'].features)
        fea_node1 = np.array(g1.nodes()[gidx2[i]]['data'].features)
        node_features[i] = np.hstack((fea_node0, fea_node1))
        vertex_labels.append((g0.nodes()[gidx1[i]]['data'].vessel_class, g1.nodes()[gidx2[i]]['data'].vessel_class))

    # assign graph edge features
    idx = 0
    for i in range(num_edges0):
        fea_tail0 = np.array(g0.nodes()[tails0[i]]['data'].features)
        fea_head0 = np.array(g0.nodes()[heads0[i]]['data'].features)
        for j in range(num_edges1):
            fea_tail1 = np.array(g1.nodes()[tails1[j]]['data'].features)
            fea_head1 = np.array(g1.nodes()[heads1[j]]['data'].features)
            senders[idx] = tails0[i] * num_nodes1 + tails1[j]
            receivers[idx] = heads0[i] * num_nodes1 + heads1[j]
            edge_features[idx] = np.hstack((fea_tail0, fea_head0, fea_tail1, fea_head1))

            idx = idx+1

    assignGraph = {"gidx1": gidx1,
                   "gidx2": gidx2,
                   "node_features": node_features,
                   "senders": senders,
                   "receivers": receivers,
                   "edge_features": edge_features,
                   "vertex_labels": vertex_labels}

    return assignGraph


def _switch(data_file_path, sample_list, sample_idx):
    _, _, g0 = _load_graph(data_file_path, sample_list[sample_idx[0]])
    _, _, g1 = _load_graph(data_file_path, sample_list[sample_idx[1]])

    n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()

    if n0 >= n1:
        return [sample_idx[1], sample_idx[0]]
    else:
        return sample_idx


def gen_random_graph_in_mem_test(sample_ids, dataset0, dataset1, category):
    assert len(sample_ids) == 2
    image0, binary_image0, g0 = dataset0[sample_ids[0]]['image'], dataset0[sample_ids[0]]['binary_image'], dataset0[sample_ids[0]]['g']
    image1, binary_image1, g1 = dataset1[sample_ids[1]]['image'], dataset1[sample_ids[1]]['binary_image'], dataset1[sample_ids[1]]['g']

    tails0, heads0 = np.nonzero(nx.adjacency_matrix(g0).todense())[0], np.nonzero(nx.adjacency_matrix(g0).todense())[1]
    tails1, heads1 = np.nonzero(nx.adjacency_matrix(g1).todense())[0], np.nonzero(nx.adjacency_matrix(g1).todense())[1]

    # record ground-truth matches
    n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()
    gX = np.zeros((n0, n1))
    for i in range(n0):
        for j in range(n1):
            if g0.nodes()[i]['data'].vessel_class == g1.nodes()[j]['data'].vessel_class:
                gX[i, j] = 1.0

    # build assign_graph
    assignGraph = _build_assign_graph(g0, g1, tails0, heads0, tails1, heads1)

    gidx1 = assignGraph["gidx1"]
    gidx2 = assignGraph["gidx2"]
    solutions = np.zeros(len(gidx1), np.bool)
    for i in range(len(gidx1)):
        if gX[gidx1[i]][gidx2[i]]:
            solutions[i] = True
    assignGraph["solutions"] = solutions
    assignGraph['sample_names'] = sample_ids

    image = {"category": category,
             "graph0": {"image": image0, "binary_image": binary_image0, "g": g0},
             "graph1": {"image": image1, "binary_image": binary_image1, "g": g1}}

    return assignGraph, image


def _gen_random_graph_in_mem(rand, category_id, dataset):

    def __switch__(sample_idx):
        sample_name0, sample_name1 = sample_list[sample_idx[0]], sample_list[sample_idx[1]]
        _, _, g0 = dataset[sample_name0]['image'], dataset[sample_name0]['binary_image'], dataset[sample_name0]['g']
        _, _, g1 = dataset[sample_name1]['image'], dataset[sample_name1]['binary_image'], dataset[sample_name1]['g']

        n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()

        if n0 >= n1:
            return [sample_idx[1], sample_idx[0]]
        else:
            return sample_idx

    all_sample_list = list(dataset.keys())
    sample_list = []
    for sample_name in all_sample_list:
        if category_id == -1:
            sample_list.append(sample_name)
        else:
            if sample_name.rfind(ARTERY_CATEGORY[category_id]) != -1:
                sample_list.append(sample_name)

    sample_idx = rand.randint(0, len(sample_list), size=2)
    sample_idx = __switch__(sample_idx)

    image0, binary_image0, g0 = \
        dataset[sample_list[sample_idx[0]]]['image'], dataset[sample_list[sample_idx[0]]]['binary_image'], dataset[sample_list[sample_idx[0]]]['g']
    image1, binary_image1, g1 = \
        dataset[sample_list[sample_idx[1]]]['image'], dataset[sample_list[sample_idx[1]]]['binary_image'], dataset[sample_list[sample_idx[1]]]['g']

    tails0, heads0 = np.nonzero(nx.adjacency_matrix(g0).todense())[0], np.nonzero(nx.adjacency_matrix(g0).todense())[1]
    tails1, heads1 = np.nonzero(nx.adjacency_matrix(g1).todense())[0], np.nonzero(nx.adjacency_matrix(g1).todense())[1]

    # record ground-truth matches
    n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()
    gX = np.zeros((n0, n1))
    for i in range(n0):
        for j in range(n1):
            if g0.nodes()[i]['data'].vessel_class == g1.nodes()[j]['data'].vessel_class:
                gX[i, j] = 1.0

    # build assign_graph
    assignGraph = _build_assign_graph(g0, g1, tails0, heads0, tails1, heads1)

    gidx1 = assignGraph["gidx1"]
    gidx2 = assignGraph["gidx2"]
    solutions = np.zeros(len(gidx1), np.bool)
    for i in range(len(gidx1)):
        if gX[gidx1[i]][gidx2[i]]:
            solutions[i] = True
    assignGraph["solutions"] = solutions
    assignGraph['sample_names'] = [sample_list[sample_idx[0]], sample_list[sample_idx[1]]]

    image = {"category": ARTERY_CATEGORY[category_id],
             "graph0": {"image": image0, "binary_image": binary_image0, "g": g0},
             "graph1": {"image": image1, "binary_image": binary_image1, "g": g1}}

    return assignGraph, image


def _gen_random_graph(rand, category_id, data_file_path):
    sample_list = _get_sample_list(data_file_path, ARTERY_CATEGORY[category_id])
    sample_idx = rand.randint(0, len(sample_list), size=2)
    sample_idx = _switch(data_file_path, sample_list, sample_idx)

    # print(f"_gen_random_graph {sample_list[sample_idx[0]]}, {sample_list[sample_idx[1]]}")
    image0, binary_image0, g0 = _load_graph(data_file_path, sample_list[sample_idx[0]])
    image1, binary_image1, g1 = _load_graph(data_file_path, sample_list[sample_idx[1]])

    tails0, heads0 = np.nonzero(nx.adjacency_matrix(g0).todense())[0], np.nonzero(nx.adjacency_matrix(g0).todense())[1]
    tails1, heads1 = np.nonzero(nx.adjacency_matrix(g1).todense())[0], np.nonzero(nx.adjacency_matrix(g1).todense())[1]

    # record ground-truth matches
    n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()
    gX = np.zeros((n0, n1))
    for i in range(n0):
        for j in range(n1):
            if g0.nodes()[i]['data'].vessel_class == g1.nodes()[j]['data'].vessel_class:
                gX[i, j] = 1.0

    # build assign_graph
    assignGraph = _build_assign_graph(g0, g1, tails0, heads0, tails1, heads1)

    gidx1 = assignGraph["gidx1"]
    gidx2 = assignGraph["gidx2"]
    solutions = np.zeros(len(gidx1), np.bool)
    for i in range(len(gidx1)):
        if gX[gidx1[i]][gidx2[i]]:
            solutions[i] = True
    assignGraph["solutions"] = solutions

    image = {"category": ARTERY_CATEGORY[category_id],
             "graph0": {"image": image0, "binary_image":binary_image0, "g":g0},
             "graph1": {"image": image1, "binary_image":binary_image1, "g":g1}}

    return assignGraph, image


def gen_random_graph_Artery_in_men(rand, num_examples, dataset, category_id=-1):
    graphs = []
    images = []
    for _ in range(num_examples):
        if category_id < 0:
            cid = rand.randint(0, len(ARTERY_CATEGORY))
        else:
            cid = category_id

        graph, image = _gen_random_graph_in_mem(rand, cid, dataset)

        graphs.append(graph)
        images.append(image)

    return graphs, images


def gen_random_graphs_Artery(rand, num_examples, num_inner_min_max, num_outlier_min_max, category_id=-1, data_file_path=DATA_FILE_PATH):
    graphs = []
    images = []

    for _ in range(num_examples):
        if category_id < 0:
            cid = rand.randint(0, len(ARTERY_CATEGORY))
        else:
            cid = category_id

        graph, image = _gen_random_graph(rand, cid, data_file_path)

        graphs.append(graph)
        images.append(image)

    return graphs, images


def get_category(sample_name):
    for category in ARTERY_CATEGORY:
        if sample_name.rfind(category) != -1:
            return category
    return ""


def __trim_graph__(g, prob, rand):
    removed_nodes_idx = []
    removed_nodes = []

    g = copy.deepcopy(g)
    for node in g.nodes():
        if g.nodes()[node]['data'].node1.degree == 1 or g.nodes()[node]['data'].node2.degree == 1:
            # removeable 
            if rand.rand() < prob: # generate random value according to pre-defined seed
                removed_nodes.append(g.nodes()[node]['data'].vessel_class)
                removed_nodes_idx.append(node)
    
    for n in removed_nodes_idx:
        g.remove_node(n)
    # re assign node index
    mapping = {old_label:new_label for new_label, old_label in enumerate(g.nodes())}
    g = nx.relabel_nodes(g, mapping)
    return g, removed_nodes