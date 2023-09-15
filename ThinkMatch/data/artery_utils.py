import networkx as nx
import re
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
import webcolors
from artery import Artery
from sklearn import metrics
from sklearn.model_selection import KFold


import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split as split

def estimate_nbins(y):
    """
    Break down target vartiable into bins.

    Args:
        y (pd.Series): stratification target variable.

    Returns:
        bins (array): bins' values.

    """
    if len(y)/10 <= 100:
        nbins = int(len(y)/10)
    else:
        nbins = 100
    bins = np.linspace(min(y), max(y), nbins)
    return bins


def find_neighbors_in_two_lists(keys_with_single_value, list_to_find_neighbors_in):
    '''Iterate over each item in first list to find a pair in the second list'''
    neighbors = []
    for i in keys_with_single_value:
        for j in [x for x in list_to_find_neighbors_in if x != i]:
            if i+1 == j:
                neighbors.append(i)
                neighbors.append(j)
            if i-1 == j:
                neighbors.append(i)
                neighbors.append(j)
    return neighbors

def no_neighbors_found(neighbors):
    '''Check if list is empty'''
    return not neighbors

def find_keys_without_neighbor(neighbors):
    '''Find integers in list without pair (consecutive increment of + or - 1 value) in the same list'''
    no_pair = []
    for i in neighbors:
        if i + 1 in neighbors:
            continue
        elif i - 1 in neighbors:
            continue
        else:
            no_pair.append(i)
    return no_pair

def not_need_further_execution(y_binned_count):
    '''Check if there are bins with single value counts'''
    return 1 not in y_binned_count.values()


def combine_single_valued_bins(y_binned):
    """
    Correct the assigned bins if some bins include a single value (can not be split).

    Find bins with single values and:
        - try to combine them to the nearest neighbors within these single bins
        - combine the ones that do not have neighbors among the single values with
        the rest of the bins.

    Args:
        y_binned (array): original y_binned values.

    Returns:
        y_binned (array): processed y_binned values.

    """
    # count number of records in each bin
    y_binned_count = dict(Counter(y_binned))

    if not_need_further_execution(y_binned_count):
        return y_binned

    # combine the single-valued-bins with nearest neighbors
    keys_with_single_value = []
    for key, value in y_binned_count.items():
        if value == 1:
            keys_with_single_value.append(key)

    # first look for neighbors among other sinle keys
    neighbors1 = find_neighbors_in_two_lists(keys_with_single_value, keys_with_single_value)
    if no_neighbors_found(neighbors1):
        # then look for neighbors among other available keys
        neighbors1 = find_neighbors_in_two_lists(keys_with_single_value, y_binned_count.keys())
    # now process keys for which no neighbor was found
    leftover_keys_to_find_neighbors = find_keys_without_neighbor(neighbors1)
    neighbors2 = find_neighbors_in_two_lists(leftover_keys_to_find_neighbors, y_binned_count.keys())
    neighbors = sorted(list(set(neighbors1 + neighbors2)))
    
    # split neighbors into groups for combining
    splits = int(len(neighbors)/2)
    neighbors = np.array_split(neighbors, splits)
    for group in neighbors:
        val_to_use = group[0] 
        for val in group:
            y_binned = np.where(y_binned == val, val_to_use, y_binned)
            keys_with_single_value = [x  for x in keys_with_single_value if x != val]
    # --------------------------------------------------------------------------------
    # now conbine the leftover keys_with_single_values with the rest of the bins
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    for val in keys_with_single_value:
        nearest = find_nearest([x for x in y_binned if x not in keys_with_single_value], val)
        ix_to_change = np.where(y_binned == val)[0][0]
        y_binned[ix_to_change] = nearest

    return y_binned

def validate_test_train_size_arguments(test_size, train_size):
    if not test_size and not train_size:
        test_size = 0.3
        train_size = 0.7

    if not test_size and not not train_size:
        test_size = 1-train_size

    if not not test_size and not train_size:
        train_size = 1-test_size

    if not not test_size and not not train_size:
        sum_train_test_size = test_size + train_size
        if sum_train_test_size > 1:
            diff = abs(1 - sum_train_test_size)
            train_size -= diff
    return test_size, train_size


def scsplit(*args, stratify, test_size = 0.3, train_size = 0.7, continuous = True, random_state = None):
    """
    Create stratfied splits for based on categoric or continuous column.

    For categoric target stratification raw sklearn is used, for continuous target
    stratification binning of the target variable is performed before split.

    Args:
        *args (pd.DataFrame/pd.Series): one dataframe to split into train, test
            or X, y to split into X_train, X_val, y_train, y_val.
        stratify (pd.Series): column used for stratification. Can be either a
        column inside dataset:
            train, test = scsplit(data, stratify = data['col'],...)
        or a separate pd.Series object:
            X_train, X_val, y_train, y_val = scsplit(X, y, stratify = y).
        test_size (float): test split size. Defaults to 0.3.
        train_size (float): train split size. Defaults to 0.7.
        continuous (bool): continuous or categoric target variabale. Defaults to True.
        random_state (int): random state value. Defaults to None.

    Returns:
        if a single object is passed for stratification (E.g. 'data'):
            return:
                train (pd.DataFrame): train split
                valid (pd.DataFrame): valid split
        if two objects are passed for stratification (E.g. 'X', 'y'):
            return:
                X_train (pd.DataFrame): train split independent features
                X_val (pd.DataFrame): valid split independent features
                X_train (pd.DataFrame): train split target variable
                X_train (pd.DataFrame): valid split target variable

    """
    # validate test_size/train_size arguments
    test_size, train_size = validate_test_train_size_arguments(test_size, train_size)

    if random_state:
        np.random.seed(random_state)

    if len(args) == 2:
        X = args[0]
        y = args[1]
    else:
        X = args[0].drop(stratify.name, axis = 1)
        y = args[0][stratify.name]

    # non continuous stratified split (raw sklearn)
    if not continuous:
        y = np.array(y)
        y = combine_single_valued_bins(y)
        if len(args) == 2:
            X_train, X_val, y_train, y_val = split(X, y,
                                                   stratify = y,
                                                   test_size = test_size if test_size else None,
                                                   train_size = train_size if train_size else None)
            return X_train, X_val, y_train, y_val
        else:
            temp = pd.concat([X, pd.DataFrame(y, columns = [stratify.name])], axis= 1)
            train, val = split(temp,
                                stratify = temp[stratify.name],
                                test_size = test_size if test_size else None,
                                train_size = train_size if train_size else None)
            return train, val
    # ------------------------------------------------------------------------
    # assign continuous target values into bins
    bins = estimate_nbins(y)
    y_binned = np.digitize(y, bins)
    # correct bins if necessary
    y_binned = combine_single_valued_bins(y_binned)

    # split
    if len(args) == 2:
        X_t, X_v, y_t, y_v = split(X, y_binned,
                                   stratify = y_binned,
                                   test_size = test_size if test_size else None,
                                   train_size = train_size if train_size else None)

        try:
            X_train = X.iloc[X_t.index]
            y_train = y.iloc[X_t.index]
            X_val = X.iloc[X_v.index]
            y_val = y.iloc[X_v.index]
        except IndexError as e:
            raise Exception(f'{e}\nReset index of dataframe/Series before applying scsplit')
        return X_train, X_val, y_train, y_val
    else:
        temp = pd.concat([X, pd.DataFrame(y_binned, columns = [stratify.name])], axis= 1)
        tr, te = split(temp,
                       stratify = temp[stratify.name],
                       test_size = test_size if test_size else None,
                       train_size = train_size if train_size else None)
        train = args[0].iloc[tr.index]
        test = args[0].iloc[te.index]
        return train, test




def visualize_semantic_cv2(graph: nx.Graph, original_image, semantic_mapping, save_path):
    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    positions = {}
    for node in list(graph.nodes):
        vessel_obj = graph.nodes[node]['data']
        # plot points
        # visualize artery centerlines
        vessel_class = ''.join([i for i in vessel_obj.vessel_class if not i.isdigit()])
        original_image[np.where(vessel_obj.vessel_centerline == 1)[0],
                       np.where(vessel_obj.vessel_centerline == 1)[1], :] = semantic_mapping[vessel_class][::-1]

        label_x = np.where(vessel_obj.vessel_centerline == 1)[1][
            len(np.where(vessel_obj.vessel_centerline == 1)[1]) // 2]
        label_y = np.where(vessel_obj.vessel_centerline == 1)[0][
            len(np.where(vessel_obj.vessel_centerline == 1)[0]) // 2]
        # print(f"{label_x},{label_y}")
        original_image = cv2.putText(original_image, vessel_obj.vessel_class, (label_x, label_y),
                                     fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                                     color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
        positions[vessel_obj.vessel_class] = (label_x, label_y)

    cv2.imwrite(save_path, original_image)
    return original_image, positions


def visualize_semantic_image_unique(graph: nx.Graph, original_image, semantic_mapping, save_path):
    """
    visualize pre-processed graph with unique artery labels
    """
    fig, ax = plt.subplots()
    ax.imshow(original_image, cmap="gray")
    assigned_nodes = []
    positions = {}
    #print(len(graph.nodes))
    #print(graph.nodes[0])
    for node in list(graph.nodes):
        vessel_obj = graph.nodes[node]['data']
        # plot points
        # visualize artery centerlines
        vessel_class = ''.join([i for i in vessel_obj.vessel_class if not i.isdigit()])
        plt.scatter(np.where(vessel_obj.vessel_centerline == 1)[1], np.where(vessel_obj.vessel_centerline == 1)[0],
                    color=webcolors.rgb_to_hex(semantic_mapping[vessel_class]), s=1)
        label_x = np.where(vessel_obj.vessel_centerline == 1)[1][len(np.where(vessel_obj.vessel_centerline == 1)[1]) // 2]
        label_y = np.where(vessel_obj.vessel_centerline == 1)[0][len(np.where(vessel_obj.vessel_centerline == 1)[0]) // 2]
        # print(f"{label_x},{label_y}")
        ax.annotate(vessel_obj.vessel_class, (label_x, label_y))
        positions[vessel_obj.vessel_class] = (label_x, label_y)

    plt.axis('off')
    fig.set_size_inches(original_image.shape[0] / 100, original_image.shape[0] / 100)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
    plt.margins(0, 0)

    plt.savefig(save_path)
    plt.close('all')

    return positions


def visualize_semantic_image(graph: nx.Graph, original_image, semantic_mapping, save_path, centerline=False, point=True):
    fig, ax = plt.subplots()
    ax.imshow(original_image, cmap="gray")
    assigned_labels = []
    assigned_nodes = []

    for edge in list(graph.edges):
        if len(graph.edges[edge[0], edge[1]].keys()) > 0:
            vessel_obj = graph.edges[edge[0], edge[1]]['data']
            # plot points
            if point:
                if graph.degree[edge[0]] == 1:
                    plt.scatter(graph.nodes[edge[0]]['data'].y, graph.nodes[edge[0]]['data'].x, c='g', linewidth=1, marker='+')
                elif graph.degree[edge[0]] == 2:
                    plt.scatter(graph.nodes[edge[0]]['data'].y, graph.nodes[edge[0]]['data'].x, c='b', linewidth=1, marker='o')
                else:
                    plt.scatter(graph.nodes[edge[0]]['data'].y, graph.nodes[edge[0]]['data'].x, c='r', linewidth=1, marker='*')

                if graph.degree[edge[1]] == 1:
                    plt.scatter(graph.nodes[edge[1]]['data'].y, graph.nodes[edge[1]]['data'].x, c='g', linewidth=1, marker='+')
                elif graph.degree[edge[1]] == 2:
                    plt.scatter(graph.nodes[edge[1]]['data'].y, graph.nodes[edge[1]]['data'].x, c='b', linewidth=1, marker='o')
                else:
                    plt.scatter(graph.nodes[edge[1]]['data'].y, graph.nodes[edge[1]]['data'].x, c='r', linewidth=1, marker='*')

                # annotate point index
                if edge[0] not in assigned_nodes:
                    ax.annotate(edge[0], (graph.nodes[edge[0]]['data'].y, graph.nodes[edge[0]]['data'].x), fontSize=10)
                    assigned_nodes.append(edge[0])

                if edge[1] not in assigned_nodes:
                    ax.annotate(edge[1], (graph.nodes[edge[1]]['data'].y, graph.nodes[edge[1]]['data'].x), fontSize=10)
                    assigned_nodes.append(edge[1])

            if centerline:
                # visualize artery centerlines
                if vessel_obj.vessel_class not in assigned_labels:
                    plt.scatter(np.where(vessel_obj.vessel_centerline==1)[1], np.where(vessel_obj.vessel_centerline==1)[0],
                                color=webcolors.rgb_to_hex(semantic_mapping[vessel_obj.vessel_class]),
                                label=vessel_obj.vessel_class, s=1)
                    assigned_labels.append(vessel_obj.vessel_class)
                else:
                    plt.scatter(np.where(vessel_obj.vessel_centerline==1)[1], np.where(vessel_obj.vessel_centerline==1)[0],
                                color=webcolors.rgb_to_hex(semantic_mapping[vessel_obj.vessel_class]), s=1)
            else:
                # visualize artery segments
                if vessel_obj.vessel_class not in assigned_labels:
                    plt.scatter(np.where(vessel_obj.vessel_mask==1)[1], np.where(vessel_obj.vessel_mask==1)[0],
                                color=webcolors.rgb_to_hex(semantic_mapping[vessel_obj.vessel_class]),
                                label=vessel_obj.vessel_class)
                    assigned_labels.append(vessel_obj.vessel_class)
                else:
                    plt.scatter(np.where(vessel_obj.vessel_mask==1)[1], np.where(vessel_obj.vessel_mask==1)[0],
                                color=webcolors.rgb_to_hex(semantic_mapping[vessel_obj.vessel_class]))

    plt.legend()
    plt.axis('off')
    fig.set_size_inches(original_image.shape[0] / 100, original_image.shape[0] / 100)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
    plt.margins(0, 0)

    plt.savefig(save_path)
    plt.close()


def record_match(raw_graph, output):
    total = np.sum(raw_graph['solutions'])
    matched = 0
    unmatched = 0

    mapping = {}
    for i in range(raw_graph['solutions'].shape[0]):
        if raw_graph['solutions'][i] == True and output[i] == 1:
            start_vessel_class = raw_graph['vertex_labels'][i][0]
            target_vessel_class = raw_graph['vertex_labels'][i][1]
            matched += 1
            mapping[start_vessel_class] = target_vessel_class
        elif raw_graph['solutions'][i] == False and output[i] == 1:
            # NOT MATCHED
            unmatched += 1
            start_vessel_class = raw_graph['vertex_labels'][i][0]
            target_vessel_class = raw_graph['vertex_labels'][i][1]
            mapping[start_vessel_class] = target_vessel_class

    return mapping, total, matched, unmatched


def plot_match(dataset, sample_names, output, gt, semantic_mapping, save_path, thickness=1):
    output = np.squeeze(output)
    image_0, g_0 = dataset[sample_names[0]]['image'], dataset[sample_names[0]]['g']
    image_1, g_1 = dataset[sample_names[1]]['image'], dataset[sample_names[1]]['g']

    vessel_classes0 = [g_0.nodes[x]['data'].vessel_class for x in g_0.nodes()]
    vessel_classes1 = [g_1.nodes[x]['data'].vessel_class for x in g_1.nodes()]
    n0, n1 = g_0.number_of_nodes(), g_1.number_of_nodes()
    g0_save_path = os.path.join(save_path, f"{sample_names[0]}.png")
    g1_save_path = os.path.join(save_path, f"{sample_names[1]}.png")


    color_im0, start_positions = visualize_semantic_cv2(g_0, image_0, semantic_mapping, g0_save_path)
    color_im1, end_positions = visualize_semantic_cv2(g_1, image_1, semantic_mapping, g1_save_path)

    # im0 = cv2.imread(g0_save_path, cv2.IMREAD_COLOR)
    # im1 = cv2.imread(g1_save_path, cv2.IMREAD_COLOR)
    # assert im0.shape[0] == im1.shape[1]
    im = np.zeros([color_im0.shape[0], color_im0.shape[1]+color_im1.shape[1], 3], dtype=np.uint8)
    im[:, 0:color_im0.shape[1], :] = color_im0
    im[:, color_im1.shape[1]:, :] = color_im1

    match_file_name = f"{save_path}/{sample_names[0]}<->{sample_names[1]}_match.png"

    for i in range(gt.flatten().shape[0]):
        if gt.flatten()[i] == 1 and output.flatten()[i] == 1:
            # MATCHED
            vessel_class = vessel_classes0[i//n1]
            start_pos = start_positions[vessel_class]
            end_pos = end_positions[vessel_class]
            end_pos = (end_pos[0]+color_im1.shape[0], end_pos[1])
            im = cv2.line(im, start_pos, end_pos, (0, 255, 0), thickness) # GREEN

        elif gt.flatten()[i] == 0 and output.flatten()[i] == 1:
            # NOT MATCHED
            vessel_class_left = vessel_classes0[i//n1]
            vessel_class_right = vessel_classes1[i//n1]
            start_pos = start_positions[vessel_class_left]
            end_pos = end_positions[vessel_class_right]
            end_pos = (end_pos[0] + color_im1.shape[0], end_pos[1])
            im = cv2.line(im, start_pos, end_pos, (0, 0, 255), thickness) # RED

    cv2.imwrite(match_file_name, im)


def split_dataset_category(data_path, ratio, seed):
    from sklearn.model_selection import train_test_split
    df_view_angles = pd.read_csv(data_path)
    train, test = train_test_split(df_view_angles, test_size=ratio, stratify=df_view_angles['category'], random_state=seed)
    training_samples = train['id'].values
    test_samples = test['id'].values
    return training_samples, test_samples


def get_split_deterministic(all_keys, fold=0, num_splits=5, random_state=12345):
    """
    Splits a list of patient identifiers (or numbers) into num_splits folds and returns the split for fold fold.
    :param all_keys:
    :param fold:
    :param num_splits:
    :param random_state:
    :return:
    """
    all_keys_sorted = np.sort(list(all_keys))
    splits = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(splits.split(all_keys_sorted)):
        if i == fold:
            train_keys = np.array(all_keys_sorted)[train_idx]
            test_keys = np.array(all_keys_sorted)[test_idx]
            break
    return train_keys, test_keys


def post_processing_voting(df, dataset):
    results_df = pd.DataFrame(columns=df.columns)
    for test_sample in np.unique(df['test_sample']):
        g = dataset[test_sample]['g']
        artery_branches = [g.nodes()[k]['data'].vessel_class for k in g.nodes()]
        sub_df = df[df['test_sample']==test_sample]
        data_row = {"test_sample": test_sample, "template_sample": "",
                    "category": np.unique(sub_df['category'])[0],
                    "n": len(artery_branches)}
        matched = 0
        for artery_branch in artery_branches:
            # print(f"[x] sample {test_sample}, branch {artery_branch}")
            sub_series = sub_df[artery_branch].dropna()
            if sub_series.shape[0] > 0:
                unique, counts = np.unique(sub_series, return_counts=True)
                label = unique[np.argmax(counts)]
                data_row[artery_branch] = label
                if artery_branch == label:
                    matched += 1

        data_row['matched'] = matched
        results_df = results_df.append(data_row, ignore_index=True)

    return results_df


def evaluate_main_branches(df, dataset, print_result=True):
    columns = []
    columns.extend(["test_sample", "category"])

    branch_mapping = {}
    for sub_branch_name in Artery.SUB_BRANCH_CATEGORY:
        for main_branch_name in Artery.MAIN_BRANCH_CATEGORY:
            if sub_branch_name.startswith(main_branch_name):
                branch_mapping[sub_branch_name] = main_branch_name
    # print(branch_mapping)

    for main_branch_name in Artery.MAIN_BRANCH_CATEGORY:
        columns.append(f"{main_branch_name}_matched")
        columns.append(f"{main_branch_name}_unmatched")

    result_df = pd.DataFrame(columns=columns)

    for test_sample in np.unique(df['test_sample']):
        g = dataset[test_sample]['g']
        sub_artery_branches = [g.nodes()[k]['data'].vessel_class for k in g.nodes()]
        sub_df = df[df['test_sample'] == test_sample]

        # init data row
        data_row = {"test_sample": test_sample, "category": np.unique(sub_df['category'])[0]}

        for branch_name in Artery.MAIN_BRANCH_CATEGORY:
            data_row[f'{branch_name}_matched'] = 0
            data_row[f'{branch_name}_unmatched'] = 0

        # assign value
        for sub_branch_name in sub_artery_branches:
            # print(f"[x] sample {test_sample}, branch {sub_branch_name}")
            sub_series = sub_df[sub_branch_name].dropna()
            if sub_series.shape[0] > 0:
                unique, counts = np.unique(sub_series, return_counts=True)
                label = unique[np.argmax(counts)]
                # data_row[artery_branch] = label
                mapped_main_branch = branch_mapping[sub_branch_name] # map sub label to main branch label defined in Artery.MAIN_BRANCH_CATEGORY
                if sub_branch_name == label:
                    data_row[f'{mapped_main_branch}_matched'] = data_row[f'{mapped_main_branch}_matched']+1
                else:
                    data_row[f'{mapped_main_branch}_unmatched'] = data_row[f'{mapped_main_branch}_unmatched'] + 1

        result_df = result_df.append(data_row, ignore_index=True)

    for main_branch_name in Artery.MAIN_BRANCH_CATEGORY:
        result_df[f"{main_branch_name}_total"] = result_df[f"{main_branch_name}_matched"]+result_df[f"{main_branch_name}_unmatched"]

    if print_result:
        for main_branch_name in Artery.MAIN_BRANCH_CATEGORY:
            if result_df[f'{main_branch_name}_total'].sum() == 0:
                print("{}, total = {}, matched = {}, acc = {}".format(main_branch_name,
                                                                      result_df[f'{main_branch_name}_total'].sum(),
                                                                      result_df[f'{main_branch_name}_matched'].sum(), 0.))
            else:
                print("{}, total = {}, matched = {}, acc = {}".format(main_branch_name,
                                        result_df[f'{main_branch_name}_total'].sum(),
                                        result_df[f'{main_branch_name}_matched'].sum(),
                                        result_df[f'{main_branch_name}_matched'].sum()/result_df[f'{main_branch_name}_total'].sum()))
    return result_df


def evaluate_main_branches_sklearn(df, dataset):

    def convert_name_to_label(artery_names):
        lbls = []
        for artery_name in artery_names:
            for i in range(len(Artery.MAIN_BRANCH_CATEGORY)):
                if artery_name == Artery.MAIN_BRANCH_CATEGORY[i]:
                    lbls.append(i)
        return lbls

    branch_mapping = {}
    for sub_branch_name in Artery.SUB_BRANCH_CATEGORY:
        for main_branch_name in Artery.MAIN_BRANCH_CATEGORY:
            if sub_branch_name.startswith(main_branch_name):
                branch_mapping[sub_branch_name] = main_branch_name

    gts = []
    preds = []

    for test_sample in np.unique(df['test_sample']):
        g = dataset[test_sample]['g']
        sub_artery_branches = [g.nodes()[k]['data'].vessel_class for k in g.nodes()]
        # all_arteries_gt.extend(sub_artery_branches)

        sub_df = df[df['test_sample'] == test_sample]

        for sub_branch_name in sub_artery_branches:
            # print(f"[x] sample {test_sample}, branch {sub_branch_name}")
            sub_series = sub_df[sub_branch_name].dropna()
            if sub_series.shape[0] > 0:
                unique, counts = np.unique(sub_series, return_counts=True)
                label = unique[np.argmax(counts)]
                # data_row[artery_branch] = label
                mapped_main_branch_gt = branch_mapping[sub_branch_name] # map sub label to main branch label defined in Artery.MAIN_BRANCH_CATEGORY
                mapped_main_branch_pred = branch_mapping[label]
                gts.append(mapped_main_branch_gt)
                preds.append(mapped_main_branch_pred)
    

    gts = convert_name_to_label(gts)
    preds = convert_name_to_label(preds)

    cm = metrics.confusion_matrix(gts, preds)
    acc = metrics.accuracy_score(gts, preds)
    precision = metrics.precision_score(gts, preds, average="weighted")
    recall = metrics.recall_score(gts, preds, average="weighted")
    f1_score = metrics.f1_score(gts, preds, average="weighted")

    try:
        clf_report = metrics.classification_report(gts, preds, target_names=Artery.MAIN_BRANCH_CATEGORY, output_dict=True)
    except:
        clf_report = metrics.classification_report([0,1,2,3,4], [0,0,0,0,0], target_names=Artery.MAIN_BRANCH_CATEGORY, output_dict=True)
    return cm, clf_report, acc, precision, recall, f1_score