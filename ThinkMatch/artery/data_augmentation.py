import networkx as nx
import numpy as np
import cv2
import os
import pickle
import copy
import matplotlib.pyplot as plt

from skimage import measure
from core.utils.module import Node, VesselSegment
from scipy import ndimage
from artery.Artery import SEMANTIC_MAPPING
from PIL import Image


parent_table = {"LAD1": "LMA", "LAD2": "LAD1", "LAD3": "LAD2", 
                "D1": "LAD1", "D2": "LAD2",
                "LCX1": "LMA", "LCX2": "LCX1", "LCX3": "LCX2",
                "OM1": "LCX1", "OM2": "LCX2", 
                "LMA": None}

children_table = {"LMA":  ["LAD1", "LAD2", "LAD3", "LCX1", "LCX2", "LCX3", "D1", "D2", "OM1", "OM2"],
                  "LAD1": ["LAD2", "LAD3", "D1", "D2"],
                  "LCX1": ["LCX2", "LCX3", "OM1", "OM2"],
                  "LAD2": ["LAD3", "D2"], 
                  "LCX2": ["LCX3", "OM2"],
                  "LAD3": [], "LCX3": [], "D1": [], "D2": [], "OM1": [], "OM2": []}

#################### visualization ######################



def visualize_semantic_cv2(graph: nx.Graph, original_image, semantic_mapping):
    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    positions = {}
    for node in list(graph.nodes):
        vessel_obj = graph.nodes[node]['data']
        # plot points
        # visualize artery centerlines
        vessel_class = ''.join([i for i in vessel_obj.vessel_class if not i.isdigit()])
        original_image[np.where(vessel_obj.vessel_centerline == 1)[0],
                       np.where(vessel_obj.vessel_centerline == 1)[1], :] = semantic_mapping[vessel_class]
        
        label_x = np.where(vessel_obj.vessel_centerline == 1)[1][
            len(np.where(vessel_obj.vessel_centerline == 1)[1]) // 2]
        label_y = np.where(vessel_obj.vessel_centerline == 1)[0][
            len(np.where(vessel_obj.vessel_centerline == 1)[0]) // 2]
        # print(f"{label_x},{label_y}")
        original_image = cv2.putText(original_image, vessel_obj.vessel_class, (label_x, label_y),
                                     fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                                     color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
        
        original_image = cv2.putText(original_image, "x", (vessel_obj.node1.y, vessel_obj.node1.x),
                                     fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                                     color=semantic_mapping[vessel_class], thickness=1, lineType=cv2.LINE_AA)
        
        original_image = cv2.putText(original_image, "x", (vessel_obj.node2.y, vessel_obj.node2.x),
                                     fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                                     color=semantic_mapping[vessel_class], thickness=1, lineType=cv2.LINE_AA)
        
        positions[vessel_obj.vessel_class] = (label_x, label_y)

#     cv2.imwrite(save_path, original_image)
    return original_image


#################### Image Blur #########################
def gaussian_blur(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


###################### Pre-processing ####################
def add_vessel_image(g, original_image):
    for node_idx in g.nodes:
        g.nodes[node_idx]['data'].vessel_image = original_image * g.nodes[node_idx]['data'].vessel_mask
    return g


def assign_new_obj_to_node(g):
    for node_idx in g.nodes:
        g.nodes[node_idx]['data'].node1 = Node(g.nodes[node_idx]['data'].node1.degree, 
                                               g.nodes[node_idx]['data'].node1.x, 
                                               g.nodes[node_idx]['data'].node1.y)
        g.nodes[node_idx]['data'].node2 = Node(g.nodes[node_idx]['data'].node2.degree, 
                                               g.nodes[node_idx]['data'].node2.x, 
                                               g.nodes[node_idx]['data'].node2.y) 

############################################################

def _find_branch_data_(g, branch_name):
    for n in g.nodes:
        if g.nodes[n]['data'].vessel_class == branch_name:
            return g.nodes[n]['data'], n
    return None, -1 # didn't find

def _find_pivot_point_(g, branch_name):
    current_segment, _ = _find_branch_data_(g, branch_name)
    if branch_name == "LMA":
        if current_segment.node1.degree == 1:
            return (current_segment.node1.y, current_segment.node1.x), 1
        else:
            return (current_segment.node2.y, current_segment.node2.x), 2
        
    parent_segment, _ = _find_branch_data_(g, parent_table[branch_name])
    assert current_segment is not None
    assert parent_segment is not None
    if parent_segment.node1.x == current_segment.node1.x and parent_segment.node1.y == current_segment.node1.y:
        return (current_segment.node1.y, current_segment.node1.x), 1
    elif parent_segment.node2.x == current_segment.node1.x and parent_segment.node2.y == current_segment.node1.y:
        return (current_segment.node1.y, current_segment.node1.x), 1
    elif parent_segment.node1.x == current_segment.node2.x and parent_segment.node1.y == current_segment.node2.y:
        return (current_segment.node2.y, current_segment.node2.x), 2
    elif parent_segment.node2.x == current_segment.node2.x and parent_segment.node2.y == current_segment.node2.y:
        return (current_segment.node2.y, current_segment.node2.x), 2
    else:
        raise ValueError("cannot find an overlapped nodes between these two artery segments")
    
#################### Rotate ##########################
def _rotate_image_(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]


def _check_rotate_centerline(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    all_centerline = imgR.sum()
    crop_centerline = imgR[padY[0] : -padY[1], padX[0] : -padX[1]].sum()
    return True if all_centerline == crop_centerline else False


def _check_rotate_point_(point, angle, pivot, img_size):
    padX = [img_size - pivot[0], pivot[0]]
    padY = [img_size - pivot[1], pivot[1]]
    imgT = np.zeros([img_size, img_size], dtype=np.uint8)
    imgT[point] = 1
    assert imgT.sum() == 1
    imgP = np.pad(imgT, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    imgR = imgR[padY[0]: -padY[1], padX[0] : -padX[1]]
    if imgR.sum() == 1:
        # print(f"check_ratate_point, {point} along {pivot} with {angle} is valid")
        return True
    else:
        # print(f"check_ratate_point, {point} along {pivot} with {angle} is invalid")
        return False


def _check_end_point_rotate_(vessel_segment, angle, pivot, step=1):
    flag = True
    current_angle = angle
    while flag:
        if _check_rotate_point_((vessel_segment.node1.x, vessel_segment.node1.y), current_angle, pivot, 512) and \
            _check_rotate_point_((vessel_segment.node2.x, vessel_segment.node2.y), current_angle, pivot, 512):
            flag = False
            current_angle = current_angle-step if angle > 0 else current_angle+step
            current_angle = max(0, current_angle) if angle > 0 else min(0, current_angle)
        else:
            current_angle = current_angle-step if angle > 0 else current_angle+step
            current_angle = max(0, current_angle) if angle > 0 else min(0, current_angle)
    return current_angle


def _check_image_rotate_(img, angle, pivot, step=1):
    """
    check if centerline and points fall withing the image bounds
    """
    flag = True
    current_angle = angle
    while flag:
        if _check_rotate_centerline(img, current_angle, pivot):
            flag = False
        else:
            current_angle = current_angle-step if angle > 0 else current_angle+step
            current_angle = max(0, current_angle) if angle > 0 else min(0, current_angle)

    return current_angle


def _rotate_point_(point, angle, pivot, img_size):
    padX = [img_size - pivot[0], pivot[0]]
    padY = [img_size - pivot[1], pivot[1]]
    imgT = np.zeros([img_size, img_size], dtype=np.uint8)
    imgT[point] = 1
    assert imgT.sum() == 1
    imgP = np.pad(imgT, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    imgR = imgR[padY[0]: -padY[1], padX[0] : -padX[1]]
    # print(f"_rotate_point_, point = {point}, angle = {angle}, pivot = {pivot}")
    assert imgR.sum() == 1
    return (np.where(imgR>0)[0][0], np.where(imgR>0)[1][0])


def rotate_artery(g, branch_name, degree):
    # 1. make a binary image for set of artery branch
    vessel_segment, node_idx = _find_branch_data_(g, branch_name)
    assert node_idx != -1
    
    # 2. rotate vessel_segment data
    # 2.1 find rotation pivot
    pivot, _ = _find_pivot_point_(g, branch_name)

    # 2.2 rotate current segment

    new_artery_mask = _rotate_image_(vessel_segment.vessel_mask, degree, pivot)
    new_artery_img = _rotate_image_(vessel_segment.vessel_image, degree, pivot)
    new_artery_centerline = _rotate_image_(vessel_segment.vessel_centerline, degree, pivot)
    # rotate image
    vessel_segment.vessel_mask = new_artery_mask
    vessel_segment.vessel_image = new_artery_img
    vessel_segment.vessel_centerline = new_artery_centerline
    # rotate point
    # def rotate_point(point, angle, pivot, img_size):
    new_point1 = _rotate_point_((vessel_segment.node1.x, vessel_segment.node1.y), degree, pivot, 512)
    vessel_segment.node1.x, vessel_segment.node1.y = new_point1[0], new_point1[1]
    
    new_point2 = _rotate_point_((vessel_segment.node2.x, vessel_segment.node2.y), degree, pivot, 512) 
    vessel_segment.node2.x, vessel_segment.node2.y = new_point2[0], new_point2[1]

    # 2.3 rotate each child coronay artery branch
    for child in children_table[branch_name]:
        child_segment, node_idx = _find_branch_data_(g, child)
        if node_idx!= -1:
            # rotate point
            new_point1 = _rotate_point_((child_segment.node1.x, child_segment.node1.y), degree, pivot, 512) 
            child_segment.node1.x, child_segment.node1.y = new_point1[0], new_point1[1]
            new_point2 = _rotate_point_((child_segment.node2.x, child_segment.node2.y), degree, pivot, 512) 
            child_segment.node2.x, child_segment.node2.y = new_point2[0], new_point2[1]

            # rotate image
            new_artery_mask = _rotate_image_(child_segment.vessel_mask, degree, pivot)
            new_artery_img = _rotate_image_(child_segment.vessel_image, degree, pivot)
            new_artery_centerline = _rotate_image_(child_segment.vessel_centerline, degree, pivot)

            child_segment.vessel_mask = new_artery_mask
            child_segment.vessel_image = new_artery_img
            child_segment.vessel_centerline = new_artery_centerline
         

def check_rotate(g, branch_name, degree, step=1):
    # 1. check rotate current artery branch
    current_angle = degree
    vessel_segment, node_idx = _find_branch_data_(g, branch_name)
    assert node_idx != -1
    pivot, _ = _find_pivot_point_(g, branch_name)
    img_angle = _check_image_rotate_(vessel_segment.vessel_centerline, current_angle, pivot, step)
    point_angle = _check_end_point_rotate_(vessel_segment, img_angle, pivot, step)
    current_angle = min(img_angle, point_angle) if degree > 0 else max(img_angle, point_angle)

    # 2. check rotate each child branch
    
    for child in children_table[branch_name]:
        child_segment, node_idx = _find_branch_data_(g, child)
        if node_idx!= -1:
            img_angle = _check_image_rotate_(child_segment.vessel_centerline, current_angle, pivot)
            point_angle = _check_end_point_rotate_(child_segment, img_angle, pivot, step)
            current_angle = min(img_angle, point_angle) if degree > 0 else max(img_angle, point_angle)

            if current_angle == 0:
                return current_angle
            
    return current_angle

####################### Resize ################################
def _check_resize_point_(point, pivot, scale_factor, img_size):
    imgT = np.zeros([img_size, img_size], dtype=np.uint8)
    imgT[point] = 1
    imgR = _resize_image_(imgT, pivot, scale_factor)
    return True if (imgR>0).sum() > 0 else False


def _check_end_point_resize_(vessel_segment, pivot, scale_factor, img_size=512, step=0.01):

    flag = True
    current_factor = scale_factor
    while flag:
        if _check_resize_point_((vessel_segment.node1.x, vessel_segment.node1.y), pivot, current_factor, img_size) and \
           _check_resize_point_((vessel_segment.node2.x, vessel_segment.node2.y), pivot, current_factor, img_size):
            flag = False
        else:
            current_factor = current_factor-step if scale_factor > 1 else current_factor+step
            current_factor = max(1, current_factor) if scale_factor > 1 else min(1, current_factor)
    
    return current_factor


def _check_image_resize_(img, pivot, scale_factor, step=0.01):
    current_factor = scale_factor
    # flag = True
    # while flag:
    #     imgT = _resize_image_(img, pivot, scale_factor)
    #     labeling = measure.regionprops(measure.label(imgT))
    #     if len(labeling) == 1:
    #         flag = False
    #     else:
    #         current_factor = current_factor-step if scale_factor > 1 else current_factor+step
    #         current_factor = max(1, current_factor) if scale_factor > 1 else min(1, current_factor)
    
    return current_factor

def _resize_image_(image, pivot, scale_factor):
    # Calculate new size
    image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    width, height = image.size
    new_width = int(scale_factor * width)
    new_height = int(scale_factor * height)

    # Calculate pivot offset
    pivot_x, pivot_y = pivot
    offset_x = int((1 - scale_factor) * pivot_x)
    offset_y = int((1 - scale_factor) * pivot_y)

    # Resize image
    image = image.resize((new_width, new_height))

    # Crop image
    left, top = offset_x, offset_y
    right = offset_x + width
    bottom = offset_y + height
    image = image.crop((left, top, right, bottom))
    
    image = _pil_to_numpy_(image)
    return image

def _resize_point_(point, pivot, scale_factor, img_size):
    imgT = np.zeros([img_size, img_size], dtype=np.uint8)
    imgT[point] = 1
    imgR = _resize_image_(imgT, pivot, scale_factor)
    return (np.where(imgR>0)[0][0], np.where(imgR>0)[1][0])

def _pil_to_numpy_(image):
    # Convert PIL image to NumPy array
    array = np.array(image)

    # Convert RGBA format to RGB if necessary
    if len(array.shape) == 3 and array.shape[2] == 4:
        array = array[:, :, :3]

    return array


#factor = check_end_point_resize(vessel_segment, pivot, factor)
def check_resize(g, branch_name, factor, step=0.01):
    factors = []
    # 1. make a binary image for set of artery branch
    vessel_segment, node_idx = _find_branch_data_(g, branch_name)
    assert node_idx != -1
    
    # 2. rotate vessel_segment data
    # 2.1 find rotation pivot
    pivot, node_idx = _find_pivot_point_(g, branch_name)
    # print(f"pivot = {pivot}")
    
    point_fact = _check_end_point_resize_(vessel_segment, pivot, factor, step=step)
    img_fact = _check_image_resize_(vessel_segment.vessel_mask, pivot, factor, step)
    if factor > 1:
        factors.append(min(point_fact, img_fact))
    else:
        factors.append(max(point_fact, img_fact))


    # resize children
    for child in children_table[branch_name]:
        child_segment, node_idx = _find_branch_data_(g, child)
        if node_idx!= -1:
            img_fact = _check_image_resize_(child_segment.vessel_mask, pivot, factor, step)
            point_fact = _check_end_point_resize_(child_segment, pivot, factor, step=step)
            if factor > 1:
                factors.append(min(point_fact, img_fact))
            else:
                factors.append(max(point_fact, img_fact))
    return min(factors) if factor > 1.0 else max(factors)

def resize_branch(g, branch_name, factor):
    # 1. make a binary image for set of artery branch
    vessel_segment, node_idx = _find_branch_data_(g, branch_name)
    assert node_idx != -1
    
    # 2. rotate vessel_segment data
    # 2.1 find rotation pivot
    pivot, node_idx = _find_pivot_point_(g, branch_name)
    # print(f"pivot = {pivot}")
    
    new_artery_mask = _resize_image_(vessel_segment.vessel_mask, pivot, factor)
    new_artery_img = _resize_image_(vessel_segment.vessel_image, pivot, factor)
    new_artery_centerline = _resize_image_(vessel_segment.vessel_centerline, pivot, factor)
    vessel_segment.vessel_mask = new_artery_mask
    vessel_segment.vessel_image = new_artery_img
    vessel_segment.vessel_centerline = new_artery_centerline
    
    # def _resize_point_(point, pivot, scale_factor, img_size):
    new_point1 = _resize_point_((vessel_segment.node1.x, vessel_segment.node1.y), pivot, factor, 512)
    # print(f"{branch_name}, node1: {(vessel_segment.node1.x, vessel_segment.node1.y)}, to {new_point1}")
    vessel_segment.node1.x, vessel_segment.node1.y = new_point1[0], new_point1[1]
    
    new_point2 = _resize_point_((vessel_segment.node2.x, vessel_segment.node2.y), pivot, factor, 512) 
    # print(f"{branch_name}, node2: {(vessel_segment.node2.x, vessel_segment.node2.y)}, to {new_point2}")
    vessel_segment.node2.x, vessel_segment.node2.y = new_point2[0], new_point2[1]

    # resize children
    for child in children_table[branch_name]:
        child_segment, node_idx = _find_branch_data_(g, child)
        if node_idx!= -1:
            new_artery_mask = _resize_image_(child_segment.vessel_mask, pivot, factor)
            new_artery_img = _resize_image_(child_segment.vessel_image, pivot, factor)
            new_artery_centerline = _resize_image_(child_segment.vessel_centerline, pivot, factor)
            # rotate image
            child_segment.vessel_mask = new_artery_mask
            child_segment.vessel_image = new_artery_img
            child_segment.vessel_centerline = new_artery_centerline
            # rotate point
            # def rotate_point(point, angle, pivot, img_size):
            new_point1 = _resize_point_((child_segment.node1.x, child_segment.node1.y), pivot, factor, 512) 
            # print(f"{child}, node1: {(child_segment.node1.x, child_segment.node1.y)}, to {new_point1}")
            child_segment.node1.x, child_segment.node1.y = new_point1[0], new_point1[1]
            
            new_point2 = _resize_point_((child_segment.node2.x, child_segment.node2.y), pivot, factor, 512) 
            # print(f"{child}, node2: {(child_segment.node2.x, child_segment.node2.y)}, to {new_point2}")
            child_segment.node2.x, child_segment.node2.y = new_point2[0], new_point2[1]


#################### Trim artery branch ######################
def trim_graph(g, prob):
    removed_nodes_idx = []
    removed_nodes = []

    g = copy.deepcopy(g)
    for node in g.nodes():
        if g.nodes()[node]['data'].vessel_class == "LMA":
            continue
        
        if g.nodes()[node]['data'].vessel_class in ["OM1", "OM2", "D1", "D2"]:
        # if g.nodes()[node]['data'].node1.degree == 1 or g.nodes()[node]['data'].node2.degree == 1:
            # removeable
            if np.random.rand() < prob:  # generate random value according to pre-defined seed
                removed_nodes.append(g.nodes()[node]['data'].vessel_class)
                removed_nodes_idx.append(node)

    for n in removed_nodes_idx:
        g.remove_node(n)
    # re assign node index
    mapping = {old_label: new_label for new_label, old_label in enumerate(g.nodes())}
    g = nx.relabel_nodes(g, mapping)
    return g, removed_nodes


def _merge_vessel_objs_(v_obj1: VesselSegment, v_obj2: VesselSegment, clazz):
    v_obj1.vessel_centerline += v_obj2.vessel_centerline
    v_obj1.vessel_centerline = np.clip(v_obj1.vessel_centerline, 0, 1)
    v_obj1.vessel_mask += v_obj2.vessel_mask
    v_obj1.vessel_mask = np.clip(v_obj1.vessel_mask, 0, 1)
    v_obj1.vessel_image += v_obj2.vessel_image
    # merge nodes from two vessel segments
    if v_obj1.node1.x == v_obj2.node1.x and v_obj1.node1.y == v_obj2.node1.y:
        v_obj1.node1 = Node(v_obj1.node2.degree, v_obj1.node2.x, v_obj1.node2.y)
        v_obj1.node2 = Node(v_obj2.node2.degree, v_obj2.node2.x, v_obj2.node2.y)
    elif v_obj1.node1.x == v_obj2.node2.x and v_obj1.node1.y == v_obj2.node2.y:
        v_obj1.node1 = Node(v_obj1.node2.degree, v_obj1.node2.x, v_obj1.node2.y)
        v_obj1.node2 = Node(v_obj2.node1.degree, v_obj2.node1.x, v_obj2.node1.y)
    elif v_obj1.node2.x == v_obj2.node2.x and v_obj1.node2.y == v_obj2.node2.y:
        v_obj1.node1 = Node(v_obj1.node1.degree, v_obj1.node1.x, v_obj1.node1.y)
        v_obj1.node2 = Node(v_obj2.node1.degree, v_obj2.node1.x, v_obj2.node1.y)
    elif v_obj1.node2.x == v_obj2.node1.x and v_obj1.node2.y == v_obj2.node1.y:
        v_obj1.node1 = Node(v_obj1.node1.degree, v_obj1.node1.x, v_obj1.node1.y)
        v_obj1.node2 = Node(v_obj2.node2.degree, v_obj2.node2.x, v_obj2.node2.y)

    r_obj = VesselSegment(v_obj1.node1, v_obj1.node2, v_obj1.vessel_centerline)
    r_obj.vessel_mask = v_obj1.vessel_mask
    r_obj.vessel_image = v_obj1.vessel_image
    r_obj.vessel_class = clazz
    return r_obj


def merge_graph(g: nx.Graph, removed_nodes):
    if "OM1" in removed_nodes and "OM2" not in removed_nodes:
        if _find_branch_data_(g, "OM2")[1] == -1:
            # no OM2, merge LCX1 and LCX2 to LCX1
            vessel_segment_1, idx_keep = _find_branch_data_(g, "LCX1")
            vessel_segment_2, idx_remove = _find_branch_data_(g, "LCX2")
            vessel_segment = _merge_vessel_objs_(vessel_segment_1, vessel_segment_2, "LCX1")
            g.remove_node(idx_remove)
            g.nodes[idx_keep]['data'] = vessel_segment
        else:
            # has OM2, merge LCX1 and LCX2 to LCX1, LCX3 to LCX2, OM2 to OM1
            vessel_segment_1, idx_keep = _find_branch_data_(g, "LCX1")
            vessel_segment_2, idx_remove = _find_branch_data_(g, "LCX2")
            vessel_segment = _merge_vessel_objs_(vessel_segment_1, vessel_segment_2, "LCX1")
            g.remove_node(idx_remove)
            g.nodes[idx_keep]['data'] = vessel_segment

            _, idx_lcx3 = _find_branch_data_(g, "LCX3")
            g.add_edge(idx_keep, idx_lcx3)

            _, idx_om2 = _find_branch_data_(g, "OM2")
            g.add_edge(idx_keep, idx_om2)

            vessel_segment, _ = _find_branch_data_(g, "LCX3")
            vessel_segment.vessel_class = "LCX2"

            vessel_segment, _ = _find_branch_data_(g, "OM2")
            vessel_segment.vessel_class = "OM1"

    if "OM2" in removed_nodes and "OM1" not in removed_nodes:
        # merge LCX2 and LCX3
        vessel_segment_1, idx_keep = _find_branch_data_(g, "LCX2")
        vessel_segment_2, idx_remove = _find_branch_data_(g, "LCX3")
        vessel_segment = _merge_vessel_objs_(vessel_segment_1, vessel_segment_2, "LCX2")
        g.remove_node(idx_remove)
        g.nodes[idx_keep]['data'] = vessel_segment
    
    if "OM1" in removed_nodes and "OM2" in removed_nodes:
        # merge LCX1, LCX2, LCX3
        vessel_segment_1, idx_keep = _find_branch_data_(g, "LCX2")
        vessel_segment_2, idx_remove = _find_branch_data_(g, "LCX3")
        vessel_segment = _merge_vessel_objs_(vessel_segment_1, vessel_segment_2, "LCX2")
        g.remove_node(idx_remove)
        g.nodes[idx_keep]['data'] = vessel_segment

        vessel_segment_1, idx_keep = _find_branch_data_(g, "LCX1")
        vessel_segment_2, idx_remove = _find_branch_data_(g, "LCX2")
        vessel_segment = _merge_vessel_objs_(vessel_segment_1, vessel_segment_2, "LCX1")
        g.remove_node(idx_remove)
        g.nodes[idx_keep]['data'] = vessel_segment

    if "D1" in removed_nodes and "D2" not in removed_nodes:
        if _find_branch_data_(g, "D2")[1] == -1:
            # no D2, merge LAD1 and LAD2 to LAD1
            vessel_segment_1, idx_keep = _find_branch_data_(g, "LAD1")
            vessel_segment_2, idx_remove = _find_branch_data_(g, "LAD2")
            vessel_segment = _merge_vessel_objs_(vessel_segment_1, vessel_segment_2, "LAD1")
            g.remove_node(idx_remove)
            g.nodes[idx_keep]['data'] = vessel_segment
        else:
            # has D2, merge LAD1 and LAD2 to LAD1, LAD3 to LAD2, D2 to D1
            vessel_segment_1, idx_keep = _find_branch_data_(g, "LAD1")
            vessel_segment_2, idx_remove = _find_branch_data_(g, "LAD2")
            vessel_segment = _merge_vessel_objs_(vessel_segment_1, vessel_segment_2, "LAD1")
            g.remove_node(idx_remove)
            g.nodes[idx_keep]['data'] = vessel_segment

            _, idx_lad3 = _find_branch_data_(g, "LAD3")
            g.add_edge(idx_keep, idx_lad3)

            _, idx_d2 = _find_branch_data_(g, "D2")
            g.add_edge(idx_keep, idx_d2)

            vessel_segment, _ = _find_branch_data_(g, "LAD3")
            vessel_segment.vessel_class = "LAD2"

            vessel_segment, _ = _find_branch_data_(g, "D2")
            vessel_segment.vessel_class = "D2"
    
    if "D2" in removed_nodes and "D1" not in removed_nodes:
        # merge LAD2 and LAD3
        vessel_segment_1, idx_keep = _find_branch_data_(g, "LAD2")
        vessel_segment_2, idx_remove = _find_branch_data_(g, "LAD3")
        vessel_segment = _merge_vessel_objs_(vessel_segment_1, vessel_segment_2, "LAD2")
        g.remove_node(idx_remove)
        g.nodes[idx_keep]['data'] = vessel_segment
    
    if "D1" in removed_nodes and "D2" in removed_nodes:
        # merger LAD1, LAD2 and LAD3
        # merge LCX1, LCX2, LCX3
        vessel_segment_1, idx_keep = _find_branch_data_(g, "LAD2")
        vessel_segment_2, idx_remove = _find_branch_data_(g, "LAD3")
        vessel_segment = _merge_vessel_objs_(vessel_segment_1, vessel_segment_2, "LAD2")
        g.remove_node(idx_remove)
        g.nodes[idx_keep]['data'] = vessel_segment

        vessel_segment_1, idx_keep = _find_branch_data_(g, "LAD1")
        vessel_segment_2, idx_remove = _find_branch_data_(g, "LAD2")
        vessel_segment = _merge_vessel_objs_(vessel_segment_1, vessel_segment_2, "LAD1")
        g.remove_node(idx_remove)
        g.nodes[idx_keep]['data'] = vessel_segment

    mapping = {}
    for idx, node in enumerate(g.nodes):
        mapping[node] = idx
    g_new = nx.relabel_nodes(g, mapping, copy=True)
    
    return g_new
