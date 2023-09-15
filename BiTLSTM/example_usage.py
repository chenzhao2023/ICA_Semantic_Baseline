import torch
import torch.nn as nn
from tree_lstm import TreeLSTM, calculate_evaluation_orders


def _label_node_index(node, n=0):
    node['index'] = n
    for child in node['children']:
        n += 1
        _label_node_index(child, n)


def _gather_node_attributes(node, key):
    features = [node[key]]
    for child in node['children']:
        features.extend(_gather_node_attributes(child, key))
    return features


def _gather_adjacency_list(node):
    adjacency_list = []
    for child in node['children']:
        adjacency_list.append([node['index'], child['index']])
        adjacency_list.extend(_gather_adjacency_list(child))

    return adjacency_list


def convert_tree_to_tensors(tree, device=torch.device('cpu')):
    # Label each node with its walk order to match nodes to feature tensor indexes
    # This modifies the original tree as a side effect
    _label_node_index(tree)

    features = _gather_node_attributes(tree, 'features')
    labels = _gather_node_attributes(tree, 'labels')
    adjacency_list = _gather_adjacency_list(tree)

    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))

    return {
        'features': torch.tensor(features, device=device, dtype=torch.float32),
        'labels': torch.tensor(labels, device=device, dtype=torch.float32),
        'node_order': torch.tensor(node_order, device=device, dtype=torch.int64),
        'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.int64),
        'edge_order': torch.tensor(edge_order, device=device, dtype=torch.int64),
    }



class TreeLSTM_MLP(nn.Module):
    def __init__(self, in_dim, output_dim):
        super().__init__()
    
    



if __name__ == '__main__':
    # Toy example
    device = "cuda:0"
    tree = {
        'features': [1, 0], 
        'labels': [0, 1], 
        'children': [
                    {'features': [0, 1], 'labels': [1, 0], 'children': []},
                    {'features': [0, 0], 'labels': [1, 0], 'children': [
                                    {'features': [1, 1], 'labels': [1, 0], 'children': []}
                    ]},
        ],
    }

    data = convert_tree_to_tensors(tree, device)

    model = TreeLSTM(2, 2).train()
    model.to(device)

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for n in range(1000):
        optimizer.zero_grad()

        h, c = model(data['features'], 
                     data['node_order'], 
                     data['adjacency_list'], 
                     data['edge_order'])
        labels = data['labels']

        loss = loss_function(h, labels)
        loss.backward()
        optimizer.step()

        print(f'Iteration {n+1} Loss: {loss}')
    print(data)