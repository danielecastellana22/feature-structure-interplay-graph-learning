import os
import os.path as osp
import torch.nn.functional as F
from .metrics import roc_auc_metric, accuracy_metric
from .synthetic import Synthetic
from torch_geometric.datasets import HeterophilousGraphDataset, Planetoid, Actor
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, remove_self_loops
import torch as th
from utils.serialisation import to_torch_file, from_torch_file
from utils.misc import eprint
from tqdm import tqdm


def get_dataset(data_root, dataset_config):
    dataset_name = dataset_config.name
    extra_params = dataset_config.params if 'params' in dataset_config else {}
    if dataset_name in ['cornell', 'texas', 'wisconsin']:
        raise NotImplementedError()
        #dataset = WebKB(root=data_root, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['chameleon', 'squirrel']:
        raise NotImplementedError()
        #dataset = WikipediaNetwork(root=data_root, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name == 'film':
        dataset = Actor(root=data_root, transform=T.NormalizeFeatures(), **extra_params)
    elif dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=data_root, name=dataset_name, transform=T.NormalizeFeatures(), split='geom-gcn', **extra_params)
    elif dataset_name in ['roman-empire', 'minesweeper', 'amazon-ratings']:
        dataset = HeterophilousGraphDataset(root=data_root, name=dataset_name)
    elif dataset_name in Synthetic.POSSIBLE_NAMES:
        dataset = Synthetic(root=data_root, name=dataset_name, **extra_params)
    else:
        raise ValueError(f'dataset {dataset_name} not supported in dataloader')

    dataset.n_splits = dataset.train_mask.shape[1]
    dataset._data.edge_index = to_undirected(remove_self_loops(dataset.edge_index)[0])
    dataset.loss = F.cross_entropy
    if dataset.num_classes == 2:
        # dataset.metric = roc_auc_metric
        dataset.metric = accuracy_metric
    else:
        dataset.metric = accuracy_metric

    # build edge_id
    edge_id_file = os.path.join(dataset.processed_dir, 'edge_id.pt')
    if os.path.exists(edge_id_file):
        dataset.edge_id = from_torch_file(edge_id_file)
    else:
        eprint('Building edge id mapping!')
        tuple_2_id = {}
        E = dataset.edge_index.shape[1]  # number of edges (in both directions)
        edge_id_list = []
        for i in tqdm(range(E), desc='Edges'):
            e = tuple(sorted(dataset.edge_index[:, i].tolist()))
            e_id = tuple_2_id.setdefault(e, len(tuple_2_id))
            edge_id_list.append(e_id)

        edge_id = th.tensor(edge_id_list)
        dataset.edge_id = edge_id
        # store the edge_id map
        to_torch_file(edge_id, edge_id_file)

    return dataset