import os
import os.path as osp
from abc import ABC

import numpy as np

import torch as th
import torch_geometric
from torch_geometric.datasets.fake import get_edge_index
from typing import List, Union, Tuple
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.utils as geom_utils
import networkx as nx


def __build_torch_geometric_fake_graph__(num_nodes, avg_degree):
    ok = False
    edge_index = None
    # to ensure connected graph
    while not ok:
        edge_index = get_edge_index(num_nodes, num_nodes, avg_degree,
                                    is_undirected=True, remove_loops=True)
        ok = not th.any(torch_geometric.utils.degree(edge_index[0]) == 0).item()

    return edge_index


def __sample_node_features__(num_nodes, num_classes, num_node_features, cluster_id=None):
    mu = 5 * th.arange(num_classes).view(-1, 1).expand(-1, num_node_features)
    sigma = th.ones((num_classes, num_node_features)) * 0.5
    if cluster_id is None:
        cluster_id = th.randint(0, num_classes, size=[num_nodes])
    x = mu[cluster_id] + sigma[cluster_id] * th.randn(size=(num_nodes, num_node_features))
    return x, cluster_id


def __compute_node_degrees_features__(edge_index: th.Tensor):
    return geom_utils.degree(edge_index[0]).reshape(-1, 1)


def __count_triangles__(edge_index: th.Tensor):
    E = edge_index.shape[1]
    N = th.max(edge_index).item() + 1
    y = th.zeros(N, dtype=th.long)

    for u in range(N):
        node_mask = th.zeros(N, dtype=th.bool)
        edge_mask = th.zeros(E, dtype=th.bool)

        hop_1_neighbours = edge_index[1, edge_index[0] == u]

        node_mask.fill_(False)
        node_mask[hop_1_neighbours] = True
        th.index_select(node_mask, 0, edge_index[0], out=edge_mask)
        hop_2_neighbours = edge_index[1, edge_mask]
        hop_2_neighbours = hop_2_neighbours[hop_2_neighbours != u]  # remove u

        idx, v = th.unique(hop_2_neighbours, return_counts=True)  # count how many times a hop-2 neighbours appears
        node_count = th.zeros(N, dtype=th.long)
        node_count[idx] = v
        y[u] = th.sum(node_count[hop_1_neighbours])  # summing the counting of hop-1 neighbours is our result

        assert y[u].item() % 2 == 0
        y[u] = y[u] // 2

    return y


def count_triangles_graph_generator(num_nodes, avg_degree, num_classes, num_node_features, structure_type):
    if structure_type == 'random':
        edge_index = __build_torch_geometric_fake_graph__(num_nodes, avg_degree)
        x = __compute_node_degrees_features__(edge_index)
        y = __count_triangles__(edge_index)

        data = Data(x=x, y=y, edge_index=geom_utils.coalesce(edge_index))
        return data
    elif structure_type == 'easy':
        n_node_in_circle = num_nodes // 6
        num_nodes = n_node_in_circle * 6
        g = nx.generators.circulant_graph(n_node_in_circle, [1])
        node_id = g.number_of_nodes()
        y = th.zeros(num_nodes, dtype=th.long)

        for u in range(g.number_of_nodes()):
            for i in range(5):
                g.add_node(node_id + i)

            g.add_edge(u, node_id)
            g.add_edge(node_id, node_id + 1)
            g.add_edge(node_id + 1, node_id + 2)
            g.add_edge(node_id + 2, node_id + 3)
            g.add_edge(node_id + 3, node_id + 4)
            g.add_edge(node_id + 4, node_id + 2)

            y[u] = 0
            y[node_id] = 0
            y[node_id + 1] = 0
            y[node_id + 2] = 1
            y[node_id + 3] = 1
            y[node_id + 4] = 1
            node_id += 5

            data = geom_utils.from_networkx(g)
            data.y = y
            data.x = __compute_node_degrees_features__(data.edge_index)
            return data

    elif structure_type == 'balanced':
        n_nodes_for_class = num_nodes // num_classes

        triangle_degree_list = [i for i in range(num_classes) for _ in range(n_nodes_for_class)]
        triangle_degree_list += [num_classes - 1] * (num_nodes - n_nodes_for_class * num_classes)  # add remaining nodes
        for i in range((3 - (sum(triangle_degree_list) % 3)) % 3):  # ensure that the sum is multiple of 3
            triangle_degree_list[i] +=1

        triangle_degree_list = np.array(triangle_degree_list)

        #ind_degree_list = np.ceil(avg_degree + 2*np.exp(-triangle_degree_list)*np.random.randn(num_nodes)).astype(int)
        #ind_degree_list = np.floor((avg_degree + 2*np.random.randn(num_nodes)) * np.exp(-triangle_degree_list)).astype(int)
        ind_degree_list = np.ceil(avg_degree + np.random.randn(num_nodes)).astype(int)
        ind_degree_list = np.clip(ind_degree_list - triangle_degree_list, 0, None)  # remove the triangle degree
        # ensure that the nodes that are not in a triangle are have at least one independent edge
        ind_degree_list[np.logical_and(triangle_degree_list == 0, ind_degree_list == 0)] += 1
        ind_degree_list[0] += np.sum(ind_degree_list) % 2  # ensure that the sum is multiple of 2
        # ind_degree_list = np.zeros(num_nodes, dtype=int)

        joint_degree_list = list(zip(ind_degree_list.tolist(), triangle_degree_list.tolist()))
        g = nx.generators.random_clustered_graph(joint_degree_list)
        g = nx.Graph(g)  # remove parallel edges
        g.remove_edges_from(nx.selfloop_edges(g))  # remove self loops
        assert nx.is_connected(g)

        data = geom_utils.from_networkx(g)
        edge_index = data.edge_index
        n_triangles = __count_triangles__(edge_index)

        # ensure that there at most num_classes classes
        is_not_ok = True
        while is_not_ok:
            max_n_triangle = th.max(n_triangles).item()
            if max_n_triangle == num_classes - 1:
                is_not_ok = False
            else:
                nodes_to_keep = n_triangles < max_n_triangle
                edge_index, _ = geom_utils.subgraph(nodes_to_keep, edge_index, relabel_nodes=True)
                n_triangles = __count_triangles__(edge_index)

        data.edge_index = edge_index
        data.y = n_triangles
        data.x = __compute_node_degrees_features__(edge_index)

        return data

    else:
        raise ValueError(f'Structure type {structure_type} is not known!')


def multipartite_graph_generator(num_nodes, avg_degree, num_classes, num_node_features, connection_type):
    # assign each node to a group
    num_nodes_for_class = num_nodes // num_classes + 1
    y = th.arange(0, num_nodes, dtype=th.long) // num_nodes_for_class

    edges_list = []
    for u in range(num_nodes):
        n_neighbours = max(int(th.ceil(avg_degree + 2 * th.randn(1)).item()), 1)

        if connection_type == 'random':
            possible_neigh = th.where(y != y[u])[0]
        elif connection_type == 'easy':
            target_cluster = (num_classes -1 - y[u])
            if target_cluster == y[u]:
                # num_classes is odd
                target_cluster = 0
            possible_neigh = th.where(y == target_cluster)[0]
        else:
            raise ValueError(f'Connection type {connection_type} is not known!')

        aux = th.randperm(len(possible_neigh))
        for v in possible_neigh[aux[:n_neighbours]]:
            edges_list.append(th.tensor([u, v]))
            edges_list.append(th.tensor([v, u]))

    edge_index = th.stack(edges_list, dim=1)

    x = __compute_node_degrees_features__(edge_index)

    data = Data(x=x, y=y, edge_index=geom_utils.coalesce(edge_index))
    return data


def count_neighbours_type_graph_generator(num_nodes, avg_degree, num_classes, num_node_features, label_type):

    edge_index = __build_torch_geometric_fake_graph__(num_nodes, avg_degree)
    num_clusters = num_classes
    x, cluster_id = __sample_node_features__(num_nodes, num_clusters, num_node_features)
    y = th.zeros(num_nodes, dtype=th.long)

    new_node_id = num_nodes

    for u in range(num_nodes):
        edge_mask = edge_index[0] == u
        neighb = edge_index[1, edge_mask]
        neighbours_cluster_id = cluster_id[neighb]
        counts_neighb_clusters = th.zeros(num_clusters)
        counts_neighb_clusters.index_add_(0, neighbours_cluster_id, th.ones_like(neighbours_cluster_id, dtype=th.float))

        if label_type == 'most-common':
            max_rep = th.max(counts_neighb_clusters)
            most_common_classes = th.nonzero(counts_neighb_clusters == max_rep, as_tuple=True)[0]
            n_most_common_classes = most_common_classes.shape[0]
            most_common_class = most_common_classes[0]
            new_x = None
            if n_most_common_classes > 1:
                # there is tie between two classes to break
                idx_most_commmon_class = th.randint(n_most_common_classes, size=[1])
                most_common_class = most_common_classes[idx_most_commmon_class]
                # we add the node and the edge
                new_x, _ = __sample_node_features__(1, num_clusters, num_node_features, cluster_id=most_common_class)
                x = th.cat([x, new_x], dim=0)
                cluster_id = th.cat([cluster_id, most_common_class], dim=0)
                new_edges= th.tensor([[u, new_node_id], [new_node_id, u]])
                edge_index = th.cat([edge_index, new_edges], dim=1)
                new_node_id += 1

            y[u] = most_common_class
            if new_x is not None:
                y = th.cat([y, th.tensor([cluster_id[u]])], dim=0)

        elif label_type == 'least-common':
            y[u] = th.argmin(counts_neighb_clusters)

        elif label_type == 'parity':
            assert num_classes in [2,4]
            if num_classes == 2:
                y[u] = counts_neighb_clusters[1] % 2
            if num_classes == 4:
                c1 = counts_neighb_clusters[1] + counts_neighb_clusters[3]
                c2 = counts_neighb_clusters[2] + counts_neighb_clusters[3]
                y[u] = 2*(c2 % 2) + c1 % 2
        else:
            raise ValueError(f'Label type {label_type} is not known!')

        assert x.numel() == y.numel() == cluster_id.numel()

    assert not torch_geometric.utils.contains_isolated_nodes(edge_index)
    data = Data(x=x, y=y, edge_index=geom_utils.coalesce(edge_index))
    return data


class Synthetic(InMemoryDataset, ABC):
    GENERATOR_FUN = {'count-neighbours-type': count_neighbours_type_graph_generator,
                     'multipartite': multipartite_graph_generator,
                     'count-triangles': count_triangles_graph_generator}

    POSSIBLE_NAMES = list(GENERATOR_FUN.keys())

    def __init__(self, root: str, name: str, num_nodes, avg_degree=None, num_classes=None, num_node_features=1,
                 **other_params):
        super().__init__(root)
        if name not in self.POSSIBLE_NAMES:
            raise ValueError(f'Dataset name {name} is not known!')

        self._name = name
        self._num_nodes = num_nodes
        self._avg_degree = avg_degree
        self._num_node_features = num_node_features
        self._num_classes = num_classes
        self._other_name = ''
        for k,v in other_params.items():
            self._other_name += str(v) + '_'
        self._other_name = self._other_name[:-1]

        if osp.exists(self.processed_paths[0]):
            print('Loading the stored data...')
            self._data, self.slices = th.load(self.processed_paths[0])
        else:
            # we generate the data
            print('Generating the new data...')
            data = self.GENERATOR_FUN[name](num_nodes, avg_degree, num_classes, num_node_features, **other_params)
            self._data, self.slices = self.collate([data])
            self.create_splits()
            os.makedirs(self.processed_dir)
            th.save((self._data, self.slices), self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        raise ValueError('Synthetic datases have no raw dir')
        # return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self._name, self._other_name, f'{self._num_nodes}_{self._num_classes}_{self._avg_degree}', 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        raise ValueError('Synthetic datases have no raw files')

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'data.pt'

    def create_splits(self, n_splits=10, split_perc=None):

        if split_perc is None:
            split_perc = [70, 10, 20]

        # create the splits
        N = self._data.y.shape[0]
        self._data.train_mask = th.zeros((N, n_splits), dtype=th.bool)
        self._data.val_mask = th.zeros((N, n_splits), dtype=th.bool)
        self._data.test_mask = th.zeros((N, n_splits), dtype=th.bool)
        for i in range(n_splits):
            N_tr = (split_perc[0] * N) // 100
            N_val = (split_perc[1] * N) // 100
            N_test = N - N_tr - N_val  # (split_perc[2] * N) // 100
            aux = th.randperm(N)
            self._data.train_mask[aux[:N_tr], i] = True
            self._data.val_mask[aux[N_tr:N_tr + N_val], i] = True
            self._data.test_mask[aux[N_test:], i] = True
