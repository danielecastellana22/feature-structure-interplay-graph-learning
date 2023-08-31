import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch_geometric
import networkx as nx
import torch as th


def plot_node_embs(node_embs, y, tr_mask=None, ax: plt.Axes = None,):
    if ax is None:
        _, ax = plt.subplots()
    else:
        ax.clear()

    node_embs = node_embs.numpy()
    y = y.numpy()

    if node_embs.shape[1] > 2:
        tsne = TSNE()
        embs_2d = tsne.fit_transform(node_embs)
    else:  # elif node_embs.shape[1] == 1:
        embs_2d = np.stack([node_embs[:,0], np.random.randn(node_embs.shape[0])], axis=1)

    cmap='viridis'
    if tr_mask is None:
        ax.scatter(embs_2d[:, 0], embs_2d[:, 1], c=y, cmap=cmap, edgecolors='k')
    else:
        tr_mask = tr_mask.numpy()
        markers = ['o', 's']
        other_mask = np.logical_not(tr_mask)
        ax.scatter(embs_2d[tr_mask, 0], embs_2d[tr_mask, 1], c=y[tr_mask],
                   marker=markers[1], cmap=cmap, edgecolors='k')
        ax.scatter(embs_2d[other_mask, 0], embs_2d[other_mask, 1], c=y[other_mask],
                   marker=markers[0], cmap=cmap, edgecolors='k')

    return ax


def plot_graph(edge_index, y, train_mask=None, e_w=None, exact_count=False, pos=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    else:
        ax.clear()

    edge_attrs = []
    data = torch_geometric.data.Data(edge_index=edge_index, y=y)
    #data.num_nodes = y.shape[0]
    if train_mask is None:
        data.train_mask = th.zeros_like(y)
    else:
        data.train_mask = train_mask

    if e_w is not None:
        edge_attrs = ['e_count', 'e_force']
        if not exact_count:
            max_v = e_w.shape[1]
            data.e_count = th.sum(e_w < 0, dim=1)  # if there are a lot of negative values -> the nodes belong to the same class
        else:
            max_v = 1
            data.e_count = th.all(e_w < 0, dim=1).to(th.long)

        data.e_force = th.exp(6*(data.e_count/max_v)-3)  # is always between e^-3 and e^3

    g = torch_geometric.utils.to_networkx(data, node_attrs=['y', 'train_mask'], edge_attrs=edge_attrs, to_undirected=True)
    plt.sca(ax)

    node_color = np.array(list(zip(*sorted(nx.get_node_attributes(g, 'y').items())))[1])
    if pos is None:
        pos = nx.spring_layout(g, weight='e_force')

    # draw the node
    tr_nodes = [u for u,v in nx.get_node_attributes(g, 'train_mask').items() if v]
    other_nodes = [u for u, v in nx.get_node_attributes(g, 'train_mask').items() if not v]
    nx.draw_networkx_nodes(g, pos=pos, node_size=100, nodelist=tr_nodes, node_shape='s',
                           node_color=node_color[tr_nodes], edgecolors='k')
    nx.draw_networkx_nodes(g, pos=pos, node_size=100, nodelist=other_nodes, node_shape='o',
                           node_color=node_color[other_nodes], edgecolors='k')

    if e_w is None:
        nx.draw_networkx_edges(g, pos=pos)
    else:
        edge_color = np.array(list(zip(*sorted(nx.get_edge_attributes(g, 'e_count').items())))[1])
        nx.draw_networkx_edges(g, pos=pos, edge_color=edge_color,
                               edge_cmap=plt.get_cmap('coolwarm_r'), edge_vmin=0, edge_vmax=max_v)
