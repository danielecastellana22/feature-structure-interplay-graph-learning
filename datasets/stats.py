import torch as th
from torch_geometric.utils import degree


def compute_edge_homophily(dataset):
    y = dataset.y
    edge_index = dataset.edge_index
    return th.mean((y[edge_index[0]] == y[edge_index[1]]).float()).item()


def compute_adjusted_homophily(dataset):
    y = dataset.y
    edge_index = dataset.edge_index
    E = edge_index.shape[1] // 2
    h_edge = compute_edge_homophily(dataset)
    D_square = degree(edge_index[0])
    sum_D = th.zeros(dataset.num_classes).index_add_(dim=0, index=y, source=D_square)
    adj_const = (th.sum(sum_D**2) / (2*E)**2).item()

    return (h_edge - adj_const) / (1 - adj_const)


def compute_label_informativeness(dataset):
    y = dataset.y
    edge_index = dataset.edge_index

    ind = y[edge_index]
    val = th.ones(ind.shape[1])

    p_c1_c2 = th.sparse_coo_tensor(indices=ind, values=val).coalesce().to_dense()
    Z = th.sum(p_c1_c2)
    assert int(Z.item()) == edge_index.shape[1]
    p_c1_c2 = p_c1_c2 / Z
    log_p_c1_c2 = th.log(p_c1_c2)
    log_p_c1_c2[th.isinf(log_p_c1_c2)] = 0

    #p_c = th.sparse_coo_tensor(indices=y.reshape(1,-1), values=th.ones_like(y, dtype=th.float)).coalesce().to_dense()
    p_c = th.sparse_coo_tensor(indices=y.reshape(1,-1), values=degree(edge_index[0])).coalesce().to_dense()
    Z = th.sum(p_c)
    assert int(Z.item()) == edge_index.shape[1]
    p_c = p_c / Z
    log_p_c = th.log(p_c)
    log_p_c[th.isinf(log_p_c)] = 0

    numerator = th.sum(p_c1_c2 * log_p_c1_c2)
    denominator = th.sum(p_c * log_p_c)
    return (2 - numerator/denominator).item() #, p_c1_c2


def compute_CCNS_matrix(dataset, symmetric_sim_measure=th.cosine_similarity):
    y = dataset.y
    edge_index = dataset.edge_index
    n_classes = dataset.num_classes
    n_nodes = y.shape[0]

    idx = th.stack([edge_index[0], y[edge_index[1]]], dim=0)
    v = th.ones(idx.shape[1])
    empirical_hisotrgam = th.sparse_coo_tensor(idx, v, size=(n_nodes, n_classes)).coalesce().to_dense()

    ccns_matrix = th.zeros(n_classes, n_classes)
    for c1 in range(n_classes):
        # assume symmetry
        for c2 in range(c1, n_classes):
            c1_node_mask = y == c1
            c2_node_mask = y == c2
            all_similarity = symmetric_sim_measure(empirical_hisotrgam[c1_node_mask, :].reshape(-1, 1, n_classes), empirical_hisotrgam[c2_node_mask, :].reshape(1, -1, n_classes), dim=2)
            mean_similarity = th.mean(all_similarity)
            ccns_matrix[c1, c2] = mean_similarity
            ccns_matrix[c2, c1] = mean_similarity

    return ccns_matrix


def __mutual__information__(joint_p_a_b: th.Tensor):
    p_a = th.sum(joint_p_a_b, dim=1, keepdim=True)
    p_b = th.sum(joint_p_a_b, dim=0, keepdim=True)
    eps = 10**-8
    ind_p = p_a * p_b

    log_joint_p = th.log(th.clamp(joint_p_a_b, min=eps))
    log_ind_p = th.log(th.clamp(ind_p, min=eps))
    return th.sum(joint_p_a_b * (log_joint_p - log_ind_p))


def __conditional_entropy__(joint_p_a_b: th.Tensor):
    p_a = th.sum(joint_p_a_b, dim=1, keepdim=True)
    eps = 10**-8

    log_joint_p = th.log(th.clamp(joint_p_a_b, min=eps))
    log_p_a = th.log(th.clamp(p_a, min=eps))
    return -th.sum(joint_p_a_b * (log_joint_p - log_p_a))


def __entropy__(p_a: th.Tensor):
    eps = 10 ** -8
    log_p_a = th.log(th.clamp(p_a, min=eps))
    return -th.sum(p_a * log_p_a)


def compute_feature_informativeness(dataset):
    y = dataset.y
    x = dataset.x
    n_classes = dataset.num_classes

    v_min = th.floor(th.min(x)).item()
    v_max = th.ceil(th.max(x)).item()

    n_bins = int(v_max - v_min)

    p_x_y = th.zeros((n_bins, n_classes), dtype=th.int)

    for label in range(n_classes):
        p_x_y[:, label], a = th.histogram(x[y == label], bins=n_bins, range=(v_min, v_max))

    p_x_y = p_x_y / th.sum(p_x_y)

    p_y = th.sum(p_x_y, dim=0)

    return (__mutual__information__(p_x_y)/__entropy__(p_y)).item()