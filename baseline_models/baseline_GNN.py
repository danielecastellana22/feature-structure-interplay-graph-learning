import math
import torch as th
import torch.nn as nn
import torch_geometric.nn as geom_nn
import torch.nn.functional as F
from training import end_to_end_training


class BaseGNN(nn.Module):

    def __init__(self, n_layers, num_in_channels, num_hidden_channels, num_out_channels,
                 p_dropout, p_input_dropout, augmented, skip_connections, concat_ego_neigh_embs,
                 share_layer=False, **other_params):
        super(BaseGNN, self).__init__()
        self.layers_list = nn.ModuleList()
        self.skip_connections = skip_connections

        self.p_dropout = p_dropout
        self.p_input_dropout = p_input_dropout

        self.concat_ego_neigh_embs = concat_ego_neigh_embs
        if concat_ego_neigh_embs:
            self.combination_module_list = nn.ModuleList()
        else:
            self.combination_module_list = None

        #self.share_layer = share_layer
        #self.num_iters = 1 if not self.share_layer else self.n_layers
        assert n_layers > 0
        in_size = num_in_channels
        for i in range(n_layers if not share_layer else 2):
            if i > 0 and self.skip_connections:
                in_size += num_in_channels
            layer, out_size = self.__init_conv__(in_channels=in_size,
                                                 out_channels=num_hidden_channels,
                                                 add_self_loops=augmented,
                                                 **other_params)
            self.layers_list.append(layer)
            if self.combination_module_list is not None:
                self.combination_module_list.append(nn.Linear(in_size + out_size, out_size))

            in_size = out_size

        if share_layer:
            self.layers_list = nn.ModuleList([self.layers_list[0]] + [self.layers_list[-1]]*(n_layers-1))
            if self.combination_module_list is not None:
                self.combination_module_list = nn.ModuleList([self.combination_module_list[0]] + [self.combination_module_list[-1]]*(n_layers-1))

        self.classifier = nn.Linear(in_size, num_out_channels)

    def __init_conv__(self, **params):
        raise NotImplementedError('Must be implemented in the sub class')

    def forward(self, data, **other_parms):
        layer_input = data.x
        edge_index = data.edge_index
        h_list = []
        e_w_list = []

        layer_input = F.dropout(layer_input, p=self.p_input_dropout, training=self.training)

        for i, l in enumerate(self.layers_list):
            # build the input
            if i>0 and self.skip_connections:
                layer_input = th.concat([layer_input, data.x], dim=1)

            # compute the output
            all_res = l(layer_input, edge_index, **other_parms)
            if not isinstance(all_res, tuple):
                # only one output
                layer_output = all_res
            else:
                # multiple output -> in the case of GAT to retrieve e_w
                layer_output = all_res[0]
                e_w = all_res[1][1]
                e_w_list.append(e_w)

            # manage the output
            if self.combination_module_list is not None:
                layer_output = self.combination_module_list[i](th.concat([layer_input, layer_output], dim=1))
            layer_output = F.relu(layer_output)
            layer_output = F.dropout(layer_output, p=self.p_dropout, training=self.training)

            h_list.append(layer_output)

            layer_input = layer_output


        y_pred = self.classifier(layer_input)

        return h_list, [y_pred], e_w_list if len(e_w_list)>0 else [None]

    @staticmethod
    def get_training_fun():
        return end_to_end_training


class GCN(BaseGNN):

    def __init__(self, **kwargs):
        super(GCN, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        conv = geom_nn.GCNConv(**params)
        return conv, conv.out_channels


class GATv2(BaseGNN):

    def __init__(self, **kwargs):
        super(GATv2, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        expected_out_channels = params['out_channels']
        n_heads = params['heads']

        true_out_channels = int(math.ceil(expected_out_channels / n_heads))
        params['out_channels'] = true_out_channels

        return geom_nn.GATv2Conv(**params), true_out_channels*n_heads

    def forward(self, data, **other_parms):
        return super(GATv2, self).forward(data, return_attention_weights=True, **other_parms)


class GraphSAGE(BaseGNN):

    def __init__(self, **kwargs):
        super(GraphSAGE, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        conv = geom_nn.SAGEConv(**params)
        return conv, conv.out_channels


class PNA(BaseGNN):

    def __init__(self, **kwargs):
        super(PNA, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        conv = geom_nn.SAGEConv(**params)
        return conv, conv.out_channels


class GIN(BaseGNN):

    def __init__(self, **kwargs):
        super(GIN, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        in_ch = params.pop('in_channels')
        out_ch = params.pop('out_channels')
        phi = nn.Sequential(nn.Linear(in_ch, out_ch), nn.ReLU(), nn.Linear(out_ch, out_ch))
        conv = geom_nn.GINConv(nn=phi, **params)
        return conv, out_ch


class AntisymmetricGNN(BaseGNN):

    def __init__(self, **kwargs):
        super(AntisymmetricGNN, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        params.pop('out_channels')
        params.pop('add_self_loops')

        conv = geom_nn.AntiSymmetricConv(**params)
        return conv, conv.in_channels


class MLP(BaseGNN):

    class MyMLPLayer(nn.Module):
        def __init__(self, in_channels, out_channels, act=F.relu, **other_params):
            super().__init__()
            self.out_channels = out_channels
            self.in_channels = in_channels
            self.l = nn.Linear(in_channels, out_channels)
            self.act = act

        def forward(self, layer_input, edge_index, **other_parms):
            return self.act(self.l(layer_input))

    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        params.pop('add_self_loops')

        conv = MLP.MyMLPLayer(**params)
        return conv, conv.out_channels
