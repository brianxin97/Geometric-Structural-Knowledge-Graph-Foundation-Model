import torch
import math
from torch import nn
import numpy as np

from . import tasks, layers
from ultra.base_nbfnet import BaseNBFNet


class Ultra(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg):
        # kept that because super Ultra sounds cool
        super(Ultra, self).__init__()

        # adding a bit more flexibility to initializing proper rel/ent classes from the configs
        self.relation_model = globals()[rel_model_cfg.pop('class')](**rel_model_cfg)
        self.entity_model = globals()[entity_model_cfg.pop('class')](**entity_model_cfg)

    def forward(self, data, batch):
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        query_rels = batch[:, 0, 2]
        relation_representations = self.relation_model(data.relation_graph, query=query_rels)
        score = self.entity_model(data, relation_representations, batch)

        return score


# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
            )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    def bellmanford(self, data, h_index, separate_grad=False):
        batch_size = len(h_index)

        # initialize initial nodes (relations of interest in the batcj) with all ones
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
        # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, query):

        # message passing and updated node representations (that are in fact relations)
        output = self.bellmanford(rel_graph, h_index=query)["node_feature"]  # (batch_size, num_nodes, hidden_dim）

        return output


class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, model_list, num_relation=1, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.model_list = model_list

        # Build a separate stack of layers for each message function branch
        self.branches = nn.ModuleDict()
        for msg in self.model_list:
            stack = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                stack.append(
                    layers.GeneralizedRelationalConv(
                        self.dims[i], self.dims[i + 1],
                        num_relation=1,
                        query_input_dim=self.dims[0],
                        message_func=msg,
                        aggregate_func=self.aggregate_func,
                        layer_norm=self.layer_norm,
                        activation=self.activation,
                        dependent=False,
                        project_relations=True
                    )
                )
            self.branches[msg] = stack

        # Output dimension of each single branch
        self.branch_feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim

        # Attention mechanism: map the query to a context vector; map outputs of each branch into the same attention space
        self.att_dim = self.branch_feature_dim
        self.query_to_ctx = nn.Linear(input_dim, self.att_dim, bias=False)
        self.branch_key_proj = nn.ModuleDict({
            msg: nn.Linear(self.branch_feature_dim, self.att_dim, bias=False)
            for msg in self.model_list
        })
        self.scale = 1.0 / math.sqrt(self.att_dim)
        self.softmax = nn.Softmax(dim=2)

        # The input is the concatenated features from all branches
        fused_dim = len(self.model_list) * self.branch_feature_dim
        mlp = []
        for _ in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(fused_dim, fused_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(fused_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def bellmanford(self, stack, data, h_index, query, separate_grad=False):
        batch_size = len(h_index)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, h_index.unsqueeze(1).unsqueeze(-1).expand(-1, 1, self.dims[0]), query.unsqueeze(1))

        hiddens = []
        layer_input = boundary
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
        for layer in stack:
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)
            return output  # (bs, num_nodes, branch_feature_dim)

    def forward(self, data, relation_representations, batch):
        h_index, t_index, r_index = batch.unbind(-1)

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index,
                                                                 num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # Obtain the query representation for each sample from the relational graph embeddings (instead of local embeddings)
        bs = h_index.size(0)
        q = relation_representations[torch.arange(bs, device=h_index.device), r_index[:, 0]]  # (bs, input_dim)

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for stack in self.branches.values():
            for layer in stack:
                layer.relation = relation_representations

        # message passing and updated node representations
        branch_features = []
        for msg in self.model_list:
            feature = self.bellmanford(self.branches[msg], data, h_index[:, 0], q)
            branch_features.append(feature)

        cand = torch.stack(branch_features, dim=2)  # (bs, num_nodes, K, branch_feature_dim)

        # Attention weights are computed as: query → context; branch features → keys; followed by a softmax over the key dimension
        ctx = self.query_to_ctx(q).unsqueeze(1).unsqueeze(2)  # (bs, 1, 1, att_dim)
        keys = torch.stack(
            [self.branch_key_proj[msg](feature) for msg, feature in zip(self.model_list, branch_features)], dim=2
        )  # (bs, num_nodes, K, att_dim)
        att_logits = (keys * ctx).sum(-1, keepdim=True) * self.scale  # (bs, num_nodes, K, 1)
        att = self.softmax(att_logits)

        # After attention weighting, concatenate the weighted branch features
        fused_feature = (att * cand).view(bs, data.num_nodes, -1)  # (bs, num_nodes, K*branch_feature_dim)

        # extract representations of tail entities from the updated node states
        index = t_index.unsqueeze(-1).expand(-1, -1, fused_feature.shape[-1])
        fused_feature = fused_feature.gather(1, index)  # (batch_size, num_negative + 1, K*branch_feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(fused_feature).squeeze(-1)
        return score.view(shape)


class QueryNBFNet(EntityNBFNet):
    """
    The entity-level reasoner for UltraQuery-like complex query answering pipelines
    Almost the same as EntityNBFNet except that
    (1) we already get the initial node features at the forward pass time
    and don't have to read the triples batch
    (2) we get `query` from the outer loop
    (3) we return a distribution over all nodes (assuming t_index = all nodes)
    """

    def bellmanford(self, data, node_features, query, separate_grad=False):

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=query.device)

        hiddens = []
        edge_weights = []
        layer_input = node_features

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, node_features, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, node_features, relation_representations, query):

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        # we already did traversal_dropout in the outer loop of UltraQuery
        # if self.training:
        #     # Edge dropout in the training mode
        #     # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
        #     # to make NBFNet iteration learn non-trivial paths
        #     data = self.remove_easy_edges(data, h_index, t_index, r_index)

        # node features arrive in shape (bs, num_nodes, dim)
        # NBFNet needs batch size on the first place
        output = self.bellmanford(data, node_features, query)  # (num_nodes, batch_size, feature_dim）
        score = self.mlp(output["node_feature"]).squeeze(-1)  # (bs, num_nodes)
        return score
