import torch
import math
from torch import nn
import numpy as np

from . import tasks, layers
from gamma.base_nbfnet import BaseNBFNet


class Gamma(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg):
        # kept that because super Gamma sounds cool
        super(Gamma, self).__init__()

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
    """
    The entity-level reasoner with multi-branch architecture and attention mechanism
    Extends BaseNBFNet to perform message passing on entity graphs using multiple message functions
    Key features:
    (1) Multiple parallel branches, each with a different message function (e.g., rotate, split, dual)
    (2) Attention mechanism to dynamically weight the outputs from different branches based on the query
    (3) Final MLP projection from concatenated branch features to scores
    (4) Returns scoring distribution over entities for tail prediction
    """

    def __init__(self, input_dim, hidden_dims, model_list, num_relation=1,
                 attn_temp: float = 2.0,
                 attn_dropout: float = 0.1,
                 attn_uniform_mix: float = 0.05,
                 entropy_reg_weight: float = 0.003,
                 **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        # list of message functions to create parallel branches
        self.model_list = model_list
        # weight for entropy regularization to encourage diverse branch usage
        self.entropy_reg_weight = float(entropy_reg_weight)
        self.debug_info = {}

        # build a separate stack of layers for each message function branch
        self.branches = nn.ModuleDict()
        for msg in self.model_list:
            stack = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                stack.append(
                    layers.GeneralizedRelationalConv(
                        self.dims[i], self.dims[i + 1],
                        num_relation=1,
                        query_input_dim=self.dims[0],
                        message_func=msg,  # Different message function for each branch
                        aggregate_func=self.aggregate_func,
                        layer_norm=self.layer_norm,
                        activation=self.activation,
                        dependent=False,
                        project_relations=True
                    )
                )
            self.branches[msg] = stack

        # output dimension of each single branch
        self.branch_feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim

        # attention mechanism: map the query to a context vector; map outputs of each branch into the same attention space
        self.att_dim = self.branch_feature_dim
        # project query relation embeddings to attention space
        self.query_to_ctx = nn.Linear(input_dim, self.att_dim, bias=False)
        # project each branch's output to attention space for key-query matching
        self.branch_key_proj = nn.ModuleDict({
            msg: nn.Linear(self.branch_feature_dim, self.att_dim, bias=False)
            for msg in self.model_list
        })

        # temperature parameter for softmax sharpness in attention
        self.attn_temp = nn.Parameter(torch.tensor(attn_temp), requires_grad=False)
        self.att_dropout = nn.Dropout(attn_dropout)
        # mix attention weights with uniform distribution to prevent over-specialization
        self.att_uniform_mix = float(attn_uniform_mix)

        # the input is the concatenated features from all branches
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

        # Bellman-Ford iteration: message passing through multiple layers
        for layer in stack:
            # each layer updates node features based on neighbor messages
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        # expand query embeddings to all nodes for concatenation
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            # concatenate all layer outputs with query embeddings
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            # only use final layer output with query embeddings
            output = torch.cat([hiddens[-1], node_query], dim=-1)
        return output

    def _compute_attention(self, q, branch_features):
        bs = q.size(0)
        K = len(self.model_list)

        # project query to attention context space and normalize
        ctx = self.query_to_ctx(q)
        ctx = F.normalize(ctx, dim=-1).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, att_dim)

        # project each branch's features to keys and stack
        keys = torch.stack(
            [self.branch_key_proj[msg](feat) for msg, feat in zip(self.model_list, branch_features)], dim=2
        )  # (batch_size, num_nodes, K, att_dim)
        keys = F.normalize(keys, dim=-1)

        # compute attention logits via dot product between context and keys
        att_logits = (keys * ctx).sum(-1)  # (batch_size, num_nodes, K)
        # apply temperature-scaled softmax to get attention weights
        att = F.softmax(att_logits / self.attn_temp.clamp(min=1e-6), dim=-1)

        # mix with uniform distribution to prevent over-specialization
        if self.att_uniform_mix > 0:
            uniform = torch.full_like(att, 1.0 / K)
            att = (1.0 - self.att_uniform_mix) * att + self.att_uniform_mix * uniform

        # compute attention entropy for regularization (encourage diversity)
        p = att.clamp(min=1e-8)
        att_entropy = (-p * p.log()).sum(dim=-1).mean()
        # track average attention per branch for monitoring
        att_mean_per_branch = att.mean(dim=(0, 1))

        # apply dropout to attention weights
        att = self.att_dropout(att)

        # stack branch features for weighted combination
        cand = torch.stack(branch_features, dim=2)  # (batch_size, num_nodes, K, branch_feature_dim)

        return att, cand, att_entropy, att_mean_per_branch

    def forward(self, data, relation_representations, batch):
        # unpack batch triples: (head, tail, relation)
        h_index, t_index, r_index = batch.unbind(-1)

        if self.training:
            # edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index,
                                                                 num_direct_rel=data.num_relations // 2)
        # verify that all samples in batch share the same head and relation
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        bs = h_index.size(0)
        # extract query relation embeddings for the batch
        q = relation_representations[torch.arange(bs, device=h_index.device), r_index[:, 0]]

        # initialize relation embeddings in each branch's layers
        for stack in self.branches.values():
            for layer in stack:
                layer.relation = relation_representations

        # run message passing through each branch independently
        branch_features = []
        for msg in self.model_list:
            feat = self.bellmanford(self.branches[msg], data, h_index[:, 0], q)
            branch_features.append(feat)

        # compute attention weights and fuse branch outputs
        att, cand, att_entropy, att_mean_per_branch = self._compute_attention(q, branch_features)

        # weighted combination: (batch_size, num_nodes, K, 1) * (batch_size, num_nodes, K, dim) -> (batch_size, num_nodes, K*dim)
        fused_feature = (att.unsqueeze(-1) * cand).reshape(bs, data.num_nodes, -1)

        # extract representations of tail entities from the updated node states
        index = t_index.unsqueeze(-1).expand(-1, -1, fused_feature.shape[-1])
        fused_feature = fused_feature.gather(1, index)

        # project fused features to scores via MLP
        score = self.mlp(fused_feature).squeeze(-1)
        score = score.view(shape)

        if self.training:
            # return auxiliary information for monitoring and regularization
            aux = {
                "att_entropy": att_entropy.detach(),
                "att_mean_per_branch": att_mean_per_branch.detach(),
            }
            # add entropy regularization loss to encourage diverse branch usage
            if self.entropy_reg_weight > 0:
                aux["aux_loss"] = -self.entropy_reg_weight * att_entropy
            else:
                aux["aux_loss"] = None
            return score, aux
        
        return score

