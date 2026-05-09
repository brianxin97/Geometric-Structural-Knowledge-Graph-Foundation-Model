import torch
import math
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter

import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree, softmax
from typing import Tuple


class GeneralizedRelationalConv(MessagePassing):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
        "rotate": "complex",
        "split": "split_complex",
        "dual": "dual",
        "mobius": "mobius",
        "mobius+": "mobius+",
        "splitmobius": "splitmobius",
        "transrotate": "transrotate",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=False, project_relations=False):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.project_relations = project_relations

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        if dependent:
            # obtain relation embeddings as a projection of the query relation
            #self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
            if self.message_func == "mobius" or self.message_func == "splitmobius" or self.message_func == "mobius+":
                self.relation_linear = nn.Linear(query_input_dim, num_relation * 4 * input_dim)
                nn.init.xavier_uniform_(self.relation_linear.weight)
            elif self.message_func == "transrotate":
                self.relation_linear = nn.Linear(query_input_dim, num_relation * 2 * input_dim)
                nn.init.xavier_uniform_(self.relation_linear.weight)
            else:
                self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
                nn.init.xavier_uniform_(self.relation_linear.weight)
        else:
            if not self.project_relations:
                # relation embeddings as an independent embedding matrix per each layer
                #self.relation = nn.Embedding(num_relation, input_dim)
                if self.message_func == "mobius" or self.message_func == "splitmobius" or self.message_func == "mobius+":
                    self.relation = nn.Embedding(num_relation, 4 * input_dim)
                    torch.nn.init.xavier_uniform_(self.relation.weight)
                elif self.message_func == "transrotate":
                    self.relation = nn.Embedding(num_relation, 2 * input_dim)
                    torch.nn.init.xavier_uniform_(self.relation.weight)
                else:
                    self.relation = nn.Embedding(num_relation, input_dim)
                    torch.nn.init.xavier_uniform_(self.relation.weight)
            else:
                # will be initialized after the pass over relation graph
                self.relation = None
                self.relation_projection = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim)
                )

    def forward(self, input, query, boundary, edge_index, edge_type, size, edge_weight=None):
        batch_size = len(query)

        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            #relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
            if self.message_func == "mobius" or self.message_func == "splitmobius" or self.message_func == "mobius+":
                relation = self.relation_linear(query).view(batch_size, self.num_relation, 4*self.input_dim)
            elif self.message_func == "transrotate":
                relation = self.relation_linear(query).view(batch_size, self.num_relation, 2*self.input_dim)
            else:
                relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            if not self.project_relations:
                # layer-specific relation features as a special embedding matrix unique to each layer
                relation = self.relation.weight.expand(batch_size, -1, -1)
            else:
                # NEW and only change: 
                # projecting relation features to unique features for this layer, then resizing for the current batch
                relation = self.relation_projection(self.relation)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)

        # note that we send the initial boundary condition (node states at layer0) to the message passing
        # correspond to Eq.6 on p5 in https://arxiv.org/pdf/2106.06935.pdf
        output = self.propagate(input=input, relation=relation, boundary=boundary, edge_index=edge_index,
                                edge_type=edge_type, size=size, edge_weight=edge_weight)
        return output

    def propagate(self, edge_index, size=None, **kwargs):
        unsupported_cuda_ops = ["transrotate", "mobius", "splitmobius", "mobius+"]

        is_pna_complex = (self.aggregate_func == "pna" and self.message_func in ["rotate", "split", "dual"])

        if kwargs["edge_weight"].requires_grad or self.message_func in unsupported_cuda_ops or is_pna_complex:
            return super(GeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        # in newer PyG, 
        # __check_input__ -> _check_input()
        # __collect__ -> _collect()
        # __fused_user_args__ -> _fuser_user_args
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._fused_user_args, edge_index, size, kwargs)

        pyg_version = [int(i) for i in torch_geometric.__version__.split(".")]
        col_fn = self.inspector.distribute if pyg_version[1] <= 4 else self.inspector.collect_param_data

        msg_aggr_kwargs = col_fn("message_and_aggregate", coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res

        # PyG 2.5+ distribute -> collect_param_data
        update_kwargs = col_fn("update", coll_dict)
        out = self.update(out, **update_kwargs)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    def message(self, input_j, relation, boundary, edge_type):
        relation_j = relation.index_select(self.node_dim, edge_type)

        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        elif self.message_func == "split":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re + x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        elif self.message_func == "dual":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        elif self.message_func == "transrotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im, r_j_re_tr, r_j_im_tr = relation_j.chunk(4, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im + r_j_re_tr
            message_im = x_j_re * r_j_im + x_j_im * r_j_re + r_j_im_tr
            message = torch.cat([message_re, -message_im], dim=-1)
        elif self.message_func == "mobius":
            re_head, im_head = input_j.chunk(2, dim=-1)
            re_relation_a, im_relation_a, re_relation_b, im_relation_b, re_relation_c, im_relation_c, re_relation_d, im_relation_d = relation_j.chunk(8, dim=-1)
            re_score_a = re_head * re_relation_a + im_head * im_relation_a
            im_score_a = -re_head * im_relation_a + im_head * re_relation_a
            # ah + b
            re_score_top = re_score_a + re_relation_b
            im_score_top = im_score_a + im_relation_b
            # ch
            re_score_c = re_head * re_relation_c + im_head * im_relation_c
            im_score_c = -re_head * im_relation_c + im_head * re_relation_c
            # ch + d
            re_score_dn = re_score_c + re_relation_d
            im_score_dn = im_score_c + im_relation_d
            # (ah + b)Conj(ch+d)
            dn_re = torch.sqrt(re_score_dn * re_score_dn + im_score_dn * im_score_dn)
            message_re = torch.div(re_score_top * re_score_dn + im_score_top * im_score_dn, dn_re)
            message_im = torch.div(-re_score_top * im_score_dn + im_score_top * re_score_dn, dn_re)
            message = torch.cat([message_re, message_im], dim=-1)
        elif self.message_func == "mobius+":
            re_head, im_head = input_j.chunk(2, dim=-1)
            re_relation_a, im_relation_a, re_relation_b, im_relation_b, re_relation_c, im_relation_c, re_relation_d, im_relation_d = relation_j.chunk(8, dim=-1)
            re_score_a = re_head * re_relation_a - im_head * im_relation_a
            im_score_a = re_head * im_relation_a + im_head * re_relation_a
            # ah + b
            re_score_top = re_score_a + re_relation_b
            im_score_top = im_score_a + im_relation_b
            # ch
            re_score_c = re_head * re_relation_c - im_head * im_relation_c
            im_score_c = re_head * im_relation_c + im_head * re_relation_c
            # ch + d
            re_score_dn = re_score_c + re_relation_d
            im_score_dn = im_score_c + im_relation_d
            # (ah + b)Conj(ch+d)
            dn_re = torch.sqrt(re_score_dn * re_score_dn + im_score_dn * im_score_dn)
            message_re = torch.div(re_score_top * re_score_dn - im_score_top * im_score_dn, dn_re)
            message_im = torch.div(re_score_top * im_score_dn + im_score_top * re_score_dn, dn_re)
            message = torch.cat([message_re, message_im], dim=-1)
        elif self.message_func == "splitmobius":
            re_head, im_head = input_j.chunk(2, dim=-1)
            re_relation_a, im_relation_a, re_relation_b, im_relation_b, re_relation_c, im_relation_c, re_relation_d, im_relation_d = relation_j.chunk(8, dim=-1)
            re_score_a = re_head * re_relation_a + im_head * im_relation_a
            im_score_a = re_head * im_relation_a + im_head * re_relation_a
            # ah + b
            re_score_top = re_score_a + re_relation_b
            im_score_top = im_score_a + im_relation_b
            # ch
            re_score_c = re_head * re_relation_c + im_head * im_relation_c
            im_score_c = re_head * im_relation_c + im_head * re_relation_c
            # ch + d
            re_score_dn = re_score_c + re_relation_d
            im_score_dn = im_score_c + im_relation_d
            # (ah + b)Conj(ch+d)
            dn_re = torch.sqrt(torch.abs(re_score_dn * re_score_dn + im_score_dn * im_score_dn))
            message_re = torch.div(re_score_top * re_score_dn + im_score_top * im_score_dn, dn_re)
            message_im = torch.div(re_score_top * im_score_dn - im_score_top * re_score_dn, dn_re)
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # augment messages with the boundary condition
        message = torch.cat([message, boundary], dim=self.node_dim)  # (num_edges + num_nodes, batch_size, input_dim)

        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        # augment aggregation index with self-loops for the boundary condition
        index = torch.cat([index, torch.arange(dim_size, device=input.device)]) # (num_edges + num_nodes,)
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        if self.aggregate_func == "pna":
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            max = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
            min = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="min")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size,
                             reduce=self.aggregate_func)

        return output

    def message_and_aggregate(self, edge_index, input, relation, boundary, edge_type, edge_weight, index, dim_size):
        # fused computation of message and aggregate steps with the custom rspmm cuda kernel
        # speed up computation by several times
        # reduce memory complexity from O(|E|d) to O(|V|d), so we can apply it to larger graphs
        from .rspmm import generalized_rspmm

        batch_size, num_node = input.shape[:2]
        hidden_dim = input.shape[-1]

        boundary_flat = boundary.transpose(0, 1).flatten(1)
        degree_out = degree(index, dim_size).unsqueeze(-1) + 1

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        is_complex_op = mul in ["complex", "split_complex", "dual"]

        if is_complex_op:
            half_dim = hidden_dim // 2
            # (bs, nodes, 2, half_dim) -> (nodes, 2, bs, half_dim) -> flatten
            input_rspmm = input.view(batch_size, num_node, 2, half_dim).permute(1, 2, 0, 3).flatten(1)
            relation_rspmm = relation.view(batch_size, -1, 2, half_dim).permute(1, 2, 0, 3).flatten(1)
        else:
            input_rspmm = input.transpose(0, 1).flatten(1)
            relation_rspmm = relation.transpose(0, 1).flatten(1)

        if self.aggregate_func == "sum":
            update_raw = generalized_rspmm(edge_index, edge_type, edge_weight, relation_rspmm, input_rspmm, sum="add",
                                           mul=mul)
            if is_complex_op:
                # (nodes, 2 * bs * half_dim) -> (nodes, bs * hidden_dim)
                update_raw = update_raw.view(num_node, 2, batch_size, half_dim).permute(0, 2, 1, 3).reshape(num_node,
                                                                                                            batch_size * hidden_dim)
            update = update_raw + boundary_flat

        elif self.aggregate_func == "mean":
            update_raw = generalized_rspmm(edge_index, edge_type, edge_weight, relation_rspmm, input_rspmm, sum="add",
                                           mul=mul)
            if is_complex_op:
                update_raw = update_raw.view(num_node, 2, batch_size, half_dim).permute(0, 2, 1, 3).reshape(num_node,
                                                                                                            batch_size * hidden_dim)
            update = (update_raw + boundary_flat) / degree_out

        elif self.aggregate_func == "max":
            update_raw = generalized_rspmm(edge_index, edge_type, edge_weight, relation_rspmm, input_rspmm, sum="max",
                                           mul=mul)
            if is_complex_op:
                update_raw = update_raw.view(num_node, 2, batch_size, half_dim).permute(0, 2, 1, 3).reshape(num_node,
                                                                                                            batch_size * hidden_dim)
            update = torch.max(update_raw, boundary_flat)

        elif self.aggregate_func == "pna":
            # we use PNA with 4 aggregators (mean / max / min / std)
            # and 3 scalars (identity / log degree / reciprocal of log degree)
            # Note: is_complex_op is guaranteed to be False here due to the block in propagate()
            sum_val = generalized_rspmm(edge_index, edge_type, edge_weight, relation_rspmm, input_rspmm, sum="add",
                                        mul=mul)
            sq_sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation_rspmm ** 2, input_rspmm ** 2,
                                       sum="add", mul=mul)
            max_val = generalized_rspmm(edge_index, edge_type, edge_weight, relation_rspmm, input_rspmm, sum="max",
                                        mul=mul)
            min_val = generalized_rspmm(edge_index, edge_type, edge_weight, relation_rspmm, input_rspmm, sum="min",
                                        mul=mul)

            mean = (sum_val + boundary_flat) / degree_out
            sq_mean = (sq_sum + boundary_flat ** 2) / degree_out
            max_val = torch.max(max_val, boundary_flat)
            min_val = torch.min(min_val, boundary_flat)  # (node, batch_size * input_dim)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()

            features = torch.cat([mean.unsqueeze(-1), max_val.unsqueeze(-1), min_val.unsqueeze(-1), std.unsqueeze(-1)],
                                 dim=-1)
            features = features.flatten(-2)  # (node, batch_size * input_dim * 4)

            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)  # (node, 3)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(
                -2)  # (node, batch_size * input_dim * 4 * 3)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        update = update.view(num_node, batch_size, -1).transpose(0, 1)
        return update

    def update(self, update, input):
        # node update as a function of old states (input) and this layer output (update)
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class TopologyAttentionConv(MessagePassing):
    """
        Topology-aware Graph Convolutional layer using an Attention mechanism.

        This layer updates relation representations by considering the topological structure
        of the relation graph, incorporating specific edge embeddings for different
        meta-relation types.
    """
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super(TopologyAttentionConv, self).__init__(aggr='add', node_dim=1)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.q_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, out_dim, bias=False)
        # Mapping for 4 types of meta-relation topology patterns
        self.edge_emb = nn.Embedding(4, out_dim)

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        # x shape: (bs, num_relations, in_dim)
        bs, num_rel, _ = x.shape
        residual = x

        q = self.q_proj(x).view(bs, num_rel, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bs, num_rel, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bs, num_rel, self.num_heads, self.head_dim)

        # Expand edge embeddings to match batch and head dimensions
        e = self.edge_emb(edge_type).view(1, -1, self.num_heads, self.head_dim).expand(bs, -1, -1, -1)

        out = self.propagate(edge_index, q=q, k=k, v=v, e=e, edge_weight=edge_weight, size=None)

        out = out.view(bs, num_rel, self.out_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        return self.layer_norm(residual + out)

    def message(self, q_i, k_j, v_j, e, edge_weight, index, ptr, size_i):
        # Incorporate topology edge information into Key and Value
        key = k_j + e
        value = v_j + e

        # Calculate attention scores: (bs, num_edges, num_heads)
        alpha = (q_i * key).sum(dim=-1) / math.sqrt(self.head_dim)
        # Normalize scores using softmax over neighbors
        alpha = softmax(alpha, index, ptr, size_i, dim=1)

        alpha = self.dropout(alpha)

        if edge_weight is not None:
            # Apply external edge weights if provided
            # edge_weight shape: (num_edges,) -> (num_edges, 1, 1)
            value = value * edge_weight.view(-1, 1, 1)

        return value * alpha.unsqueeze(-1)


class CoEvolutionCrossAttention(nn.Module):
    """
        Cross-attention module for the co-evolution of Relation and Entity embeddings.

        This layer allows relation features to attend to entity features, capturing
        the mutual influence between the two types of representations.
    """
    def __init__(self, rel_dim, ent_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = rel_dim // num_heads

        self.q_proj = nn.Linear(rel_dim, rel_dim, bias=False)
        self.k_proj = nn.Linear(ent_dim, rel_dim, bias=False)
        self.v_proj = nn.Linear(ent_dim, rel_dim, bias=False)

        self.out_proj = nn.Linear(rel_dim, rel_dim)
        self.layer_norm = nn.LayerNorm(rel_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, rel_feat, ent_feat):
        """
        rel_feat: (batch_size, num_relations, rel_dim)
        ent_feat: (batch_size, num_nodes, ent_dim)
        """
        bs, num_rel, _ = rel_feat.shape
        _, num_nodes, _ = ent_feat.shape
        residual = rel_feat

        # Project features to Queries (from relations), Keys, and Values (from entities)
        # (bs, num_rel, num_heads, head_dim) -> (bs, num_heads, num_rel, head_dim)
        Q = self.q_proj(rel_feat).view(bs, num_rel, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(ent_feat).view(bs, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(ent_feat).view(bs, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores: (bs, num_heads, num_rel, num_nodes)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Compute weighted sum of values and project back
        # Output: (bs, num_heads, num_rel, head_dim) -> (bs, num_rel, rel_dim)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(bs, num_rel, -1)
        out = self.out_proj(out)

        return self.layer_norm(residual + out)
