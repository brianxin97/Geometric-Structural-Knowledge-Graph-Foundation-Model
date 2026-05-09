import torch
import math
from torch import nn
from torch_scatter import scatter
import torch.nn.functional as F
import numpy as np

from . import tasks, layers
from gamma.base_nbfnet import BaseNBFNet


class QueryGuidedFusion(nn.Module):
    def __init__(self, model_list, input_dim, branch_dim, num_heads=4, dropout=0.1, num_mlp_layers=2):
        super().__init__()
        self.model_list = model_list
        self.num_branches = len(model_list)
        self.num_heads = num_heads
        self.head_dim = branch_dim // num_heads
        self.branch_dim = branch_dim

        # Independent normalization for each branch
        self.branch_norms = nn.ModuleList([nn.LayerNorm(branch_dim) for _ in model_list])

        # Multi-head cross-attention mapping (Query + Entities)
        if self.num_branches > 1:
            self.q_proj = nn.Linear(input_dim, branch_dim, bias=False)
            self.k_proj = nn.ModuleList([nn.Linear(branch_dim, branch_dim, bias=False) for _ in model_list])
            self.v_proj = nn.ModuleList([nn.Linear(branch_dim, branch_dim, bias=False) for _ in model_list])
            self.dropout = nn.Dropout(dropout)

        # Scoring MLP after fusion
        fused_dim = branch_dim * self.num_branches
        fused_dim += input_dim  # 拼入 Node Query

        mlp = []
        for _ in range(num_mlp_layers - 1):
            mlp.append(nn.Linear(fused_dim, fused_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(fused_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, query_rel, branch_features, t_index=None):
        """
        query_rel: (bs, input_dim)
        branch_features: list of (bs, num_nodes, branch_dim)
        t_index: (bs, num_targets)
        """
        bs = branch_features[0].size(0)

        # Gather target features if indices are provided
        if t_index is not None:
            num_targets = t_index.size(1)
            # t_index shape: (bs, num_targets) -> (bs, num_targets, branch_dim)
            gather_index = t_index.unsqueeze(-1).expand(-1, -1, self.branch_dim)
            target_features = [feat.gather(1, gather_index) for feat in branch_features]
        else:
            num_targets = branch_features[0].size(1)
            target_features = branch_features

        normed_features = [norm(feat) for norm, feat in zip(self.branch_norms, target_features)]

        # Bypass attention for single branch models
        if self.num_branches == 1:
            out = normed_features[0]
            node_query = query_rel.unsqueeze(1).expand(-1, num_targets, -1)
            fused_feature = torch.cat([out, node_query], dim=-1)
            score = self.mlp(fused_feature).squeeze(-1)
            return score, None, None

        # Multi-head cross-attention mechanism
        q_flat = query_rel.unsqueeze(1).expand(-1, num_targets, -1).reshape(bs * num_targets, -1)
        Q = self.q_proj(q_flat).view(-1, 1, self.num_heads, self.head_dim).transpose(1, 2)

        K_list = [proj(feat.view(bs * num_targets, -1)) for proj, feat in zip(self.k_proj, normed_features)]
        V_list = [proj(feat.view(bs * num_targets, -1)) for proj, feat in zip(self.v_proj, normed_features)]

        K_stack = torch.stack(K_list, dim=1)
        V_stack = torch.stack(V_list, dim=1)

        K = K_stack.view(-1, self.num_branches, self.num_heads, self.head_dim).transpose(1, 2)
        V = V_stack.view(-1, self.num_branches, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)  # (bs*num_targets, num_heads, 1, num_branches)

        # Entropy and branch usage for monitoring/regularization
        p = attn.clamp(min=1e-8)
        att_entropy = (-p * p.log()).sum(dim=-1).mean()
        att_mean_per_branch = attn.mean(dim=(0, 1, 2))

        attn = self.dropout(attn)

        # Feature fusion
        attn_squeezed = attn.squeeze(2).transpose(1, 2)  # (bs*num_targets, num_branches, num_heads)
        V_reshaped = V.transpose(1, 2)  # (bs*num_targets, num_branches, num_heads, head_dim)

        weighted_branches = []
        for i in range(self.num_branches):
            w = attn_squeezed[:, i, :].unsqueeze(-1)
            b_feat = V_reshaped[:, i, :, :]
            weighted_b = (w * b_feat).reshape(bs * num_targets, -1)
            weighted_branches.append(weighted_b)

        out = torch.cat(weighted_branches, dim=-1).view(bs, num_targets, -1)

        # Residual concatenation and final scoring
        node_query = query_rel.unsqueeze(1).expand(-1, num_targets, -1)
        fused_feature = torch.cat([out, node_query], dim=-1)

        score = self.mlp(fused_feature).squeeze(-1)  # shape: (bs, num_targets)

        return score, att_entropy, att_mean_per_branch


class Gamma(BaseNBFNet):
    def __init__(self, rel_model_cfg, entity_model_cfg, num_relation=1,
                 entropy_reg_weight: float = 0.003, attn_dropout: float = 0.1):

        entity_model_cfg['num_relation'] = num_relation
        super(Gamma, self).__init__(**entity_model_cfg)

        self.num_layers = len(self.dims) - 1
        self.model_list = entity_model_cfg['model_list']
        self.entropy_reg_weight = float(entropy_reg_weight)

        # Structured Relative Position Encoding (PE)
        self.max_dist = self.num_layers + 1
        self.dist_emb = nn.Embedding(self.max_dist + 1, self.dims[0])
        nn.init.xavier_uniform_(self.dist_emb.weight)

        # Dynamic PE Gating based on relation and frequency
        self.pe_gate_proj = nn.Linear(self.dims[0] + 1, 1)

        # Bias initialization to suppress noise initially
        nn.init.constant_(self.pe_gate_proj.bias, -2.0)
        nn.init.xavier_uniform_(self.pe_gate_proj.weight)

        self.pe_gate_temp = nn.Parameter(torch.tensor(1.0))

        # Topology-aware Relation Convolution
        self.rel_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.rel_layers.append(
                layers.TopologyAttentionConv(
                    in_dim=rel_model_cfg['hidden_dim'],
                    out_dim=rel_model_cfg['hidden_dim'],
                    dropout=attn_dropout
                )
            )

        # Entity-Relation Co-evolution modules
        self.co_evolution_layers = nn.ModuleList()
        for i in range(self.num_layers):
            context_dim = self.dims[i + 1] * len(self.model_list)
            self.co_evolution_layers.append(
                layers.CoEvolutionCrossAttention(
                    rel_dim=rel_model_cfg['hidden_dim'],
                    ent_dim=context_dim,
                    dropout=attn_dropout
                )
            )

        # Entity Graph dual-branch evolution network
        self.entity_branches = nn.ModuleDict()
        for msg in self.model_list:
            stack = nn.ModuleList()
            for i in range(self.num_layers):
                stack.append(
                    layers.GeneralizedRelationalConv(
                        self.dims[i], self.dims[i + 1],
                        num_relation=1, query_input_dim=self.dims[0],
                        message_func=msg, aggregate_func=self.aggregate_func,
                        layer_norm=self.layer_norm, activation=self.activation,
                        dependent=False, project_relations=True
                    )
                )
            self.entity_branches[msg] = stack

        # Fusion layer
        self.branch_feature_dim = sum(self.dims[1:]) if self.concat_hidden else self.dims[-1]

        self.fusion = QueryGuidedFusion(
            model_list=self.model_list,
            input_dim=self.dims[0],
            branch_dim=self.branch_feature_dim,
            num_heads=4,
            dropout=attn_dropout,
            num_mlp_layers=getattr(self, 'num_mlp_layers', 2)
        )

    def compute_batched_spd(self, h_index, edge_index, num_nodes):
        """ Parallel BFS to compute Shortest Path Distance from query nodes """
        bs = h_index.size(0)
        device = h_index.device
        dist = torch.full((bs, num_nodes), self.max_dist, dtype=torch.long, device=device)
        dist[torch.arange(bs, device=device), h_index[:, 0]] = 0

        active = torch.zeros((bs, num_nodes), dtype=torch.float, device=device)
        active[torch.arange(bs, device=device), h_index[:, 0]] = 1.0

        row, col = edge_index

        for d in range(1, self.num_layers + 1):
            msg = active[:, row]
            next_active = scatter(msg, col.unsqueeze(0).expand(bs, -1), dim=1, dim_size=num_nodes, reduce="max")
            new_reached = (next_active > 0.5) & (dist == self.max_dist)
            dist[new_reached] = d
            active = next_active
        return dist

    def forward(self, data, batch):
        h_index, t_index, r_index = batch.unbind(-1)
        device = h_index.device

        if self.training:
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index, num_direct_rel=data.num_relations // 2
        )
        bs = h_index.size(0)

        # Initial signals
        query_rels_init = torch.ones(bs, self.dims[0], device=device, dtype=torch.float)

        rel_feat_batch = torch.zeros(bs, data.num_relations, self.dims[0], device=device)
        rel_feat_batch.scatter_add_(1, r_index[:, [0]].unsqueeze(-1).expand(-1, 1, self.dims[0]),
                                    query_rels_init.unsqueeze(1))

        query_rels = rel_feat_batch[torch.arange(bs, device=device), r_index[:, 0]]

        ent_boundaries = {}
        current_ent_feat = {}

        # Entity boundaries with PE injection
        ent_boundary_shared = torch.zeros(bs, data.num_nodes, self.dims[0], device=device)
        ent_boundary_shared.scatter_add_(1, h_index[:, [0]].unsqueeze(-1).expand(-1, 1, self.dims[0]),
                                         query_rels_init.unsqueeze(1))

        for msg in self.model_list:
            ent_boundaries[msg] = ent_boundary_shared
            current_ent_feat[msg] = ent_boundary_shared

        # Position Encoding & Gating
        distances = self.compute_batched_spd(h_index, data.edge_index, data.num_nodes)
        pe_feat = self.dist_emb(distances)

        current_rels = r_index[:, 0]
        total_edges = data.edge_index.size(1)

        if isinstance(data.num_relations, torch.Tensor):
            num_rels_int = int(data.num_relations.item())
        else:
            num_rels_int = int(data.num_relations)

        rel_counts = torch.bincount(data.edge_type, minlength=num_rels_int)
        query_rel_counts = rel_counts[current_rels].float()

        rel_freq = (query_rel_counts / total_edges).unsqueeze(-1)  # shape: (bs, 1)

        gate_input = torch.cat([query_rels, rel_freq], dim=-1)
        gate_logits = self.pe_gate_proj(gate_input)
        scaled_logits = gate_logits / self.pe_gate_temp.clamp(min=0.01)

        pe_gate_weight = torch.sigmoid(scaled_logits).unsqueeze(-1)
        pe_feat = pe_feat * pe_gate_weight

        for msg in self.model_list:
            ent_boundaries[msg] = ent_boundaries[msg] + pe_feat
            current_ent_feat[msg] = current_ent_feat[msg] + pe_feat

        branch_hiddens_history = {msg: [] for msg in self.model_list}
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=device)
        rel_edge_weight = None

        # Layer-wise Co-evolution
        for i in range(self.num_layers):
            # A. Entity propagation
            next_ent_feat = {}
            for msg in self.model_list:
                layer = self.entity_branches[msg][i]
                layer.relation = rel_feat_batch

                # Graph Convolution Output
                hidden = layer(
                    current_ent_feat[msg], query_rels, ent_boundaries[msg],
                    data.edge_index, data.edge_type, size, edge_weight
                )

                is_shape_match = (hidden.shape == current_ent_feat[msg].shape)
                if self.short_cut and is_shape_match:
                    hidden = hidden + current_ent_feat[msg]

                next_ent_feat[msg] = hidden
                branch_hiddens_history[msg].append(hidden)

            current_ent_feat = next_ent_feat

            # B. PNA-style global context pooling (bs, num_nodes, context_dim)
            ent_context = torch.cat([current_ent_feat[msg] for msg in self.model_list], dim=-1)

            ent_mean = ent_context.mean(dim=1, keepdim=True)
            ent_max = ent_context.max(dim=1, keepdim=True)[0]
            ent_min = ent_context.min(dim=1, keepdim=True)[0]
            ent_std = ent_context.std(dim=1, keepdim=True, unbiased=False)

            global_ent_context = torch.cat([ent_mean, ent_max, ent_min, ent_std], dim=1)  # Shape: (bs, 4, dim)

            # C. Relation propagation and D. Cross-tuning
            rel_feat_batch = self.rel_layers[i](
                rel_feat_batch, data.relation_graph.edge_index, data.relation_graph.edge_type, rel_edge_weight
            )

            rel_feat_batch = self.co_evolution_layers[i](rel_feat_batch, global_ent_context)

        # Final pooling and scoring
        final_branch_features = []
        for msg in self.model_list:
            if self.concat_hidden:
                feat = torch.cat(branch_hiddens_history[msg], dim=-1)
            else:
                feat = branch_hiddens_history[msg][-1]
            final_branch_features.append(feat)

        updated_query_rels = rel_feat_batch[torch.arange(bs, device=device), r_index[:, 0]]

        score, att_entropy, att_mean_per_branch = self.fusion(
            updated_query_rels, final_branch_features, t_index=t_index
        )

        score = score.view(shape)

        if self.training:
            aux = {}
            if att_entropy is not None:
                aux["att_entropy"] = att_entropy.detach()
                if self.entropy_reg_weight > 0:
                    aux["aux_loss"] = -self.entropy_reg_weight * att_entropy

            if att_mean_per_branch is not None:
                aux["att_mean_per_branch"] = att_mean_per_branch.detach()

            if "aux_loss" not in aux:
                aux["aux_loss"] = None

            return score, aux

        return score