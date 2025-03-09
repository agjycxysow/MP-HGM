import torch
import torch.nn as nn 
from torch_geometric.nn import HeteroConv, Linear

class HeteroRGCN(torch.nn.Module):
    def __init__(self, node_types, hidden_dim, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                rel: torch.nn.Sequential(
                    Linear(-1, hidden_dim),
                    torch.nn.ReLU()
                ) for rel in node_types
            })
            self.convs.append(conv)
            
    def forward(self, graph):
        x_dict = graph.x_dict
        edge_index_dict = graph.edge_index_dict
        
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            
        return x_dict


class MetaPathEncoder(torch.nn.Module):
    def __init__(self, input_dim, n_heads, metapaths):
        super().__init__()
        self.metapaths = metapaths
        self.path_attn = torch.nn.MultiheadAttention(input_dim, n_heads)
        self.node_attn = torch.nn.ModuleDict({
            mp: torch.nn.MultiheadAttention(input_dim, n_heads) 
            for mp in metapaths
        })
        
    def forward(self, emb_dict):
        all_paths = [self._encode_path(emb_dict, mp) for mp in self.metapaths]
        global_weights, _ = self.path_attn(all_paths, all_paths, all_paths)
        
        weighted_embs = []
        for i, mp in enumerate(self.metapaths):
            local_emb, _ = self.node_attn[mp](all_paths[i], all_paths[i], all_paths[i])
            weighted_embs.append(global_weights[i] * local_emb)
            
        return torch.cat(weighted_embs, dim=-1)  


class CrossGraphMatcher(nn.Module):
    def __init__(self, feat_dim, use_vit=True, vit_depth=4):
        super().__init__()
        self.type_proj = nn.ModuleDict({
            'intra': nn.Linear(feat_dim, feat_dim),
            'inter': nn.Linear(feat_dim, feat_dim)
        })
        
        if use_vit:
            self.vit = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=feat_dim,
                    nhead=8,
                    dim_feedforward=512
                ),
                num_layers=vit_depth
            )
        
        self.gate = nn.Sequential(
            nn.Linear(feat_dim*2, 1),
            nn.Sigmoid()
        )
    def build_interaction_matrix(self, emb1, emb2):
        intra_sim = torch.matmul(
            self.type_proj['intra'](emb1),
            self.type_proj['intra'](emb2).T
        )
        inter_sim = torch.matmul(
            self.type_proj['inter'](emb1),
            self.type_proj['inter'](emb2).T
        )
        return intra_sim + inter_sim
    def forward(self, emb1, emb2):
        sim_matrix = self.build_interaction_matrix(emb1, emb2)
        
        if hasattr(self, 'vit'):
            b, h, w = sim_matrix.shape
            vit_input = sim_matrix.view(b, h*w, -1)
            global_feat = self.vit(vit_input).mean(dim=1)
        else:
            global_feat = sim_matrix.flatten(1)
        
        gate = self.gate(torch.cat([emb1, emb2], dim=-1))
        final_sim = gate * global_feat + (1-gate) * sim_matrix.mean(dim=-1)
        return final_sim