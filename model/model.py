import torch
from model.modules import HeteroRGCN, MetaPathEncoder, CrossGraphMatcher

class MPHGM(torch.nn.Module):
    def __init__(self, config, node_types, metapaths):
        super().__init__()

        self.encoder = HeteroRGCN(
            node_types=node_types,
            hidden_dim=config.hidden_dim,
            num_layers=config.rgcn_layers
        )
        
        self.path_encoder = MetaPathEncoder(
            input_dim=config.hidden_dim,
            n_heads=config.n_heads,
            metapaths=metapaths
        )
        
        self.matcher = CrossGraphMatcher(
            feat_dim=config.hidden_dim * config.n_heads,
            use_vit=config.vit_layers > 0,
            vit_depth=config.vit_layers
        )

    def forward(self, graph1, graph2):
        emb1 = self.encoder(graph1)
        emb2 = self.encoder(graph2)
        
        path_emb1 = self.path_encoder(emb1)
        path_emb2 = self.path_encoder(emb2)
        
        similarity = self.matcher(path_emb1, path_emb2)
        return similarity



