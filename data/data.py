from torch_geometric.data import HeteroData
import numpy as np
import torch 

class HeteroGraphPairDataset:
    def __init__(self, graph_list, metapaths):
        self.graph_pairs = graph_list  
        self.metapaths = metapaths
        
    def __len__(self):
        return len(self.graph_pairs)
    
    def __getitem__(self, idx):
        pair = self.graph_pairs[idx]
        return {
            'g1': self._to_heterodata(pair['g1']),
            'g2': self._to_heterodata(pair['g2']),
            'label': pair['label']
        }
    
    def _to_heterodata(self, raw_graph):
        data = HeteroData()
        
        for nt in raw_graph['node_types']:
            data[nt].x = torch.FloatTensor(raw_graph['node_feats'][nt])
        
        for et in raw_graph['edge_types']:
            data[et].edge_index = torch.LongTensor(raw_graph['edges'][et])
        
        return data
