import torch
from model.model import MPHGM
from data.data import HeteroGraphPairDataset
from torch_geometric.loader import DataLoader
from configs.parser import parse_args

def train():
    args = parse_args()
    
    model = MPHGM(
        config=args,
        node_types=['user', 'item', 'tag'],
        metapaths=args.metapaths
    )
    
    graph_list = [] 
    metapaths = []  
    dataset = HeteroGraphPairDataset(graph_list, metapaths)
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CosineEmbeddingLoss()
    
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            g1, g2 = batch['g1'], batch['g2']
            labels = batch['label']
            
            sim_scores = model(g1, g2)
            loss = criterion(sim_scores, torch.ones_like(sim_scores), labels)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train()
