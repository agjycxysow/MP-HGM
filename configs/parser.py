import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SiameseRGCN Configuration")
    
    # --------------------------
    # 模型架构参数
    # --------------------------
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--rgcn_layers", type=int, default=3)
    model_group.add_argument("--hidden_dim", type=int, default=128)
    model_group.add_argument("--vit_layers", type=int, default=2)
    model_group.add_argument("--n_heads", type=int, default=8)
    model_group.add_argument("--proj_dim", type=int, default=128)

    # --------------------------
    # 训练超参数
    # --------------------------
    train_group = parser.add_argument_group("Training Setup")
    train_group.add_argument("--lr", type=float, default=5e-4)
    train_group.add_argument("--weight_decay", type=float, default=1e-4)
    train_group.add_argument("--batch_size", type=int, default=32,)
    train_group.add_argument("--epochs", type=int, default=200)
    train_group.add_argument("--patience", type=int, default=10)

    # --------------------------
    # 数据集参数
    # --------------------------
    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument("--dataset", type=str, default="ACM",
                         choices=["ACM", "DBLP", "IMDB"])
    data_group.add_argument("--metapaths", nargs='+', 
                         default=["APA", "APVPA", "APTPA"])

    # --------------------------
    # 系统参数
    # --------------------------
    system_group = parser.add_argument_group("System")
    system_group.add_argument("--seed", type=int, default=42)
    system_group.add_argument("--cuda", type=int, default=0)

    return parser.parse_args()
