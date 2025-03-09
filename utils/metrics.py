import torch

class GraphMatchEvaluator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.pred_scores = []
        self.true_labels = []
        self.rank_pool = []
    
    def update(self, preds, targets):
        """更新批次数据
        Args:
            preds: 预测相似度矩阵 [batch, num_candidates]
            targets: 真实标签矩阵 [batch, num_candidates] (连续值)
                    或索引标签 [batch] (离散索引)
        """
        self.pred_scores.append(preds.detach())
        
        if targets.dim() == 1:
            batch_size, num_c = preds.shape
            one_hot = torch.zeros_like(preds)
            one_hot[torch.arange(batch_size), targets] = 1.0
            self.true_labels.append(one_hot)
        else:
            self.true_labels.append(targets.float())

    def compute(self):
        preds = torch.cat(self.pred_scores)
        trues = torch.cat(self.true_labels)
        
        preds = preds.float()
        trues = trues.float()

        return {
            "MSE": self._compute_mse(preds, trues),
            "Spearman": self._compute_spearman(preds, trues),
            "Precision@10": self._compute_precision_at_k(preds, trues, k=10)
        }

    def _compute_mse(self, preds, trues):
        return torch.mean((preds - trues) ** 2).item()

    def _compute_spearman(self, preds, trues):
        pred_rank = torch.argsort(torch.argsort(preds, dim=1, descending=True))
        true_rank = torch.argsort(torch.argsort(trues, dim=1, descending=True))
        
        pred_centered = pred_rank - pred_rank.mean(dim=1, keepdim=True)
        true_centered = true_rank - true_rank.mean(dim=1, keepdim=True)
        
        cov = (pred_centered * true_centered).sum(dim=1)
        std = torch.sqrt(pred_centered.pow(2).sum(dim=1)) * \
              torch.sqrt(true_centered.pow(2).sum(dim=1))
        
        spearman = cov / (std + 1e-6)
        return torch.mean(torch.nan_to_num(spearman)).item()

    def _compute_precision_at_k(self, preds, trues, k=10):
        if trues.max() > 1.0:
            trues = (trues > trues.median(dim=1, keepdim=True).values).float()
        
        _, topk_idx = torch.topk(preds, k=k, dim=1)
        batch_range = torch.arange(preds.size(0))[:, None]
        
        hits = trues[batch_range, topk_idx].sum(dim=1)
        return (hits.float().mean() / k).item()
