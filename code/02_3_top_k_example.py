import torch

def top_k_sampling(probs, k):
    """Top-K 采样
    
    Args:
        probs: 概率分布 (torch.Tensor)
        k: 候选词数量
    """
    # 获取前k个最大概率的值和索引
    top_k_probs, top_k_idx = torch.topk(probs, k)
    
    # 归一化概率
    top_k_probs = torch.softmax(top_k_probs, dim=-1)
    
    # 从top k中采样
    next_word_idx = torch.multinomial(top_k_probs, num_samples=1)
    
    # 获取实际的词表索引
    next_word_idx = top_k_idx[next_word_idx]
    
    return next_word_idx.item()