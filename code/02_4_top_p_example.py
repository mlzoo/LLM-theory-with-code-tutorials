import torch

def top_p_sampling(probs, p):
    """Top-P 采样
    
    Args:
        probs: 概率分布 (torch.Tensor)
        p: 阈值
    """
    # 对概率分布进行降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    
    # 找到累积概率大于等于p的词的索引
    # nonzero返回满足条件的索引
    # squeeze去掉维度为1的维度
    p_idx = torch.nonzero(cumulative_probs >= p)[0].item()
    
    # 获取前p_idx+1个词作为候选集
    candidate_probs = sorted_probs[:p_idx+1]
    
    # 归一化概率
    candidate_probs = candidate_probs / candidate_probs.sum()
    
    # 从候选集中采样
    # multinomial进行多项分布采样
    next_word_idx = torch.multinomial(candidate_probs, num_samples=1).item()
    
    # 返回原始词表中的索引
    return sorted_indices[next_word_idx].item()