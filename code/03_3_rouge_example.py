def rouge_n(reference, candidate, n):
    """
    计算ROUGE-N分数
    
    Args:
        reference: 参考文本(字符串)
        candidate: 生成文本(字符串) 
        n: n-gram的n值
    
    Returns:
        float: ROUGE-N分数
    """
    # 将文本分词
    ref_tokens = reference.split()
    can_tokens = candidate.split()
    
    # 获取参考文本中的n-gram
    ref_ngrams = {}
    for i in range(len(ref_tokens) - n + 1):
        ngram = tuple(ref_tokens[i:i+n])
        ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1
        
    # 获取生成文本中的n-gram
    can_ngrams = {}
    for i in range(len(can_tokens) - n + 1):
        ngram = tuple(can_tokens[i:i+n])
        can_ngrams[ngram] = can_ngrams.get(ngram, 0) + 1
    
    # 计算匹配的n-gram数量
    match_count = 0
    for ngram, count in can_ngrams.items():
        match_count += min(count, ref_ngrams.get(ngram, 0))
    
    # 计算参考文本中n-gram总数
    total_ref_ngrams = sum(ref_ngrams.values())
    
    # 避免除零错误
    if total_ref_ngrams == 0:
        return 0.0
        
    # 计算ROUGE-N分数
    rouge_n_score = match_count / total_ref_ngrams
    
    return rouge_n_score

# 使用示例
reference = "the cat sat on the mat"
candidate = "the cat was on the mat"
rouge_1 = rouge_n(reference, candidate, 1)
rouge_2 = rouge_n(reference, candidate, 2)

print(f"ROUGE-1分数: {rouge_1:.3f}")  # 输出: ROUGE-1分数: 0.833
print(f"ROUGE-2分数: {rouge_2:.3f}")  # 输出: ROUGE-2分数: 0.600
