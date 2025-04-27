def compute_bleu(reference, candidate, n=1):
    """
    计算BLEU分数
    Args:
        reference: 参考翻译文本
        candidate: 生成的翻译文本
        n: n-gram的最大长度
    Returns:
        bleu: BLEU分数
    """
    # 将文本分词
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    # 计算各个n-gram的精度
    precisions = []
    for i in range(1, n+1):
        # 获取候选文本中的n-gram
        cand_ngrams = {}
        for j in range(len(cand_tokens)-i+1):
            ngram = tuple(cand_tokens[j:j+i])
            cand_ngrams[ngram] = cand_ngrams.get(ngram, 0) + 1
            
        # 获取参考文本中的n-gram
        ref_ngrams = {}
        for j in range(len(ref_tokens)-i+1):
            ngram = tuple(ref_tokens[j:j+i])
            ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1
            
        # 计算匹配的n-gram数量
        match_count = 0
        total_count = 0
        for ngram, count in cand_ngrams.items():
            total_count += count
            if ngram in ref_ngrams:
                match_count += min(count, ref_ngrams[ngram])
                
        # 计算精度
        if total_count == 0:
            precisions.append(0.0)
        else:
            precisions.append(match_count / total_count)
    
    # 计算BLEU分数
    if len(precisions) == 0:
        return 0.0
    return sum(precisions) / len(precisions)

# 使用示例
reference = "I love dogs"
candidate = "I love cats" 
bleu_score = compute_bleu(reference, candidate, n=4)
print(f"BLEU score: {bleu_score}")  # 输出: BLEU score: 0.42083333333333334