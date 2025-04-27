import torch
from transformers import AutoTokenizer, AutoModel

def bert_score(candidate, reference):
    # 加载预训练的模型和分词器
    tokenizer = AutoTokenizer.from_pretrained('model-name')
    model = AutoModel.from_pretrained('model-name')
    
    # 对文本进行分词和编码
    candidate_tokens = tokenizer(candidate, return_tensors='pt', padding=True, truncation=True)
    reference_tokens = tokenizer(reference, return_tensors='pt', padding=True, truncation=True)
    
    # 获取BERT词嵌入向量，并进行归一化
    with torch.no_grad():
        candidate_embeddings = model(**candidate_tokens).last_hidden_state.squeeze(0)
        reference_embeddings = model(**reference_tokens).last_hidden_state.squeeze(0)
        
        # L2归一化
        candidate_embeddings = torch.nn.functional.normalize(candidate_embeddings, p=2, dim=-1)
        reference_embeddings = torch.nn.functional.normalize(reference_embeddings, p=2, dim=-1)
    
    # 计算余弦相似度矩阵
    sim_matrix = torch.matmul(candidate_embeddings, reference_embeddings.transpose(0, 1))
    
    # 忽略[PAD]token的影响
    candidate_mask = candidate_tokens.attention_mask.squeeze(0).bool()
    reference_mask = reference_tokens.attention_mask.squeeze(0).bool()
    
    # 计算Precision (只考虑非padding的token)
    P = sim_matrix[candidate_mask].max(dim=1)[0].mean()
    
    # 计算Recall (只考虑非padding的token)
    R = sim_matrix[:, reference_mask].max(dim=0)[0].mean()
    
    # 计算F1分数
    F1 = 2 * (P * R) / (P + R)
    
    return {
        'precision': P.item(),
        'recall': R.item(),
        'f1': F1.item()
    }

# 使用示例
candidate = "今天天气真不错"
reference = "今天是个好天气"

scores = bert_score(candidate, reference)
print(f"BERTScore评分结果:")
print(f"Precision: {scores['precision']:.3f}")
print(f"Recall: {scores['recall']:.3f}") 
print(f"F1: {scores['f1']:.3f}")