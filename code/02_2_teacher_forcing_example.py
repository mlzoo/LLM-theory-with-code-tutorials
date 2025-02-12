import torch
from torch.utils.data import Dataset

def prepare_training_data(text, tokenizer, max_length):
    """准备训练数据,将文本处理成teacher forcing格式
    
    Args:
        text: 原始文本,如"Ming had a new laptop"
        tokenizer: 分词器
        max_length: 最大序列长度
    
    Returns:
        input_ids: 输入序列列表,如:
                  ["Ming"]
                  ["Ming", "had"]
                  ["Ming", "had", "a"]
                  ["Ming", "had", "a", "new"]
                  ...
        target_ids: 对应的目标token
    """
    # 对文本分词
    tokens = tokenizer.encode(text)

    # 如果文本长度超过最大长度,截断
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    # 初始化输入序列和目标序列
    input_sequences = []
    target_sequences = []
    
    # 生成teacher forcing训练数据，从1开始，因为第一个词是输入，不需要预测
    for i in range(1, len(tokens)):
        # 获取输入序列和目标序列
        input_seq = tokens[:i]
        target = tokens[i]
        
        # padding
        padding_length = max_length - len(input_seq)
        # 如果padding长度大于0，则进行padding
        if padding_length > 0:
            input_seq = input_seq + [tokenizer.pad_token_id] * padding_length
        
        # 添加到输入序列和目标序列中
        input_sequences.append(input_seq)
        target_sequences.append(target)
        
    return input_sequences, target_sequences

# 示例：TextDataset在返回数据时，会调用prepare_training_data函数，将文本处理成teacher forcing格式
class TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids, target_ids = prepare_training_data(text, self.tokenizer, self.max_length)
        return {
            'input_ids': torch.tensor(input_ids),
            'target_ids': torch.tensor(target_ids)
        }