import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def calculate_perplexity(model, test_text, tokenizer, device='cuda'):
    """计算语言模型在测试文本上的困惑度
    
    Args:
        model: 语言模型
        test_text: 测试文本
        tokenizer: 分词器
        device: 运行设备
        
    Returns:
        float: 困惑度值
    """
    model.eval()
    
    # 对测试文本进行编码
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    
    with torch.no_grad():
        # 获取模型输出概率分布
        outputs = model(input_ids)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # 获取每个位置的预测概率，并直接取对数
        # log_softmax(x) = log(softmax(x))
        probs = F.log_softmax(logits, dim=-1) 
        
        # 获取真实token的概率
        shift_probs = probs[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # 获取每个位置对应的真实token概率
        # 1. shift_probs的形状为[batch_size, seq_len-1, vocab_size]，表示每个位置上所有词的预测概率
        # 2. shift_labels的形状为[batch_size, seq_len-1]，表示每个位置的真实token索引
        # 3. 使用torch.gather函数从预测概率中提取真实token的概率:
        #    - input: shift_probs，预测概率矩阵
        #    - dim=2，表示在vocab_size维度上收集
        #    - index: shift_labels.unsqueeze(-1)，将真实token索引扩展一维以匹配输入形状
        # 4. 示例说明torch.gather的工作原理:
        #    输入tensor:               索引tensor:
        #    [[1, 2, 3],              [[0],
        #     [4, 5, 6],     dim=1     [1], 
        #     [7, 8, 9]]               [2]]
        #    结果: [[1],    # 第0行取第0个元素
        #          [5],     # 第1行取第1个元素  
        #          [9]]     # 第2行取第2个元素
        # 5. 最后用squeeze(-1)去掉收集后多余的维度，得到形状[batch_size, seq_len-1]
        token_probs = torch.gather(
            shift_probs, 
            2, 
            shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 计算序列长度(去掉padding)
        # 1. input_ids != tokenizer.pad_token_id 会生成一个形状为[batch_size, seq_len]的布尔张量
        # 2. 使用sum()方法计算每个序列中非padding token的数量
        # 3. 减去1是因为序列长度不包括起始token
        seq_length = (input_ids != tokenizer.pad_token_id).sum().item() - 1
        
        # 计算困惑度
        ppl = torch.exp(-token_probs.sum() / seq_length)
        
    return ppl.item()

# 使用示例
"""
tokenizer = AutoTokenizer.from_pretrained('model-name')
model = AutoModelForCausalLM.from_pretrained('model-name')

test_text = "今天天气真不错"
ppl = calculate_perplexity(model, test_text, tokenizer)
print(f"困惑度: {ppl}")
"""