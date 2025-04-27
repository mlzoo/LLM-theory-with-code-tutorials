import torch
import torch.nn as nn
import math

class RWKVBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 时间混合模块参数
        self.time_decay = nn.Parameter(torch.zeros(hidden_size))  # w参数
        self.time_first = nn.Parameter(torch.zeros(hidden_size))  # u参数
        
        # 时间混合的线性变换
        self.time_mix_r = nn.Linear(hidden_size, hidden_size)
        self.time_mix_k = nn.Linear(hidden_size, hidden_size)
        self.time_mix_v = nn.Linear(hidden_size, hidden_size)
        
        # 通道混合模块参数
        self.channel_mix_r = nn.Linear(hidden_size, hidden_size)
        self.channel_mix_k = nn.Linear(hidden_size, hidden_size)
        
    def time_mixing(self, x, state=None):
        # 生成R、K、V向量
        r = self.time_mix_r(x)
        k = self.time_mix_k(x)
        v = self.time_mix_v(x)
        
        # 计算时间衰减
        time_decay = torch.exp(-torch.exp(self.time_decay))
        k = torch.exp(k)
        
        if state is not None:
            # 完整的WKV计算
            # 当前时刻的贡献
            current_contribution = torch.exp(self.time_first + k) * v
            
            # 历史信息的贡献（使用累积状态）
            numerator = state['num'] * time_decay + current_contribution
            denominator = state['den'] * time_decay + torch.exp(self.time_first + k)
            
            # 计算WKV
            wkv = numerator / (denominator + 1e-6)  # 添加小值避免除零
            
            # 更新状态
            new_state = {
                'num': numerator,
                'den': denominator
            }
        else:
            # 首个时间步的处理
            wkv = v
            new_state = {
                'num': torch.exp(k) * v,
                'den': torch.exp(k)
            }
        
        # 使用接收向量R进行门控
        output = torch.sigmoid(r) * wkv
        return output, new_state
    
    def channel_mixing(self, x):
        # 通道混合的实现
        r = self.channel_mix_r(x)
        k = self.channel_mix_k(x)
        
        # 使用接收向量进行门控
        return torch.sigmoid(r) * k
    
    def forward(self, x, state=None):
        # 时间混合
        time_mix_out, new_state = self.time_mixing(x, state)
        
        # 通道混合
        channel_mix_out = self.channel_mixing(x)
        
        # 组合输出
        out = time_mix_out + channel_mix_out
        
        return out, new_state

# 使用示例
def test_rwkv():
    hidden_size = 512
    rwkv_block = RWKVBlock(hidden_size)
    
    # 模拟输入序列
    batch_size = 1
    seq_len = 10
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # 初始状态为None
    state = None
    
    # 按序列逐步处理
    outputs = []
    for t in range(seq_len):
        out, state = rwkv_block(x[:, t:t+1, :], state)
        outputs.append(out)
    
    # 合并所有输出
    final_output = torch.cat(outputs, dim=1)
    print(f"输出张量形状: {final_output.shape}")
    
    return final_output

if __name__ == "__main__":
    test_rwkv()