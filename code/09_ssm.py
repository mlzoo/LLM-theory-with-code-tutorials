import torch
import torch.nn as nn

class SSM(nn.Module):
    def __init__(self, input_dim, state_dim, output_dim):
        """
        初始化SSM模型
        Args:
            input_dim: 输入维度
            state_dim: 隐状态维度 
            output_dim: 输出维度
        """
        super().__init__()
        # 初始化状态空间参数 A[state_dim, state_dim], B[state_dim, input_dim], C[output_dim, state_dim]
        self.A = nn.Parameter(torch.randn(state_dim, state_dim))  # 状态转移矩阵
        self.B = nn.Parameter(torch.randn(state_dim, input_dim))  # 输入矩阵
        self.C = nn.Parameter(torch.randn(output_dim, state_dim)) # 输出矩阵
        self.state_dim = state_dim
        
    def forward_recursive(self, x):
        """
        递归形式的前向传播
        实现公式:
        h_t = Ah_{t-1} + Bx_t
        y_t = Ch_t
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
        Returns:
            输出张量 [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        
        # 初始化隐状态 h_0 = 0
        # h shape: [batch_size, state_dim]
        h = torch.zeros(batch_size, self.state_dim, device=device)
        outputs = []
        
        # 按时间步迭代
        for t in range(seq_len):
            # x_t shape: [batch_size, input_dim]
            x_t = x[:, t, :]
            
            # 更新隐状态: h_t = Ah_{t-1} + Bx_t
            # h shape: [batch_size, state_dim]
            h = torch.matmul(h, self.A.T) + torch.matmul(x_t, self.B.T)
            
            # 计算输出: y_t = Ch_t
            # y_t shape: [batch_size, output_dim]
            y_t = torch.matmul(h, self.C.T)
            outputs.append(y_t)
            
        # 最终输出 shape: [batch_size, seq_len, output_dim]
        return torch.stack(outputs, dim=1)
    
    def forward_convolutional(self, x):
        """
        卷积形式的前向传播
        实现公式:
        y_k = \sum_{i=0}^{k-1}(\overline{\mathbf{C}}\overline{\mathbf{A}}^i\overline{\mathbf{B}})u_{k-i}
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
        Returns:
            输出张量 [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        
        # 构建卷积核 K = (CB, CAB, CA^2B, ..., CA^{k-1}B)
        kernel = []
        A_power = torch.eye(self.state_dim, device=device)  # 初始为A^0
        for t in range(seq_len):
            # k_t shape: [output_dim, input_dim]
            k_t = torch.matmul(torch.matmul(self.C, A_power), self.B)
            kernel.append(k_t)
            A_power = torch.matmul(A_power, self.A)  # 计算下一个A幂次
        
        # kernel shape: [seq_len, output_dim, input_dim]
        kernel = torch.stack(kernel, dim=0)
        
        # 转换输入shape以便进行卷积运算
        # x_reshaped shape: [batch_size, input_dim, seq_len]
        x_reshaped = x.transpose(1, 2)
        
        # 执行卷积运算
        y = []
        for i in range(batch_size):
            # y_i shape: [seq_len, output_dim]
            y_i = torch.zeros(seq_len, self.C.shape[0], device=device)
            for t in range(seq_len):
                for tau in range(min(t + 1, seq_len)):
                    y_i[t] += torch.matmul(kernel[tau], x_reshaped[i, :, t-tau])
            y.append(y_i)
            
        # 最终输出 shape: [batch_size, seq_len, output_dim]
        return torch.stack(y, dim=0)
    
    def forward(self, x, mode='recursive'):
        """
        前向传播，支持递归和卷积两种模式
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            mode: 'recursive' 或 'convolutional'
        Returns:
            输出张量 [batch_size, seq_len, output_dim]
        """
        if mode == 'recursive':
            return self.forward_recursive(x)
        elif mode == 'convolutional':
            return self.forward_convolutional(x)
        else:
            raise ValueError(f"不支持的模式: {mode}")

# 使用示例
if __name__ == "__main__":
    # 创建SSM模型实例
    input_dim = 4
    state_dim = 8
    output_dim = 2
    ssm = SSM(input_dim, state_dim, output_dim)
    
    # 生成测试数据
    batch_size = 3
    seq_len = 10
    x = torch.randn(batch_size, seq_len, input_dim)  # [3, 10, 4]
    
    # 使用递归模式
    y_recursive = ssm(x, mode='recursive')  # [3, 10, 2]
    print("递归模式输出形状:", y_recursive.shape)
    
    # 使用卷积模式
    y_conv = ssm(x, mode='convolutional')  # [3, 10, 2]
    print("卷积模式输出形状:", y_conv.shape)