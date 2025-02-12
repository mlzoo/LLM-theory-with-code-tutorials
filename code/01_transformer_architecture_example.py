import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


def get_sinusoid_encoding_table(n_position, d_model):
    """生成正弦位置编码表
    
    Args:
        n_position: 最大序列长度
        d_model: 模型维度
        
    Returns:
        position_enc: [n_position, d_model] 的位置编码表
    """
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]
    
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    
    return torch.FloatTensor(sinusoid_table)


class MultiHeadAttention(nn.Module):
    def __init__(self, args: Dict):
        super().__init__()
        
        # 设置头的数量和每个头的维度
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        # 定义QKV变换矩阵
        self.query_proj = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.key_proj = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.value_proj = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        # Dropout层
        # Attention Dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # Residual Dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        # Dropout
        self.dropout = args.dropout
        
        # KV缓存
        self.key_cache, self.value_cache = None, None

        # 注意力掩码 - 用于确保当前token只能看到之前的token

        # 创建一个形状为(1,1,max_seq_len,max_seq_len)的掩码矩阵
        # 使用float("-inf")填充,这样在softmax后会变成0,实现掩码效果
        # 这个掩码用于确保每个token只能看到它之前的token,不能看到未来的token
        # 形状解释:
        # - 第一维=1: batch维度广播
        # - 第二维=1: 注意力头维度广播 
        # - 第三维=max_seq_len: query序列长度
        # - 第四维=max_seq_len: key序列长度
        max_len = args.max_seq_len
        attn_mask = torch.full((1, 1, max_len, max_len), float("-inf"))
        
        # torch.triu(xxx, diagonal=1) 上三角矩阵, diagonal=1表示从对角线开始,屏蔽对角线及其右边的元素
        # register_buffer("name", tensor, persistent=False) - 注册一个不需要梯度的张量作为模块的缓冲区
        # - 与普通的Parameter不同,buffer不会被优化器更新
        # - 但会被保存到模型的state_dict中,在加载模型时可以恢复
        # - 使用persistent=False表示这个buffer在保存模型时不会被保存
        # - 因为掩码可以在加载模型时重新生成
        # 原始论文中的编码器不需要使用掩码矩阵来屏蔽未来的信息，因为编码器处理的是整个输入序列，每个位置的token可以自由地访问序列中的其他位置。
        # 这里为了方便，就统一使用掩码矩阵来屏蔽未来的信息
        self.register_buffer("attn_mask", torch.triu(attn_mask, diagonal=1), persistent=False)

    def forward(self, x: torch.Tensor, encoder_output: Optional[torch.Tensor] = None):
        """前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, dim]
            encoder_output: 编码器输出，用于交叉注意力。
                          如果为None则为自注意力模式
        """
        batch_size, seq_len, _ = x.shape

        # 生成query向量 - 始终使用输入x
        query = self.query_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        if encoder_output is None:
            # 自注意力模式：key和value来自同一输入x
            key = self.key_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
            value = self.value_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        else:
            # 交叉注意力模式：key和value来自encoder的输出
            key = self.key_proj(encoder_output).view(batch_size, -1, self.n_heads, self.head_dim)
            value = self.value_proj(encoder_output).view(batch_size, -1, self.n_heads, self.head_dim)

        # 维度转置 [batch_size, n_heads, seq_len, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # 注意力计算
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(query, key.transpose(2, 3)) * scale
        
        # 只在自注意力模式下使用注意力掩码
        if encoder_output is None:
            attn_scores = attn_scores + self.attn_mask[:, :, :seq_len, :seq_len]
        
        attn_probs = F.softmax(attn_scores.float(), dim=-1).type_as(query)
        attn_probs = self.attn_dropout(attn_probs)
        
        output = torch.matmul(attn_probs, value)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )
        
        return self.resid_dropout(self.output_proj(output))


class FeedForward(nn.Module):
    """FFN实现"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        
        # 定义两个线性变换层,w1,w2
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 第一层线性变换后接ReLU激活函数
        x = F.relu(self.fc1(x))
        # 第二层线性变换后接dropout
        x = self.dropout(self.fc2(x))
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, args: Dict):
        super().__init__()
        
        # 自注意力层
        self.self_attention = MultiHeadAttention(args)
        # 前馈网络层
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            dropout=args.dropout
        )
        
        self.attention_norm = nn.LayerNorm(args.dim)
        self.ffn_norm = nn.LayerNorm(args.dim)

    def forward(self, x: torch.Tensor):

        # 原文的Pre-LN结构LayerNorm在残差连接之前
        h = x + self.self_attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))

        # # Post-LN结构：先残差连接，再LayerNorm
        # h = self.attention_norm(x + self.self_attention(x))
        # out = self.ffn_norm(h + self.feed_forward(h))
        return out


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, args: Dict):
        super().__init__()
        
        # 词嵌入层
        self.token_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        # 位置编码
        self.register_buffer(
            'pos_embedding',
            get_sinusoid_encoding_table(args.max_seq_len, args.dim)
        )
        
        # 编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(args) for _ in range(args.n_layers)
        ])
        
        # 最后的层归一化
        self.final_norm = nn.LayerNorm(args.dim)
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, tokens: torch.Tensor):
        # 词嵌入 + 位置编码
        h = self.dropout(
            self.token_embeddings(tokens) + self.pos_embedding[:tokens.size(1), :]
        )
        
        # 多层编码器块
        for layer in self.layers:
            h = layer(h)
            
        return self.final_norm(h)


class Transformer(nn.Module):
    """完整的Transformer模型"""
    def __init__(self, args: Dict):
        super().__init__()
        
        # 编码器
        self.encoder = TransformerEncoder(args)
        # 解码器
        self.decoder = TransformerDecoder(args)
        
        # 输出层
        self.output_proj = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor):
        """
        Args:
            src_tokens: 源序列 [batch_size, src_len]
            tgt_tokens: 目标序列 [batch_size, tgt_len]
        """
        # 编码器前向传播
        encoder_out = self.encoder(src_tokens)
        
        # 解码器前向传播
        decoder_out = self.decoder(tgt_tokens, encoder_out)
        
        # 输出层
        logits = self.output_proj(decoder_out)
        
        return logits


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    def __init__(self, args: Dict):
        super().__init__()
        
        # 词嵌入层
        self.token_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        # 位置编码
        self.register_buffer(
            'pos_embedding',
            get_sinusoid_encoding_table(args.max_seq_len, args.dim)
        )
        
        # 解码器层
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(args) for _ in range(args.n_layers)
        ])
        
        # 最后的层归一化
        self.final_norm = nn.LayerNorm(args.dim)
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, tokens: torch.Tensor, encoder_out: torch.Tensor):
        # 词嵌入 + 位置编码
        h = self.dropout(
            self.token_embeddings(tokens) + self.pos_embedding[:tokens.size(1), :]
        )
        
        # 多层解码器块
        for layer in self.layers:
            h = layer(h, encoder_out)
            
        return self.final_norm(h)


class TransformerDecoderBlock(nn.Module):
    """Transformer解码器块 - 使用Post-LN结构"""
    def __init__(self, args: Dict):
        super().__init__()
        
        # 自注意力层
        self.self_attention = MultiHeadAttention(args)
        # 交叉注意力层
        self.cross_attention = MultiHeadAttention(args)
        # 前馈网络层
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            dropout=args.dropout
        )
        
        # Post-LN: LayerNorm在残差连接之后
        self.self_attention_norm = nn.LayerNorm(args.dim)
        self.cross_attention_norm = nn.LayerNorm(args.dim)
        self.ffn_norm = nn.LayerNorm(args.dim)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor):
        # 1. 自注意力层 (带掩码)
        h = self.self_attention_norm(x + self.self_attention(x))
        
        # 2. 交叉注意力层
        # query来自解码器，key和value来自编码器输出
        h = self.cross_attention_norm(h + self.cross_attention(h, encoder_out))
        
        # 3. 前馈网络层
        out = self.ffn_norm(h + self.feed_forward(h))
        return out


# Pre-LN版本的编码器块示例
class TransformerEncoderBlockPreLN(nn.Module):
    """Transformer编码器块 - 使用Pre-LN结构"""
    def __init__(self, args: Dict):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            dropout=args.dropout
        )
        
        # Pre-LN: LayerNorm在残差连接之前
        self.attention_norm = nn.LayerNorm(args.dim)
        self.ffn_norm = nn.LayerNorm(args.dim)

    def forward(self, x: torch.Tensor):
        # Pre-LN结构：先LayerNorm，再残差连接
        h = x + self.self_attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out