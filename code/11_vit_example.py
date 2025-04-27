import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from PIL import Image

def seed_everything(seed=42):
    random.seed(seed)  # Python的random模块
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python哈希种子
    np.random.seed(seed)  # NumPy随机数生成器
    torch.manual_seed(seed)  # PyTorch CPU随机数生成器
    torch.cuda.manual_seed(seed)  # 单GPU的随机数生成器
    torch.cuda.manual_seed_all(seed)  # 多GPU的随机数生成器
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 禁用cudnn自动寻找最快算法，因为这会引入随机性

# 辅助函数：将单个数值转换为元组, 例如同时支持 square=224 和 rectangle=(224,196) 这两种输入方式
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ViT 模型定义部分
# ===============================================

# 前馈网络模块，Transformer中的FFN部分
class FeedForward(nn.Module):
    # dim: 输入的维度
    # hidden_dim: 隐藏层的维度
    # dropout: 应用的dropout率
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # ViT用pre-norm结构提升稳定性，Transformer用GELU激活函数而非ReLU
        self.net = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, hidden_dim), # 扩展维度
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, dim),  # 恢复原始维度
            nn.Dropout(dropout)  # 用两次dropout增强泛化能力
        )
    # 输入/输出的 shape均为: [batch, seq_len, dim]
    def forward(self, x):
        return self.net(x)

# 多头注意力机制，Transformer的核心组件
class Attention(nn.Module):
    # dim: 输入的维度
    # heads: 多头注意力的头数
    # dim_head: 每个头的维度
    # dropout: 应用的dropout率
    # 输入 x 的 shape: [batch, seq_len, dim]
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        # 总注意力维度 = 头数 * 每个头的维度
        inner_dim = dim_head * heads
        # 只有当单头且dim_head=dim时才不需要最后的投影层
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads  # 多头注意力中的头数
        self.scale = dim_head ** -0.5  # 缩放因子，sqrt(dim_head)，防止softmax梯度消失
        
        self.norm = nn.LayerNorm(dim)  # 同样采用Pre-norm结构
        self.attend = nn.Softmax(dim=-1) 
        self.dropout = nn.Dropout(dropout)  # 应用于注意力权重
        
        # QKV投影，合并为一个线性层以提高效率
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False) # shape: [batch, seq_len, dim] -> [batch, seq_len, inner_dim * 3]
        
        # 输出投影，如果需要的话
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), # shape: [batch, seq_len, inner_dim] -> [batch, seq_len, dim]
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() # Identity() 表示恒等变换，不改变输入/输出的 shape
    
    # 输入 x 的 shape: [batch, seq_len, dim]
    # 输出 x 的 shape: [batch, seq_len, dim]
    def forward(self, x):
        x = self.norm(x)  # 先标准化，Pre-norm结构
        
        # 一次性计算QKV并分块
        qkv = self.to_qkv(x).chunk(3, dim=-1) # shape: [batch, seq_len, dim] -> [batch, seq_len, dim_head * heads * 3] -> [batch, seq_len, dim_head * heads], [batch, seq_len, dim_head * heads], [batch, seq_len, dim_head * heads]
        # 使用einops进行形状变换，将每个QKV分离为多头形式
        # 从 [batch, seq_len, heads*dim_head] 变为 [batch, heads, seq_len, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # 注意力计算：Q和K的矩阵乘法，然后应用缩放因子, QK^T/sqrt(dim_head)
        # shape: [batch, heads, seq_len, dim_head] * [batch, heads, dim_head, seq_len] -> [batch, heads, seq_len, seq_len]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 应用softmax得到注意力权重
        attn = self.attend(dots) # shape: [batch, heads, seq_len, seq_len] -> [batch, heads, seq_len, seq_len]
        attn = self.dropout(attn)  # 对注意力权重应用dropout
        
        # 注意力权重与V相乘得到输出
        # shape: [batch, heads, seq_len, seq_len] * [batch, heads, seq_len, dim_head] -> [batch, heads, seq_len, dim_head]
        out = torch.matmul(attn, v) 
        # 重新排列多头结果，恢复原始形状
        # shape: [batch, heads, seq_len, dim_head] -> [batch, seq_len, heads * dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 最后通过输出投影层
        # shape: [batch, seq_len, heads * dim_head] -> [batch, seq_len, dim]
        return self.to_out(out)

# Transformer的编码器层，由多个注意力层和前馈网络构成
class Transformer(nn.Module):
    # dim: 输入的维度
    # depth: Transformer的层数
    # heads: 多头注意力的头数
    # dim_head: 每个头的维度
    # mlp_dim: 前馈网络的维度
    # dropout: 应用的dropout率
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 最终的LayerNorm
        self.layers = nn.ModuleList([])  # 存储多个Transformer层
        
        # 创建多个Transformer层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),  # 多头注意力
                FeedForward(dim, mlp_dim, dropout=dropout)  # 前馈网络
            ]))
    
    # 输入 x 的 shape: [batch, seq_len, dim]
    # 输出 x 的 shape: [batch, seq_len, dim]
    def forward(self, x):
        # 按顺序应用每个Transformer层
        for attn, ff in self.layers:
            # 注意这里使用了残差连接，是Transformer的关键设计
            x = attn(x) + x  # 注意力层 + 残差连接
            x = ff(x) + x  # 前馈层 + 残差连接
        
        # 最后应用LayerNorm
        return self.norm(x)

# Vision Transformer主模型
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, 
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        """
        image_size: 输入图像的大小
        patch_size: 每个图像块的大小
        num_classes: 分类类别数
        dim: 模型的隐层维度
        depth: Transformer的层数
        heads: 多头注意力的头数
        mlp_dim: 前馈网络的隐层维度
        pool: 池化方式，'cls'或'mean'
        channels: 输入图像的通道数
        dim_head: 每个注意力头的维度
        dropout: 模型内部使用的dropout率
        emb_dropout: 应用于嵌入层的dropout率
        """
        super().__init__()
        
        # 处理图像和块的尺寸
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        # 确保图像尺寸能被块尺寸整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        
        # 计算块的数量和每个块的维度
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        # 确保池化方式有效
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # 图像块嵌入层
        # 这是ViT的关键部分，将2D图像转换为1D序列
        self.to_patch_embedding = nn.Sequential(
            # 将图像重新排列成块序列
            # 从 [batch, channel, height, width] 变为 [batch, patches, patch_dim]
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),  # 对每个块进行标准化
            nn.Linear(patch_dim, dim),  # 线性投影到模型维度
            nn.LayerNorm(dim),  # 再次标准化
        )
        
        # 位置编码，用于提供序列位置信息
        # +1是为了额外的分类标记
        # nn.Parameter 表示这是一个可训练的参数
        # torch.randn(1, num_patches + 1, dim) 表示生成一个形状为 (1, num_patches + 1, dim) 的张量，其中 num_patches + 1 是序列的长度，dim 是每个位置的维度
        # 之所以 num_patches + 1，是因为需要一个额外的分类标记，用于表示整个序列的特征
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # 分类标记，类似于BERT的[CLS]标记
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # 嵌入层dropout
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # 池化方式：cls或mean
        self.pool = pool
        
        # 输出前的恒等变换，可用于额外处理
        self.to_latent = nn.Identity()
        
        # 最终的分类头
        self.mlp_head = nn.Linear(dim, num_classes)
    
    def forward(self, img):
        # 将图像变为块嵌入
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape  # 批次大小，块数量，特征维度
        
        # 扩展cls标记到批次大小
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        
        # 将cls标记添加到序列开头
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 加入位置编码
        # 这里使用切片是为了处理可能的位置编码和输入序列长度不匹配的情况
        x += self.pos_embedding[:, :(n + 1)]
        
        # 应用dropout
        x = self.dropout(x)
        
        # 通过Transformer编码器
        x = self.transformer(x)
        
        # 池化：平均池化或使用cls标记
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        
        # 通过潜在空间投影
        x = self.to_latent(x) # shape: [batch, dim] -> [batch, dim]
        
        # 通过分类头得到最终输出
        return self.mlp_head(x) # shape: [batch, dim] -> [batch, num_classes]

# 训练和评估部分
# ===============================================

# 单个训练epoch
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # 设置为训练模式，启用dropout等
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm创建进度条
    pbar = tqdm(dataloader)
    for inputs, targets in pbar:
        # 将数据移动到指定设备(CPU/GPU)
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 梯度清零，防止梯度累积
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 参数更新
        optimizer.step()
        
        # 累积统计信息
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # 获取最高概率的类别
        total += targets.size(0)  # 累计样本数
        correct += predicted.eq(targets).sum().item()  # 累计正确预测数
        
        # 更新进度条显示
        pbar.set_description(f'Loss: {running_loss/(pbar.n+1):.4f} | Acc: {100.*correct/total:.2f}%')
    
    # 返回整个epoch的平均损失和准确率
    return running_loss/len(dataloader), 100.*correct/total

# 验证函数，用于评估模型在验证集上的性能
def validate(model, dataloader, criterion, device):
    model.eval()  # 设置为评估模式，禁用dropout等
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 不需要计算梯度，节省内存并加速计算
    with torch.no_grad():
        pbar = tqdm(dataloader)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 累积统计信息
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            pbar.set_description(f'Loss: {running_loss/(pbar.n+1):.4f} | Acc: {100.*correct/total:.2f}%')
    
    # 返回整个验证集的平均损失和准确率
    return running_loss/len(dataloader), 100.*correct/total

# 单张图像推理函数
def predict(model, image_path, transform, device, class_names=None):
    model.eval()  # 评估模式
    
    # 加载并转换图像
    # 转换为RGB确保与模型训练时的输入一致
    image = Image.open(image_path).convert('RGB')
    
    # 应用图像变换并添加批次维度
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 进行推理
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
    
    # 获取预测类别索引
    predicted_class = predicted.item()
    
    # 如果提供了类别名称，返回对应名称而非索引
    if class_names is not None:
        return class_names[predicted_class]
    
    return predicted_class

# 可视化训练过程，帮助分析模型性能
def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(12, 4))  # 创建大小合适的图形
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()  # 调整布局防止重叠
    plt.show()

# 主函数：完整的训练和评估流程
def main():
    # 设置随机种子确保可复现性
    seed_everything(42)
    
    # 配置训练参数
    batch_size = 64  # 批次大小，较大值可加速训练但需要更多内存
    num_epochs = 10  # 训练轮数
    learning_rate = 1e-4  # 学习率，Transformer通常使用较小的学习率
    image_size = 224  # 输入图像尺寸，标准的ImageNet预处理尺寸
    patch_size = 16  # 图像块大小，ViT标准设置
    num_classes = 10  # CIFAR-10有10个类别
    
    # 设置计算设备，优先使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据增强和标准化
    # 数据增强对防止过拟合非常重要，特别是对小数据集
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),  # 随机裁剪并缩放到指定大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转增加多样性
        transforms.ToTensor(),  # 转换为张量，同时将像素值缩放到[0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化到[-1,1]区间
    ])
    
    # 测试集不需要数据增强，但需要相同的标准化
    transform_test = transforms.Compose([
        transforms.Resize(image_size),  # 缩放到指定大小
        transforms.CenterCrop(image_size),  # 中心裁剪确保一致性
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载CIFAR-10数据集
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))  # 80%用于训练
    val_size = len(dataset) - train_size  # 20%用于验证
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 加载测试集
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # 创建数据加载器
    # num_workers=2使用多进程加速数据加载
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # CIFAR-10的类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 创建ViT模型实例
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=512,  # 模型维度
        depth=6,  # Transformer层数
        heads=8,  # 注意力头数
        mlp_dim=1024,  # 前馈网络维度，通常是dim的2-4倍
        dropout=0.1,  # 模型内部dropout
        emb_dropout=0.1  # 嵌入层dropout
    ).to(device)  # 移动到指定设备
    
    # 设置损失函数，分类任务标准选择
    criterion = nn.CrossEntropyLoss()
    
    # 设置优化器，Adam对Transformer通常效果较好
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器，使用余弦退火策略
    # 这可以使学习率从初始值逐渐下降，有助于模型收敛
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 初始化训练记录容器
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # 开始训练循环
    print('开始训练')
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # 训练阶段
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 更新学习率
        scheduler.step()
        
        # 打印当前epoch的结果
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
    
    # 绘制训练历史图表
    # plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # 在测试集上进行最终评估
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
    
    # 保存训练好的模型以便将来使用
    torch.save(model.state_dict(), 'vit_model.pth')
    print('模型已保存到 vit_model.pth')
    
    # 如果有示例图像，进行测试推理
    print('\n示例推理：')
    sample_image_path = 'sample_image.jpg'
    if os.path.exists(sample_image_path):
        predicted_class = predict(model, sample_image_path, transform_test, device, class_names)
        print(f'预测类别: {predicted_class}')

# 用于加载已保存的模型并进行推理的函数
def load_and_predict(image_path, model_path='vit_model.pth', num_classes=10, image_size=224, patch_size=16):
    # 设置计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 重新创建模型架构，必须与保存时一致
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024
    ).to(device)
    
    # 加载保存的模型权重
    # map_location参数确保模型能在当前设备上加载
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 定义与训练时相同的图像变换
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # CIFAR-10的类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 进行预测
    predicted_class = predict(model, image_path, transform, device, class_names)
    return predicted_class

# 主入口点
if __name__ == '__main__':
    main()