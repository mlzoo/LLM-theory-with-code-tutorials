def train_step(model, optimizer, input_ids, target_ids, criterion):
    """训练一步
    
    Args:
        model: Transformer模型
        optimizer: 优化器
        input_ids: 输入序列的token ids, shape=[batch_size, seq_len]
        target_ids: 目标序列的token ids, shape=[batch_size, seq_len] 
        criterion: 损失函数(通常是CrossEntropyLoss)
    
    Returns:
        loss: 当前步的损失值
    """
    # 将模型设为训练模式
    model.train()
    
    # 清空梯度
    optimizer.zero_grad()
    
    # 前向传播,得到模型输出
    # outputs shape: [batch_size, seq_len, vocab_size]
    outputs = model(input_ids)
    
    # 计算损失
    # 将outputs重塑为[batch_size * seq_len, vocab_size]
    # target_ids重塑为[batch_size * seq_len]
    loss = criterion(
        outputs.view(-1, outputs.size(-1)),
        target_ids.view(-1)
    )
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    optimizer.step()
    
    return loss.item()

def train_epoch(model, train_dataloader, optimizer, criterion, device):
    """训练一个epoch
    
    Args:
        model: Transformer模型
        train_dataloader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 训练设备(CPU/GPU)
    
    Returns:
        epoch_loss: 当前epoch的平均损失
    """
    model.train()
    total_loss = 0
    
    # 遍历训练数据
    for batch in train_dataloader:
        # 将数据移到指定设备
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        # 训练一步
        loss = train_step(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            target_ids=target_ids,
            criterion=criterion
        )
        
        total_loss += loss
        
    # 计算平均损失
    avg_loss = total_loss / len(train_dataloader)
    
    return avg_loss