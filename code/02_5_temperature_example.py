import torch

def apply_temperature(probs, temperature):
    """Temperature 机制
    
    Args:
        probs: 概率分布
        temperature: 温度
    """
    # 对概率分布进行归一化
    probs = probs / torch.sum(probs)

    # 对概率分布进行温度调节
    probs = torch.pow(probs, 1/temperature)

    # 归一化概率
    probs = probs / torch.sum(probs)

    return probs