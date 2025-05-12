import torch.nn as nn
from torch.nn import functional as F
from Config import Config
import torch
from Utils import _calculate_state
# 优化的残差块
class ResBlock(nn.Module):
    def __init__(self, channels, config: Config):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=config.num_groups, num_channels=channels)  # 使用分组归一化
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=config.num_groups, num_channels=channels)  # 使用分组归一化

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

# 简化的策略网络
class SimplePolicyNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # 基础卷积层
        self.conv = nn.Conv2d(2, config.num_filters, kernel_size=3, padding=1)
        self.bn = nn.GroupNorm(num_groups=config.num_groups, num_channels=config.num_filters)
        
        # 减少的残差块数量
        self.res_blocks = nn.Sequential(
            *[ResBlock(config.num_filters, config) for _ in range(config.num_res_blocks)]
        )
        
        # 策略头
        self.policy_conv = nn.Conv2d(config.num_filters, config.policy_filters, kernel_size=1)
        self.policy_bn = nn.GroupNorm(num_groups=config.policy_num_groups, num_channels=config.policy_filters)
        self.policy_fc = nn.Linear(config.policy_filters * config.board_size * config.board_size, config.num_actions) 
        #注意到这里的num_actions 修改为80了 需要将这个长度为80的tenssor对应到12*12棋盘上的位置
        
    def forward(self, x):
        # 处理输入形状
        if len(x.shape) == 3:  # (1, 3, 3)
            x = x.unsqueeze(0)  # -> (1, 1, 3, 3)
        weight_mask = _calculate_state(x) 
        # 特征提取
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        logits = self.policy_fc(policy)
        # print(f"logits: {logits}")
               

        
        if weight_mask is not None:
            # Normalize logits with softmax
            logits_norm = F.softmax(logits, dim=-1)
            
            # Normalize weight_mask to [0,1]
            weight_norm = (weight_mask - weight_mask.min()) / (weight_mask.max() - weight_mask.min() + 1e-8)
            
            # Combine in log space
            combined = logits_norm * (weight_norm + 0.1)  # Add small constant to preserve some original policy
            logits = torch.log(combined + 1e-8)  # Back to log space

        return {"logits": logits}


# 简化的价值网络
class SimpleValueNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 基础卷积层
        self.conv = nn.Conv2d(2, config.num_filters, kernel_size=3, padding=1)
        self.bn = nn.GroupNorm(num_groups=config.num_groups, num_channels=config.num_filters)
        
        # 减少的残差块数量
        self.res_blocks = nn.Sequential(
            *[ResBlock(config.num_filters, config) for _ in range(config.num_res_blocks)]
        )
        
        # 简化的价值头
        self.value_conv = nn.Conv2d(config.num_filters, config.value_filters, kernel_size=1)
        self.value_bn = nn.GroupNorm(num_groups = config.value_num_groups, num_channels=config.value_filters)
        self.value_fc = nn.Linear(config.value_filters * config.board_size * config.board_size, 1)
        
    def forward(self, x):
        # 处理输入形状
        if len(x.shape) == 3:  # (1, 12, 12)
            x = x.unsqueeze(0)  # -> (1, 1, 12, 12)
        
        # 特征提取
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = self.value_fc(value)
        value = torch.sigmoid(value)  # 使用sigmoid激活函数限制输出范围
        return {"state_value": value}