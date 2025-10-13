import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScale1DCNN(nn.Module):
    def __init__(self, in_channels, out_channels=128, scales=[5, 10, 25], num_class=4):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in scales:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=k//2),  # 保持时间维度不变
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1),    # 细化特征
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
            self.branches.append(branch)
        
        # 通道注意力模块（参考MixNet[3](@ref)）
        # self.SE = nn.Sequential(
        #     nn.Linear(out_channels, 1),
        #     nn.Sigmoid()
        # )
        self.clf = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, num_class),
        )

    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            branch_outputs.append(out)
        
        # 对齐特征尺度（上采样较小尺度分支）
        max_time = branch_outputs[0].shape[2]  # 最大尺度分支的时间维度
        aligned_outputs = [branch_outputs[0]]
        for i, out in enumerate(branch_outputs[1:]):
            aligned_outputs.append(F.interpolate(out, size=max_time, mode='linear'))
        aligned_outputs = torch.stack(aligned_outputs).permute(1,0,2,3)  # [batch, branches, channels, time]
        # 加权融合（归一化权重）
        # weights = F.softmax(aligned_outputs.mean(3), dim=0)
        feat  = aligned_outputs.mean((1,3))
        out = self.clf(feat)
        return feat, out
    
# 输入: [batch=32, channel=64, time=100]
model = MultiScale1DCNN(in_channels=64, out_channels=128)
input_tensor = torch.randn(32, 64, 100)
output = model(input_tensor)  # 输出: [32, 128, 100]