import torch
from torch import nn
import torch.nn.functional as F

class MultiScale1DCNN_v2(nn.Module):
    def __init__(self, in_channels, out_channels=128, scales=[5, 10, 25], num_class=4):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in scales:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=1),
                nn.BatchNorm1d(out_channels),
                # nn.Dropout1d(0.2),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=1),
                nn.BatchNorm1d(out_channels),
                nn.Dropout1d(0.2),
                nn.ReLU()
            )
            self.branches.append(branch)
        self.clf = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, num_class),
        )
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        std = x.std(dim=-1, keepdim=True) * std_scaling
        noise = torch.randn_like(mean) * std + mean
        x = x + noise
        return x

    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            branch_outputs.append(out.mean(-1))  # 对每个分支的输出在时间维度上取平均
    
        outs = [self.clf(i)for i in branch_outputs]
        features = torch.stack(branch_outputs).mean(0)
        out = self.clf(features)
        outs.append(out)
        return outs, F.normalize(features, p=2, dim=-1)
