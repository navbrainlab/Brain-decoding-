import numpy as np
import torch
from torch import nn, Tensor
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from typing import Any, Optional, Union
from timm.layers import LayerType, Mlp
# from model.wavelets import Morlet
from torchaudio.transforms import Resample
from scipy import signal
from mne.filter import create_filter
# from model.SincNet import SincConv_fast, SincConv
from einops import rearrange, repeat    

class ArcFace(nn.Module):
    def __init__(self, num_classes, embedding_size, s=30.0, m=0.5, reduction='mean'):
        super(ArcFace, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_normal_(self.W)  # 权重初始化

        self.reduction = reduction

    def forward(self, embeddings, labels):
        # 1. 特征和权重归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)  # (batch_size, embedding_size)
        W_norm = F.normalize(self.W, p=2, dim=1)  # (num_classes, embedding_size)

        # 2. 计算余弦相似度矩阵
        logits = torch.matmul(embeddings, W_norm.T)  # (batch_size, num_classes)

        # 3. 提取对应类别的余弦值，并添加角度间隔
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))  # 避免数值不稳定
        one_hot = F.one_hot(labels, self.num_classes)
        logits_with_margin = torch.cos(theta + self.m * one_hot)  # 仅对目标类添加m

        # 4. 缩放logits并计算交叉熵损失
        logits_scaled = logits_with_margin * self.s
        loss = F.cross_entropy(logits_scaled, labels, reduction=self.reduction)
        return loss


class SqueezeExcitation1D(nn.Module):  # 通道注意力机制
    def __init__(self, in_channels: int, reduction: int = 4):
        super(SqueezeExcitation1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: Any = 'same',
                 bias: bool = True,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d):
        super(ConvBlock1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, ),
            norm_layer(out_channels) if norm_layer == nn.BatchNorm1d else norm_layer(out_channels // 4, out_channels),
            act_layer(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x) + x  # 512 256 -》 512 256
        return x


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: Any = 'same',
                 bias: bool = True,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 n_layers: int = 2):
        super(ResidualBlock1D, self).__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        layers = []
        for i in range(n_layers):
            layers.append(
                ConvBlock1D(out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            act_layer=act_layer))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1), bias: bool = True, act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d):
        super(ConvBlock2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride,
                      bias=bias),
            norm_layer(out_channels) if norm_layer == nn.BatchNorm2d else norm_layer(out_channels // 4, out_channels),
            act_layer()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x) + x
        return x


class ConvUpSample2D(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 padding: Any = 1,
                 bias: bool = True,
                 act_layer: Optional[LayerType] = nn.GELU):
        super(ConvUpSample2D, self).__init__()
        self.layer = nn.ModuleList(
            [nn.ConvTranspose2d(in_channels, out_channels,
                                kernel_size=(1, kernel_size),
                                padding=(0, padding),
                                stride=(1, stride),
                                bias=bias),
             nn.GroupNorm(out_channels // 4, out_channels),
             act_layer()]
        )
        self.output_size = None

    def forward(self, x: Tensor, output_size=None) -> Tensor:
        B, C, H, W = x.shape
        if output_size is None and self.output_size is None:
            output_size = [B, C // 2, H, W * 4]
            self.output_size = output_size
        elif self.output_size[0] != B:
            self.output_size[0] = B

        for layer in self.layer:
            x = layer(x, output_size=self.output_size) if isinstance(layer, nn.ConvTranspose2d) else layer(x)
        return x


class ConvDownSample2D(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 padding: Any = 1,
                 bias: bool = True,
                 act_layer: Optional[LayerType] = nn.GELU):
        super(ConvDownSample2D, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=(5, kernel_size),
                      padding=(2, padding),
                      stride=(1, stride),
                      bias=bias),
            nn.GroupNorm(out_channels // 4, out_channels),
            act_layer()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class Decoder(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 upsample: int = 4,
                 padding: Any = 'same',
                 bias: bool = True,
                 act_layer: Optional[LayerType] = nn.GELU,
                 n_layers: int = 4,
                 is_residual: bool = False,
                 n_residual: int = 2):
        super(Decoder, self).__init__()

        layers = []
        dk = kernel_size + 2
        for i in range(n_layers):
            if is_residual:
                for _ in range(n_residual):
                    layers.append(
                        ConvBlock2D(in_channels,
                                    in_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    bias=bias,
                                    act_layer=act_layer))
            layers.append(
                ConvUpSample2D(in_channels,
                               in_channels // 2,
                               kernel_size=dk,
                               stride=upsample,
                               padding=dk // 2,
                               bias=bias,
                               act_layer=act_layer))
            in_channels = in_channels // 2

        self.post_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv = nn.ModuleList(layers)
        self.in_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.conv:
            x = layer(x)
        x = self.post_conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 downsample: int = 2,
                 padding: Any = 'same',
                 bias: bool = True,
                 act_layer: Optional[LayerType] = nn.GELU,
                 n_layers: int = 4,
                 is_residual: bool = False,
                 n_residual: int = 2):
        super(Encoder, self).__init__()

        dk = kernel_size + 2
        layers = []

        for i in range(n_layers):
            if is_residual:
                for _ in range(n_residual):
                    layers.append(
                        ConvBlock2D(in_channels if len(layers) == 0 else out_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias,
                                    act_layer=act_layer
                                    ))
            layers.append(
                ConvDownSample2D(out_channels,
                                 out_channels * 2,
                                 kernel_size=dk,
                                 stride=downsample,
                                 padding=dk // 2,
                                 bias=bias,
                                 act_layer=act_layer
                                 ))
            out_channels = out_channels * 2

        self.conv = nn.ModuleList(layers)
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.conv:
            x = layer(x)
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 downsample: int = 4,
                 padding: Any = 'same',
                 bias: bool = True,
                 act_layer: Optional[LayerType] = nn.GELU,
                 n_layers: int = 4,
                 is_residual: bool = False,
                 n_residual: int = 2):
        super(DownConv, self).__init__()

        dk = kernel_size + 2
        layers = []

        for i in range(n_layers):
            if is_residual:
                for _ in range(n_residual):
                    layers.append(
                        ConvBlock2D(in_channels if len(layers) == 0 else out_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    bias=bias,
                                    act_layer=act_layer
                                    ))
            layers.append(
                ConvDownSample2D(out_channels,
                                 out_channels,
                                 kernel_size=dk,
                                 stride=downsample,
                                 padding=dk // 2,
                                 bias=bias,
                                 act_layer=act_layer
                                 ))
            # out_channels = out_channels * 2

        self.conv = nn.ModuleList(layers)
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.conv:
            x = layer(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 'same',
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 ):
        super(ResNet, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=first_kernel_size, padding=padding),
            norm_layer(out_channels) if norm_layer == nn.BatchNorm1d else norm_layer(out_channels // 4, out_channels),
            act_layer()
        )
        layers = []
        for i in range(n_layers):
            layers.append(
                ConvBlock1D(out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            act_layer=act_layer,
                            norm_layer=norm_layer))

        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = Mlp(out_channels,
                       hidden_features=out_channels,
                       out_features=n_classes,
                       act_layer=act_layer,
                       drop=drop_out,
                       norm_layer=nn.LayerNorm)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # if self.training:
        #     x = self.add_noise(x)
        x = self.first_layer(x)  # 128 256 -》 512 256
        x = self.layers(x)# 512 256 -》 512 256
        x = self.avg_pool(x)# 512 256 -》 512 1
        features = x.view(x.size(0), -1) # 512 1 -》 512
        x = self.mlp(features)
        if not self.training:
            # x = F.softmax(x, dim=-1)
            x = F.sigmoid(x)
        return x, F.normalize(features, p=2, dim=-1)  # 特征归一化

    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        std = x.std(dim=-1, keepdim=True) * std_scaling
        noise = torch.randn_like(mean) * std + mean
        x = x + noise
        return x

    def update_class_weights(self, y=None, weights=None):
        """更新类别权重"""
        if weights is not None:
            self._class_weights = weights.to(self.classifier.fc1.weight.device)
        elif y is not None:
            self.configure_loss(y)

    def configure_loss(self, y: torch.Tensor):
        """动态计算类别权重"""
        counts = torch.bincount(y)
        weights = 1.0 / (counts.float() + 1e-8)
        weights /= weights.sum()
        # self._class_weights = weights.to(self.classifier.fc1.weight.device)

    def compute_loss(self, outputs, targets):
        # 加权损失计算
        loss_func = nn.CrossEntropyLoss()
        weighted_loss = loss_func(outputs, targets)

        return weighted_loss






#-------------------new-------------------
electrod_index = torch.tensor([
    [torch.inf, 102, 120, 80, 97, 73, 56, 33, 48, 9, 23, torch.inf],
    [65, 104, 118, 67, 95, 72, 57, 36, 63, 11, 24, 62],
    [96, 92, 114, 127, 91, 75, 54, 40, 2, 15, 35, 32],
    [99, 86, 111, 125, 89, 74, 55, 44, 4, 18, 39, 30],
    [101, 84, 110, 123, 79, 107, 27, 51, 6, 19, 43, 28],
    [103, 82, 100, 121, 76, 98, 31, 49, 8, 25, 52, 26],
    [116, 81, 105, 119, 66, 94, 37, 64, 10, 22, 47, 13],
    [117, 128, 93, 115, 69, 90, 41, 61, 14, 34, 1, 12],
    [109, 126, 87, 113, 71, 88, 45, 60, 16, 38, 3, 20],
    [108, 124, 85, 112, 68, 77, 53, 59, 17, 42, 5, 21],
    [torch.inf, 122, 83, 106, 70, 78, 46, 58, 29, 50, 7, torch.inf],
])

class Spatial2dconv(nn.Module):
    # Spatial2dconv
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 pool_type='avg',
                 ):
        super(Spatial2dconv, self).__init__()

        # spatial 2D convolution part
        self.electrod_index = electrod_index
        self.valid_pos = (self.electrod_index!=torch.inf)
        self.first_conv = nn.Conv2d(1, 64, kernel_size, padding=kernel_size//2, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(
            *[Residual2DBlock(64, 64, kernel_size) for _ in range(1)]
        )
        self.pool_type = pool_type

        # temporal convlution part
        out_channels = 64
        self.first_layer = nn.Sequential(
            nn.Conv1d(64, out_channels, kernel_size=first_kernel_size, padding=padding),
            norm_layer(out_channels) if norm_layer == nn.BatchNorm1d else norm_layer(out_channels // 4, out_channels),
            act_layer()
        )
        layers = []
        for i in range(1):
            layers.append(
                ConvBlock1D(out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            act_layer=act_layer,
                            norm_layer=norm_layer))

        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = Mlp(out_channels,
                       hidden_features=out_channels,
                       out_features=n_classes,
                       act_layer=act_layer,
                       drop=drop_out,
                       norm_layer=nn.LayerNorm)
        self.initialize_weights()


    def forward(self, x):
        x = self.restore2d(x)  # 恢复为2D形状 [B, 1, 11, 12, 256]
        # x: [B, 1, 11, 12, 256]
        B, C, H, W, T = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, T, C, H, W]
        x = x.view(B * T, C, H, W)  # [B*T, C, H, W]
        x = self.first_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.res_blocks(x)  # 残差结构
        if self.pool_type == 'avg':
            x = F.adaptive_avg_pool2d(x, (1, 1))  # [B*T, out_channels, 1, 1]
        else:
            x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.view(B, T, -1)  # [B, T, out_channels]
        x = x.permute(0, 2, 1).contiguous()  # [B, out_channels, T]

        #-----temporal conv
        x = self.first_layer(x)  # 128 256 -》 512 256
        x = self.layers(x)# 512 256 -》 512 256
        x = self.avg_pool(x)# 512 256 -》 512 1
        features = x.view(x.size(0), -1) # 512 1 -》 512
        x = self.mlp(features)
        if not self.training:
            # x = F.softmax(x, dim=-1)
            x = F.sigmoid(x)
        return x, F.normalize(features, p=2, dim=-1)  # 特征归一化      


    def restore2d(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        x_2d = torch.zeros((B, 132, T), device=x.device, dtype=x.dtype)
        x_2d[:, self.valid_pos.flatten()] = x[:,(self.electrod_index[self.valid_pos]-1).long()]
        return x_2d.view(B,1,11,12,T)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
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

    def update_class_weights(self, y=None, weights=None):
        """更新类别权重"""
        if weights is not None:
            self._class_weights = weights.to(self.classifier.fc1.weight.device)
        elif y is not None:
            self.configure_loss(y)

    def configure_loss(self, y: torch.Tensor):
        """动态计算类别权重"""
        counts = torch.bincount(y)
        weights = 1.0 / (counts.float() + 1e-8)
        weights /= weights.sum()
        # self._class_weights = weights.to(self.classifier.fc1.weight.device)

    def compute_loss(self, outputs, targets):
        # 加权损失计算
        loss_func = nn.CrossEntropyLoss()
        weighted_loss = loss_func(outputs, targets)

        return weighted_loss

class Residual2DBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=128, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        identity = self.downsample(identity)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class SpatialResNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, n_blocks=2, pool_type='avg'):
        super().__init__()
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(
            *[Residual2DBlock(out_channels, kernel_size) for _ in range(n_blocks)]
        )
        self.pool_type = pool_type

    def forward(self, x):
        # x: [B, 1, 11, 12, 256]
        B, C, H, W, T = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, T, C, H, W]
        x = x.view(B * T, C, H, W)  # [B*T, C, H, W]
        x = self.first_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.res_blocks(x)  # 残差结构
        if self.pool_type == 'avg':
            x = F.adaptive_avg_pool2d(x, (1, 1))  # [B*T, out_channels, 1, 1]
        else:
            x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.view(B, T, -1)  # [B, T, out_channels]
        x = x.permute(0, 2, 1).contiguous()  # [B, out_channels, T]
        return x  # [B, c, 256]

# a = torch.randn([1,132,256])

# b =  rearrange(a, 'b (h w) c -> b h w c',h=11, w=12, c=256,b=1)
# c = a.view(1,11,12,256)
# d= c==b

# -----treat t as channel
class Spatial2dconv_v2(Spatial2dconv):
    # Spatial2dconv
    def __init__(self, 
                 in_channels: 3,
                 out_channels: 3,
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 pool_type='avg',
                 ):
        super(Spatial2dconv, self).__init__()

        # spatial 2D convolution part
        self.electrod_index = electrod_index
        self.valid_pos = (self.electrod_index!=torch.inf)
        self.conv_block = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size, padding=0, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(128, 128, 5, padding=0, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.clf = nn.Linear(128, n_classes)
        self.initialize_weights()


    def forward(self, x):
        x = self.restore2d(x)  # 恢复为2D形状 [B, 1, 11, 12, 256]
        # x: [B, 1, 11, 12, 256]
        B, H, W, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T, H, W]
        x = self.conv_block(x)
        features = x.view(x.size(0), -1) # 512 1 -》 512
        x = self.clf(features)
        if not self.training:
            # x = F.softmax(x, dim=-1)
            x = F.sigmoid(x)
        return x, F.normalize(features, p=2, dim=-1)  # 特征归一化      
    
    def restore2d(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        x_2d = torch.zeros((B, 132, T), device=x.device, dtype=x.dtype)
        x_2d[:, self.valid_pos.flatten()] = x[:,(self.electrod_index[self.valid_pos]-1).long()]
        return x_2d.view(B,1,11,12,T)


# a = Spatial2dconv_v2(3,3)
# input = torch.randn(64, 128, 256)
# a(input)



class Spatial2dconv_v3(Spatial2dconv):
    # Spatial2dconv
    def __init__(self, 
                 in_channels: 3,
                 out_channels: 3,
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 pool_type='avg',
                 ):
        super(Spatial2dconv, self).__init__()

        # spatial 2D convolution part
        self.electrod_index = electrod_index
        self.valid_pos = (self.electrod_index!=torch.inf)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 512, (3,4), padding=0, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, (3,3), padding=0, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 512, (4,4), padding=0, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )

        self.clf = nn.Linear(512, n_classes)
        self.initialize_weights()


    def forward(self, x):
        x = self.restore2d(x)  # 恢复为2D形状 [B, 1, 11, 12, 256]
        # x: [B, 1, 11, 12, 256]
        B, H, W, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T, H, W]
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        features = torch.mean(x3, dim=(2, 3))  # [B, C, T] -> [B, C]
        x = self.clf(features)
        if not self.training:
            # x = F.softmax(x, dim=-1)
            x = F.sigmoid(x)
        return x, F.normalize(features, p=2, dim=-1)  # 特征归一化      
    
    def restore2d(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        x_2d = torch.zeros((B, 132, T), device=x.device, dtype=x.dtype)
        x_2d[:, self.valid_pos.flatten()] = x[:,(self.electrod_index[self.valid_pos]-1).long()]
        return x_2d.view(B,1,11,12,T)


# a = Spatial2dconv_v3(3,3)
# input = torch.randn(64, 128, 256)
# a(input)


# --------3d conv
class Spatial3dconv(Spatial2dconv):
    # Spatial2dconv
    def __init__(self, 
                 in_channels: 3,
                 out_channels: 3,
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 pool_type='avg',
                 ):
        super(Spatial2dconv, self).__init__()
        self.electrod_index = electrod_index
        self.valid_pos = (self.electrod_index!=torch.inf)
        self.conv3d = nn.Sequential(
                    nn.Conv3d(
                        in_channels=1, 
                        out_channels=32,  # 扩展特征通道
                        kernel_size=(3, 4, 25),  # H×W×T卷积核
                        stride=(2, 2, 3),  # 保持空间尺寸不变
                        
                    ),
                    nn.BatchNorm3d(32),
                    nn.ELU(inplace=True),
                    nn.Dropout3d(0.3),
                    nn.Conv3d(
                        in_channels=32, 
                        out_channels=64,  # 扩展特征通道
                        kernel_size=(3, 3, 4),  # H×W×T卷积核
                        stride=(2, 2, 2),  # 保持空间尺寸不变
                        
                    ),
                    nn.BatchNorm3d(64),
                    nn.Dropout3d(0.2),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(
                        in_channels=64, 
                        out_channels=128,  # 扩展特征通道
                        kernel_size=(2, 2, 3),  # H×W×T卷积核
                        stride=(1, 1, 3),  # 保持空间尺寸不变
                    
                    ),
                    nn.BatchNorm3d(128),
                    nn.ReLU(inplace=True)
                )
            
        self.clf = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
            
        self.initialize_weights()
        
    def forward(self, x):
        x = self.restore2d(x)  # 恢复为2D形状 [B, 1, 11, 12, 256]
        # x: [B, 1, 11, 12, 256]
        B, C, H, W, T = x.shape
        x = x.contiguous()  # [B, T, H, W]
        x = self.conv3d(x)
        features = torch.mean(x, dim=(2, 3, 4))
        x = self.clf(features)
        if not self.training:
            x = F.softmax(x, dim=-1)
            # x = F.sigmoid(x)
        return x, F.normalize(features, p=2, dim=-1)  # 特征归一化   
      
    
    def restore2d(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        x_2d = torch.zeros((B, 132, T), device=x.device, dtype=x.dtype)
        x_2d[:, self.valid_pos.flatten()] = x[:,(self.electrod_index[self.valid_pos]-1).long()]
        return x_2d.view(B,1,11,12,T)


# a = Spatial3dconv(3,3)
# input = torch.randn(64, 128, 256)
# a(input)


class Spatial3dconvMultiscale(nn.Module):
    # Spatial2dconv
    def __init__(self, 
                 in_channels: 3,
                 out_channels: 3,
                 scales=[5,15,25],
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 pool_type='avg',
                 ):
        super().__init__()
        self.electrod_index = electrod_index
        self.valid_pos = (self.electrod_index!=torch.inf)
        self.branches = nn.ModuleList()
        for k in scales:
            branch = nn.Sequential(
                    nn.Conv3d(
                        in_channels=1, 
                        out_channels=64,  # 扩展特征通道
                        kernel_size=(3, 4, k),  # H×W×T卷积核
                        stride=(2, 2, 3),  # 保持空间尺寸不变
                        
                    ),
                    nn.BatchNorm3d(64),
                    nn.ELU(inplace=True),
                    nn.Dropout3d(0.3),
                    nn.Conv3d(
                        in_channels=64, 
                        out_channels=128,  # 扩展特征通道
                        kernel_size=(3, 3, 3),  # H×W×T卷积核
                        stride=(2, 2, 2),  # 保持空间尺寸不变
                        
                    ),
                    nn.BatchNorm3d(128),
                    nn.Dropout3d(0.1),
                    nn.ReLU(inplace=True),
                )
            self.branches.append(branch)
                 
        self.clf = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
            
        self.initialize_weights()
    def initialize_weights(self):
        self.apply(self._init_weights)
    def forward(self, x):
        x = self.restore2d(x)  # 恢复为2D形状 [B, 1, 11, 12, 256]
        # x: [B, 1, 11, 12, 256]
        B, C, H, W, T = x.shape
        x = x.contiguous()  # [B, T, H, W]
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            branch_outputs.append(out.mean((2, 3, 4)))
        
        outs = [self.clf(i)for i in branch_outputs]
        features = torch.stack(branch_outputs).mean(0)
        out = self.clf(features)
        outs.append(out)
        return outs, F.normalize(features, p=2, dim=-1)  # 特征归一化   
      
    
    def restore2d(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        x_2d = torch.zeros((B, 132, T), device=x.device, dtype=x.dtype)
        x_2d[:, self.valid_pos.flatten()] = x[:,(self.electrod_index[self.valid_pos]-1).long()]
        return x_2d.view(B,1,11,12,T)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
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
# a = Spatial3dconvMultiscale(3,3)
# input = torch.randn(64, 128, 256)
# a(input)
class Spatial3dconvMultiscale_v2(nn.Module):
    def __init__(self, 
                 in_channels: 3,
                 out_channels: 3,
                 scales=[5,15,25],
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 pool_type='avg',
                 ):
        super().__init__()
        self.electrod_index = electrod_index
        self.valid_pos = (self.electrod_index!=torch.inf)
        conv1 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=64,  # 扩展特征通道
                    kernel_size=(11, 12, 25),  # H×W×T卷积核
                    stride=(1, 1, 3),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(64),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=64, 
                    out_channels=128,  # 扩展特征通道
                    kernel_size=(1, 1, 3),  # H×W×T卷积核
                    stride=(1, 1, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(128),
                nn.Dropout3d(0.1),
                nn.ReLU(inplace=True),
            )
        conv2 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=64,  # 扩展特征通道
                    kernel_size=(6, 6, 15),  # H×W×T卷积核
                    stride=(2, 2, 3),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(64),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=64, 
                    out_channels=128,  # 扩展特征通道
                    kernel_size=(2, 2, 2),  # H×W×T卷积核
                    stride=(1, 1, 1),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(128),
                nn.Dropout3d(0.1),
                nn.ReLU(inplace=True),
            )
        conv3 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=64,  # 扩展特征通道
                    kernel_size=(4, 4, 5),  # H×W×T卷积核
                    stride=(2, 2, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(64),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=64, 
                    out_channels=128,  # 扩展特征通道
                    kernel_size=(3, 3, 3),  # H×W×T卷积核
                    stride=(2, 2, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(128),
                nn.Dropout3d(0.2),
                nn.ReLU(inplace=True),
            )
        self.branches = nn.ModuleList([conv1, conv2, conv3])
                 
        self.clf = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
            
        self.initialize_weights()
    def initialize_weights(self):
        self.apply(self._init_weights)
    def forward(self, x):
        x = self.restore2d(x)  # 恢复为2D形状 [B, 1, 11, 12, 256]
        # x: [B, 1, 11, 12, 256]
        B, C, H, W, T = x.shape
        x = x.contiguous()  # [B, T, H, W]
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            branch_outputs.append(out.mean((2, 3, 4)))
        
        outs = [self.clf(i)for i in branch_outputs]
        features = torch.stack(branch_outputs).mean(0)
        out = self.clf(features)
        outs.append(out)
        return outs, F.normalize(features, p=2, dim=-1)  # 特征归一化   
      
    
    def restore2d(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        x_2d = torch.zeros((B, 132, T), device=x.device, dtype=x.dtype)
        x_2d[:, self.valid_pos.flatten()] = x[:,(self.electrod_index[self.valid_pos]-1).long()]
        return x_2d.view(B,1,11,12,T)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
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
    

class Spatial3dconvMultiscale_v2_sep(nn.Module):
    def __init__(self, 
                 in_channels: 3,
                 out_channels: 3,
                 scales=[5,15,25],
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 pool_type='avg',
                 ):
        super().__init__()
        self.electrod_index = electrod_index
        self.valid_pos = (self.electrod_index!=torch.inf)
        conv_t1 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=16,  # 扩展特征通道
                    kernel_size=(1, 1, 25),  # H×W×T卷积核
                    stride=(1, 1, 3),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(16),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=16, 
                    out_channels=32,  # 扩展特征通道
                    kernel_size=(1, 1, 3),  # H×W×T卷积核
                    stride=(1, 1, 2),  # 保持空间尺寸不变
                    
                ),
            )
        conv_t2 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=16,  # 扩展特征通道
                    kernel_size=(1, 1, 15),  # H×W×T卷积核
                    stride=(1, 1, 3),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(16),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=16, 
                    out_channels=32,  # 扩展特征通道
                    kernel_size=(1, 1, 3),  # H×W×T卷积核
                    stride=(1, 1, 2),  # 保持空间尺寸不变
                    
                ),
            )
        conv_t3 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=16,  # 扩展特征通道
                    kernel_size=(1, 1, 5),  # H×W×T卷积核
                    stride=(1, 1, 3),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(16),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=16, 
                    out_channels=32,  # 扩展特征通道
                    kernel_size=(1, 1, 3),  # H×W×T卷积核
                    stride=(1, 1, 2),  # 保持空间尺寸不变
                    
                ),
            )
        self.conv_spatial = nn.Sequential(
                nn.Conv3d(
                    in_channels=32, 
                    out_channels=64,  # 扩展特征通道
                    kernel_size=(11, 12, 1),  # H×W×T卷积核
                    stride=(1, 1, 1),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(64),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.1),
            )
        self.branches = nn.ModuleList([conv_t1, conv_t2, conv_t3])
                 
        self.clf = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32,64),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
            
        self.initialize_weights()
    def initialize_weights(self):
        self.apply(self._init_weights)
    def forward(self, x):
        x = self.restore2d(x)  # 恢复为2D形状 [B, 1, 11, 12, 256]
        # x: [B, 1, 11, 12, 256]
        B, C, H, W, T = x.shape
        x = x.contiguous()  # [B, T, H, W]
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            out = self.conv_spatial(out)
            branch_outputs.append(out.mean((2, 3, 4)))
        
        outs = [self.clf(i)for i in branch_outputs]
        features = torch.stack(branch_outputs).mean(0)
        out = self.clf(features)
        outs.append(out)
        return outs, F.normalize(features, p=2, dim=-1)  # 特征归一化   
      
    
    def restore2d(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        x_2d = torch.zeros((B, 132, T), device=x.device, dtype=x.dtype)
        x_2d[:, self.valid_pos.flatten()] = x[:,(self.electrod_index[self.valid_pos]-1).long()]
        return x_2d.view(B,1,11,12,T)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
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
    
a = Spatial3dconvMultiscale_v2_sep(3,3)
input = torch.randn(64, 128, 256)
a(input)


class Spatial3dconvMultiscale_v2_sweep(nn.Module):
    def __init__(self, 
                 scale1_small,
                 scale1_mid,
                 scale1_large,
                 out_channels: int = 64,
                 n_classes: int = 4,
                 ):
        super().__init__()
        self.electrod_index = electrod_index
        self.valid_pos = (self.electrod_index!=torch.inf)
        conv1 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=out_channels,  # 扩展特征通道
                    kernel_size=(scale1_small[0], scale1_small[1],scale1_small[2]),  # H×W×T卷积核
                    stride=(1, 1, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(out_channels),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=out_channels, 
                    out_channels=out_channels*2,  # 扩展特征通道
                    kernel_size=(1, 1, 3),  # H×W×T卷积核
                    stride=(1, 1, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(out_channels*2),
                nn.Dropout3d(0.1),
                nn.ReLU(inplace=True),
            )
        conv2 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=out_channels,  # 扩展特征通道
                    kernel_size=(scale1_mid[0], scale1_mid[1],scale1_mid[2]),  # H×W×T卷积核
                    stride=(2, 2, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(out_channels),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=out_channels, 
                    out_channels=out_channels*2,  # 扩展特征通道
                    kernel_size=(1, 1, 3),  # H×W×T卷积核
                    stride=(1, 1, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(out_channels*2),
                nn.Dropout3d(0.1),
                nn.ReLU(inplace=True),
            )
        conv3 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=out_channels,  # 扩展特征通道
                    kernel_size=(scale1_large[0], scale1_large[1],scale1_large[2]),  # H×W×T卷积核
                    stride=(1, 1, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(out_channels),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=out_channels, 
                    out_channels=out_channels*2,  # 扩展特征通道
                    kernel_size=(2, 2, 2),  # H×W×T卷积核
                    stride=(1, 1, 1),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(out_channels*2),
                nn.Dropout3d(0.2),
                nn.ReLU(inplace=True),
            )
        self.branches = nn.ModuleList([conv1, conv2, conv3])
                 
        self.clf = nn.Sequential(
            nn.Linear(out_channels*2, out_channels*4),
            nn.LeakyReLU(),
            nn.Linear(out_channels*4,out_channels*2),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(out_channels*2, n_classes)
        )
            
        self.initialize_weights()
    def initialize_weights(self):
        self.apply(self._init_weights)
    def forward(self, x):
        x = self.restore2d(x)  # 恢复为2D形状 [B, 1, 11, 12, 256]
        # x: [B, 1, 11, 12, 256]
        B, C, H, W, T = x.shape
        x = x.contiguous()  # [B, T, H, W]
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            branch_outputs.append(out.mean((2, 3, 4)))
        
        outs = [self.clf(i)for i in branch_outputs]
        features = torch.stack(branch_outputs).mean(0)
        out = self.clf(features)
        outs.append(out)
        return outs, F.normalize(features, p=2, dim=-1)  # 特征归一化   
      
    
    def restore2d(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        x_2d = torch.zeros((B, 132, T), device=x.device, dtype=x.dtype)
        x_2d[:, self.valid_pos.flatten()] = x[:,(self.electrod_index[self.valid_pos]-1).long()]
        return x_2d.view(B,1,11,12,T)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
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
    

class Spatial3dconvMultiscale_v2_Att(nn.Module):
    def __init__(self, 
                 scale1_small,
                 scale1_mid,
                 scale1_large,
                 n_classes: int = 4,
                 ):
        super().__init__()
        self.electrod_index = electrod_index
        self.valid_pos = (self.electrod_index!=torch.inf)
        conv1 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=64,  # 扩展特征通道
                    kernel_size=(scale1_small[0], scale1_small[1],scale1_small[2]),  # H×W×T卷积核
                    stride=(1, 1, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(64),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=64, 
                    out_channels=128,  # 扩展特征通道
                    kernel_size=(1, 1, 3),  # H×W×T卷积核
                    stride=(1, 1, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(128),
                nn.Dropout3d(0.1),
                nn.ReLU(inplace=True),
            )
        conv2 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=64,  # 扩展特征通道
                    kernel_size=(scale1_mid[0], scale1_mid[1],scale1_mid[2]),  # H×W×T卷积核
                    stride=(2, 2, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(64),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=64, 
                    out_channels=128,  # 扩展特征通道
                    kernel_size=(1, 1, 3),  # H×W×T卷积核
                    stride=(1, 1, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(128),
                nn.Dropout3d(0.1),
                nn.ReLU(inplace=True),
            )
        conv3 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=64,  # 扩展特征通道
                    kernel_size=(scale1_large[0], scale1_large[1],scale1_large[2]),  # H×W×T卷积核
                    stride=(1, 1, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(64),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=64, 
                    out_channels=128,  # 扩展特征通道
                    kernel_size=(2, 2, 2),  # H×W×T卷积核
                    stride=(1, 1, 1),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(128),
                nn.Dropout3d(0.2),
                nn.ReLU(inplace=True),
            )
        self.branches = nn.ModuleList([conv1, conv2, conv3])
                 
        self.clf = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        self.scale_pos = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, 128)) for i in range(3)])
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            batch_first=True,  # 使用(batch, seq, feature)格式
            dropout = 0.1
        )      
        self.initialize_weights()


    def initialize_weights(self):
        self.apply(self._init_weights)


    def forward(self, x):
        x = self.restore2d(x)  # 恢复为2D形状 [B, 1, 11, 12, 256]
        # x: [B, 1, 11, 12, 256]
        B, C, H, W, T = x.shape
        x = x.contiguous()  # [B, T, H, W]
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            branch_outputs.append(out)
        pooled_branch_outputs = [i.mean((2,3,4)) for i in branch_outputs]
        
        mha_input = torch.concat([i.mean(-1).flatten(2).permute(0,2,1)+j for i,j in zip(branch_outputs,self.scale_pos)],1)
        attn_output, attn_weights = self.multihead_attn(
            query=mha_input,    # (B, num_query, time_len)
            key=mha_input,      # (B, channel_num, time_len)  
            value=mha_input,    # (B, channel_num, time_len)
        )# 输出形状: (B, num_query, time_len)
        outs = [self.clf(i)for i in pooled_branch_outputs]

        features = attn_output.mean(1)
        out = self.clf(features)
        outs.append(out)
        return outs, F.normalize(features, p=2, dim=-1)  # 特征归一化   
      
    
    def restore2d(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        x_2d = torch.zeros((B, 132, T), device=x.device, dtype=x.dtype)
        x_2d[:, self.valid_pos.flatten()] = x[:,(self.electrod_index[self.valid_pos]-1).long()]
        return x_2d.view(B,1,11,12,T)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
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
# a = Spatial3dconvMultiscale_v2_Att(scale1_small=[5,5,15], scale1_mid=[8,8,15], scale1_large=[10,10,25])
# input = torch.randn(64, 128, 256)
# a(input)
    

class Spatial3dconvMultiscale_v3(Spatial3dconv, nn.Module):
    # Spatial2dconv
    def __init__(self, 
                 in_channels: 3,
                 out_channels: 3,
                 scales=[5,15,25],
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 pool_type='avg',
                 ):
        nn.Module.__init__(self)
        self.electrod_index = electrod_index
        self.valid_pos = (self.electrod_index!=torch.inf)
        conv1 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=64,  # 扩展特征通道
                    kernel_size=(11, 12, 25),  # H×W×T卷积核
                    stride=(1, 1, 3),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(64),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=64, 
                    out_channels=128,  # 扩展特征通道
                    kernel_size=(1, 1, 3),  # H×W×T卷积核
                    stride=(1, 1, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(128),
                nn.Dropout3d(0.1),
                nn.ReLU(inplace=True),
            )
        conv2 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=64,  # 扩展特征通道
                    kernel_size=(6, 6, 15),  # H×W×T卷积核
                    stride=(2, 2, 3),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(64),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=64, 
                    out_channels=128,  # 扩展特征通道
                    kernel_size=(2, 2, 2),  # H×W×T卷积核
                    stride=(1, 1, 1),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(128),
                nn.Dropout3d(0.1),
                nn.ReLU(inplace=True),
            )
        conv3 = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, 
                    out_channels=64,  # 扩展特征通道
                    kernel_size=(4, 4, 5),  # H×W×T卷积核
                    stride=(2, 2, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(64),
                nn.ELU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(
                    in_channels=64, 
                    out_channels=128,  # 扩展特征通道
                    kernel_size=(3, 3, 3),  # H×W×T卷积核
                    stride=(2, 2, 2),  # 保持空间尺寸不变
                    
                ),
                nn.BatchNorm3d(128),
                nn.Dropout3d(0.2),
                nn.ReLU(inplace=True),
            )
        self.branches = nn.ModuleList([conv1, conv2, conv3])
                 
        self.clf = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
        import copy
        self.clf2 = copy.deepcopy(self.clf)
        self.clf3 = copy.deepcopy(self.clf)
        self.initialize_weights()

    def forward(self, x):
        x = self.restore2d(x)  # 恢复为2D形状 [B, 1, 11, 12, 256]
        # x: [B, 1, 11, 12, 256]
        B, C, H, W, T = x.shape
        x = x.contiguous()  # [B, T, H, W]
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            branch_outputs.append(out.mean((2, 3, 4)))
        
        # outs = [self.clf(i)for i in branch_outputs]
        outs = [self.clf(branch_outputs[0]), self.clf2(branch_outputs[1]), self.clf3(branch_outputs[2])]
        features = torch.stack(branch_outputs).mean(0)
        out = torch.stack(outs).mean(0)
        outs.append(out)
        return outs, F.normalize(features, p=2, dim=-1)  # 特征归一化   
      
    
# a = Spatial3dconvMultiscale_v3(3,3)
# input = torch.randn(64, 128, 256)
# a(input)



class Spatial3dconv_v2(Spatial2dconv):
    # Spatial2dconv
    def __init__(self, 
                 in_channels: 3,
                 out_channels: 3,
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 pool_type='avg',
                 ):
        super(Spatial2dconv, self).__init__()
        self.electrod_index = electrod_index
        self.valid_pos = (self.electrod_index!=torch.inf)
        self.conv1 = nn.Sequential(
                    nn.Conv3d(
                        in_channels=1, 
                        out_channels=64,  # 扩展特征通道
                        kernel_size=(4, 4, 25),  # H×W×T卷积核
                        stride=(2, 2, 3),  # 保持空间尺寸不变
                        
                    ),
                    nn.BatchNorm3d(64),
                    nn.ELU(inplace=True),
                    nn.Dropout3d(0.1),
                    # nn.Conv3d(
                    #     in_channels=32, 
                    #     out_channels=64,  # 扩展特征通道
                    #     kernel_size=(4, 5, 18),  # H×W×T卷积核
                    #     stride=(2, 2, 2),  # 保持空间尺寸不变
                        
                    # ),
                )
        self.conv2 = nn.Sequential(
                    # nn.Conv3d(
                    #     in_channels=1, 
                    #     out_channels=32,  # 扩展特征通道
                    #     kernel_size=(7, 7, 25),  # H×W×T卷积核
                    #     stride=(3, 3, 13),  # 保持空间尺寸不变
                        
                    # ),
                    # nn.BatchNorm3d(32),
                    # nn.ELU(inplace=True),
                    # nn.Dropout3d(0.3),
                    # nn.Conv3d(
                    #     in_channels=32, 
                    #     out_channels=64,  # 扩展特征通道
                    #     kernel_size=(2, 2, 18),  # H×W×T卷积核
                    #     stride=(2, 2, 2),  # 保持空间尺寸不变
                        
                    # ),
                    nn.Identity(),  # 占位符，实际使用时可以替换为其他卷积层
                )
        self.conv3 = nn.Sequential(
                    # nn.Conv3d(
                    #     in_channels=1, 
                    #     out_channels=32,  # 扩展特征通道
                    #     kernel_size=(11, 12, 25),  # H×W×T卷积核
                    #     stride=(3, 3, 13),  # 保持空间尺寸不变
                        
                    # ),
                    # nn.BatchNorm3d(32),
                    # nn.ELU(inplace=True),
                    # nn.Dropout3d(0.3),
                    # nn.Conv3d(
                    #     in_channels=32, 
                    #     out_channels=64,  # 扩展特征通道
                    #     kernel_size=(1, 1, 18),  # H×W×T卷积核
                    #     stride=(2, 2, 2),  # 保持空间尺寸不变
                        
                    # ),
                    nn.Identity(),
                )
        self.clf = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes)
        )
            
        self.initialize_weights()
        
    def forward(self, x):
        x = self.restore2d(x)  # 恢复为2D形状 [B, 1, 11, 12, 256]
        # x: [B, 1, 11, 12, 256]
        B, C, H, W, T = x.shape
        x = x.contiguous()  # [B, T, H, W]
        x1 = self.conv1(x)
        # x2 = self.conv2(x) 
        # x3 = self.conv3(x)
        features = torch.concat([torch.mean(i,(2,3,4)) for i in [x1]], dim=1)  # [B, C, H, W, T]\
        # x = torch.mean(x, dim=(2, 3, 4))  # [B, C, T] -> [B, C, T]
        x = self.clf(features)
        if not self.training:
            # x = F.softmax(x, dim=-1)
            x = F.sigmoid(x)
        return x, F.normalize(features, p=2, dim=-1)  # 特征归一化   
      
    
    def restore2d(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        x_2d = torch.zeros((B, 132, T), device=x.device, dtype=x.dtype)
        x_2d[:, self.valid_pos.flatten()] = x[:,(self.electrod_index[self.valid_pos]-1).long()]
        return x_2d.view(B,1,11,12,T)


class Spatial3dconv_v3(Spatial2dconv):
    # Spatial2dconv
    def __init__(self, 
                 in_channels: 3,
                 out_channels: 3,
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 pool_type='avg',
                 ):
        super(Spatial2dconv, self).__init__()
        self.electrod_index = electrod_index
        self.valid_pos = (self.electrod_index!=torch.inf)
        self.conv1 = nn.Sequential(
                    nn.Conv3d(
                        in_channels=1, 
                        out_channels=32,  # 扩展特征通道
                        kernel_size=(11, 12, 25),  # H×W×T卷积核
                        stride=(2, 2, 3),  # 保持空间尺寸不变
                        
                    ),
                    nn.BatchNorm3d(32),
                    nn.ELU(inplace=True),
                    nn.Dropout3d(0.1),
                    nn.Conv3d(
                        in_channels=32, 
                        out_channels=64,  # 扩展特征通道
                        kernel_size=(1, 1, 3),  # H×W×T卷积核
                        stride=(2, 2, 2),  # 保持空间尺寸不变
                        
                    ),
                )
        self.conv2 = nn.Sequential(
                    nn.Conv3d(
                        in_channels=1, 
                        out_channels=32,  # 扩展特征通道
                        kernel_size=(11, 12, 10),  # H×W×T卷积核
                        stride=(2, 2, 3),  # 保持空间尺寸不变
                        
                    ),
                    nn.BatchNorm3d(32),
                    nn.ELU(inplace=True),
                    nn.Dropout3d(0.1),
                    nn.Conv3d(
                        in_channels=32, 
                        out_channels=64,  # 扩展特征通道
                        kernel_size=(1, 1, 3),  # H×W×T卷积核
                        stride=(2, 2, 2),  # 保持空间尺寸不变
                        
                    ),
                )
        self.conv3 = nn.Sequential(
                    nn.Conv3d(
                        in_channels=1, 
                        out_channels=32,  # 扩展特征通道
                        kernel_size=(11, 12, 7),  # H×W×T卷积核
                        stride=(2, 2, 3),  # 保持空间尺寸不变
                        
                    ),
                    nn.BatchNorm3d(32),
                    nn.ELU(inplace=True),
                    nn.Dropout3d(0.1),
                    nn.Conv3d(
                        in_channels=32, 
                        out_channels=64,  # 扩展特征通道
                        kernel_size=(1, 1, 3),  # H×W×T卷积核
                        stride=(2, 2, 2),  # 保持空间尺寸不变
                        
                    ),
                )
        self.clf = nn.Sequential(
            nn.Linear(192, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes)
        )
            
        self.initialize_weights()


    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        std = x.std(dim=-1, keepdim=True) * std_scaling
        noise = torch.randn_like(mean) * std + mean
        x = x + noise
        return x

    def initialize_weights(self):
        self.apply(self._init_weights)


    def forward(self, x):
        x = self.restore2d(x)  # 恢复为2D形状 [B, 1, 11, 12, 256]
        # x: [B, 1, 11, 12, 256]
        B, C, H, W, T = x.shape
        x = x.contiguous()  # [B, T, H, W]
        x1 = self.conv1(x)
        x2 = self.conv2(x) 
        x3 = self.conv3(x)
        features = torch.concat([torch.mean(i,(2,3,4)) for i in [x1,x2,x3]], dim=1)  # [B, C, H, W, T]\
        # x = torch.mean(x, dim=(2, 3, 4))  # [B, C, T] -> [B, C, T]
        x = self.clf(features)
        if not self.training:
            # x = F.softmax(x, dim=-1)
            x = F.sigmoid(x)
        return x, F.normalize(features, p=2, dim=-1)  # 特征归一化   
      
    
    def restore2d(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        x_2d = torch.zeros((B, 132, T), device=x.device, dtype=x.dtype)
        x_2d[:, self.valid_pos.flatten()] = x[:,(self.electrod_index[self.valid_pos]-1).long()]
        return x_2d.view(B,1,11,12,T)

# a = Spatial3dconv_v2(3,3)
# input = torch.randn(64, 128, 256)
# a(input)

# from dc1d.nn import DeformConv1d, PackedDeformConv1d
# conv = PackedDeformConv1d(
#                 in_channels = 128,
#                 out_channels = 55,
#                 kernel_size = 3,
#             )
# input = torch.randn(64, 128, 256)
# conv(input)

class ResNetv2(nn.Module):
    def __init__(self, in_channels: int,
                 conv1_channels: int,
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 out_channels = 64,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 ):
        super(ResNetv2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, conv1_channels, kernel_size=25, stride=3, padding=0),
            nn.BatchNorm1d(conv1_channels),
            nn.ELU(inplace=False),
            nn.Dropout1d(0.1),
            nn.Conv1d(conv1_channels, out_channels, kernel_size=3, stride=2, padding=padding),
            nn.BatchNorm1d(out_channels),
        )
        self.mlp = Mlp(out_channels,
                       hidden_features=32,
                       out_features=n_classes,
                       act_layer=act_layer,
                       drop=drop_out,
                       norm_layer=nn.LayerNorm)
        # self.mlp = nn.Linear(128, n_classes)
        # self.weight = nn.Parameter(torch.ones(3), requires_grad=True)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Parameter):
            # 你的初始化代码，例如：
            nn.init.constant_(m, 1.0)
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


    def forward(self, x: Tensor) -> Tensor:
        # if self.training:
        #     x = self.add_noise(x)
        x = torch.mean(self.conv1(x),2)
        # x, features = self.mhs_clf(self.conv1(x))
        # x2 = self.conv2(x)
        # x3 = self.conv3(x2)
        # x = torch.stack([torch.mean(i,(2)) for i in [x, x2, x3]], dim=1)
        # x = x*self.weight[None,:,None]

        features = x.view(x.size(0), -1) # 512 1 -》 512
        x = self.mlp(features)
        if not self.training:
            x = F.softmax(x, dim=-1)
            # x = F.sigmoid(x)
        return x, F.normalize(features, p=2, dim=-1)  # 特征归一化

    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        # mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        # std = x.std(dim=-1, keepdim=True) * std_scaling
        # noise = torch.randn_like(mean) * std + mean
        # x = x + noise
        return x

    def update_class_weights(self, y=None, weights=None):
        """更新类别权重"""
        if weights is not None:
            self._class_weights = weights.to(self.classifier.fc1.weight.device)
        elif y is not None:
            self.configure_loss(y)

    def configure_loss(self, y: torch.Tensor):
        """动态计算类别权重"""
        counts = torch.bincount(y)
        weights = 1.0 / (counts.float() + 1e-8)
        weights /= weights.sum()
        # self._class_weights = weights.to(self.classifier.fc1.weight.device)

    def compute_loss(self, outputs, targets):
        # 加权损失计算
        loss_func = nn.CrossEntropyLoss()
        weighted_loss = loss_func(outputs, targets)

        return weighted_loss

# a = ResNetv2(128,256)
# input = torch.randn(64, 128, 256)
# a(input)
class ResNetv2_group(nn.Module):
    def __init__(self, in_channels: int,
                 conv1_channels: int,
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 ):
        super(ResNetv2_group, self).__init__()
        idx_bound = np.load('viz_channel_allday-allclass-cluster_tree-png/idx_bound.npy')
        idx = np.load('viz_channel_allday-allclass-cluster_tree-png/idx.npy')
        self.part1 = torch.tensor(idx[:41],dtype=torch.long)
        self.part2 = torch.tensor(idx[41:88],dtype=torch.long)
        self.part3 = torch.tensor(idx[88:128],dtype=torch.long)
        self.g1 = nn.Sequential(
            nn.Conv1d(len(self.part1), conv1_channels, kernel_size=25, stride=3, padding=0),
            nn.BatchNorm1d(conv1_channels),
            nn.ELU(inplace=False),
            nn.Dropout1d(0.1),
            nn.Conv1d(conv1_channels, 64, kernel_size=3, stride=2, padding=padding),
            nn.BatchNorm1d(64),
        )
        self.g2 = nn.Sequential(
            nn.Conv1d(len(self.part2), conv1_channels, kernel_size=25, stride=3, padding=0),
            nn.BatchNorm1d(conv1_channels),
            nn.ELU(inplace=False),
            nn.Dropout1d(0.1),
            nn.Conv1d(conv1_channels, 64, kernel_size=3, stride=2, padding=padding),
            nn.BatchNorm1d(64),
        )
        self.g3 = nn.Sequential(
            nn.Conv1d(len(self.part3), conv1_channels, kernel_size=25, stride=3, padding=0),
            nn.BatchNorm1d(conv1_channels),
            nn.ELU(inplace=False),
            nn.Dropout1d(0.1),
            nn.Conv1d(conv1_channels, 64, kernel_size=3, stride=2, padding=padding),
            nn.BatchNorm1d(64),
        )
        self.g_all = nn.Sequential(
            nn.Conv1d(128, conv1_channels, kernel_size=25, stride=3, padding=0),
            nn.BatchNorm1d(conv1_channels),
            nn.ELU(inplace=False),
            nn.Dropout1d(0.1),
            nn.Conv1d(conv1_channels, 64, kernel_size=3, stride=2, padding=padding),
            nn.BatchNorm1d(64),
        )
        self.mlp = Mlp(64,
                       hidden_features=32,
                       out_features=n_classes,
                       act_layer=act_layer,
                       drop=drop_out,
                       norm_layer=nn.LayerNorm)
        # self.mlp = nn.Linear(128, n_classes)
        # self.weight = nn.Parameter(torch.ones(3), requires_grad=True)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Parameter):
            # 你的初始化代码，例如：
            nn.init.constant_(m, 1.0)
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


    def forward(self, x: Tensor) -> Tensor:
        # if self.training:
        #     x = self.add_noise(x)
        g1_input = x[:,self.part1,:]
        g2_input = x[:,self.part2,:]
        g3_input = x[:,self.part3,:]
        feat_g1 = torch.mean(self.g1(g1_input),2)
        feat_g2 = torch.mean(self.g2(g2_input),2)
        feat_g3 = torch.mean(self.g3(g3_input),2)
        feat_gall = torch.mean(self.g_all(x),2)
        feat_all = feat_g1 + feat_g2 + feat_g3+feat_gall
        x = [self.mlp(feat_g1), self.mlp(feat_g2), self.mlp(feat_g3),self.mlp(feat_gall),self.mlp(feat_all)]
        return x, F.normalize(feat_all, p=2, dim=-1)  # 特征归一化

    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        std = x.std(dim=-1, keepdim=True) * std_scaling
        noise = torch.randn_like(mean) * std + mean
        x = x + noise
        return x

    def update_class_weights(self, y=None, weights=None):
        """更新类别权重"""
        if weights is not None:
            self._class_weights = weights.to(self.classifier.fc1.weight.device)
        elif y is not None:
            self.configure_loss(y)

    def configure_loss(self, y: torch.Tensor):
        """动态计算类别权重"""
        counts = torch.bincount(y)
        weights = 1.0 / (counts.float() + 1e-8)
        weights /= weights.sum()
        # self._class_weights = weights.to(self.classifier.fc1.weight.device)

    def compute_loss(self, outputs, targets):
        # 加权损失计算
        loss_func = nn.CrossEntropyLoss()
        weighted_loss = loss_func(outputs, targets)

        return weighted_loss


# a = ResNetv2_group(128,256)
# input = torch.randn(64, 128, 256)
# a(input)


class M1S1(nn.Module):
    def __init__(self, 
                 m1_mapping: int,
                 s1_mapping: int,
                 conv1_channels: int,
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 ):
        super(M1S1, self).__init__()
        self.m1_mapping = m1_mapping
        self.s1_mapping = s1_mapping
        m1_num_c = len(self.m1_mapping)
        s1_num_c = len(self.s1_mapping)
        self.conv_m1 = nn.Sequential(
            nn.Conv1d(m1_num_c, conv1_channels, kernel_size=25, stride=3, padding=0),
            nn.BatchNorm1d(conv1_channels),
            nn.ELU(inplace=False),
            nn.Dropout1d(0.1),
            nn.Conv1d(conv1_channels, 128, kernel_size=3, stride=2, padding=padding),
            nn.BatchNorm1d(128),
        )
        self.conv_s1 = nn.Sequential(
            nn.Conv1d(s1_num_c, conv1_channels, kernel_size=25, stride=3, padding=0),
            nn.BatchNorm1d(conv1_channels),
            nn.ELU(inplace=False),
            nn.Dropout1d(0.1),
            nn.Conv1d(conv1_channels, 128, kernel_size=3, stride=2, padding=padding),
            nn.BatchNorm1d(128),
        )
        self.mlp = Mlp(128,
                       hidden_features=256,
                       out_features=n_classes,
                       act_layer=act_layer,
                       drop=drop_out,
                       norm_layer=nn.LayerNorm)
        # self.mlp = nn.Linear(128, n_classes)
        # self.weight = nn.Parameter(torch.ones(3), requires_grad=True)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


    def forward(self, x: Tensor) -> Tensor:
        # if self.training:
        #     x = self.add_noise(x)
        m1_feat = torch.mean(self.conv_m1(x[:,self.m1_mapping]),2)
        s1_feat = torch.mean(self.conv_s1(x[:,self.s1_mapping]),2)
        
        # x, features = self.mhs_clf(self.conv1(x))
        # x2 = self.conv2(x)
        # x3 = self.conv3(x2)
        # x = torch.stack([torch.mean(i,(2)) for i in [x, x2, x3]], dim=1)
        # x = x*self.weight[None,:,None]

        features = m1_feat+s1_feat # 512 1 -》 512
        x = [self.mlp(s1_feat), self.mlp(m1_feat), self.mlp(features)]
        # if not self.training:
        #     x = F.softmax(x, dim=-1)
            # x = F.sigmoid(x)
        return x, F.normalize(features, p=2, dim=-1)  # 特征归一化

    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        std = x.std(dim=-1, keepdim=True) * std_scaling
        noise = torch.randn_like(mean) * std + mean
        x = x + noise
        return x

    def update_class_weights(self, y=None, weights=None):
        """更新类别权重"""
        if weights is not None:
            self._class_weights = weights.to(self.classifier.fc1.weight.device)
        elif y is not None:
            self.configure_loss(y)

    def configure_loss(self, y: torch.Tensor):
        """动态计算类别权重"""
        counts = torch.bincount(y)
        weights = 1.0 / (counts.float() + 1e-8)
        weights /= weights.sum()
        # self._class_weights = weights.to(self.classifier.fc1.weight.device)

    def compute_loss(self, outputs, targets):
        # 加权损失计算
        loss_func = nn.CrossEntropyLoss()
        weighted_loss = loss_func(outputs, targets)

        return weighted_loss







class Simple1d(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 c1_kernel_num: int,
                 c1_kernel_size: int,
                 c1_stride: int,
                 
                 c2_kernel_num: int,
                 c2_kernel_size: int,
                 c2_stride: int,

                 c3_kernel_num: int,
                 c3_kernel_size: int,
                 c3_stride: int,

                 activation: str,
                 n_classes: int = 4,
                 ):
        super(Simple1d, self).__init__()
        if activation =='elu':
            act_layer = nn.ELU()
        if activation =='relu':
            act_layer = nn.ReLU()
        if activation =='leakyrelu':
            act_layer = nn.LeakyReLU()
        if activation =='prelu':
            act_layer = nn.PReLU()
        if activation =='gelu':
            act_layer = nn.GELU()
        if activation =='silu':
            act_layer = nn.SiLU()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, c1_kernel_num, kernel_size=c1_kernel_size, stride=c1_stride, ),
            nn.BatchNorm1d(c1_kernel_num),
            act_layer,
            nn.Dropout1d(0.1),

            nn.Conv1d(c1_kernel_num, c2_kernel_num, kernel_size=c2_kernel_size, stride=c2_stride, ),
            nn.BatchNorm1d(c2_kernel_num),
            act_layer,

            nn.Conv1d(c2_kernel_num, c3_kernel_num, kernel_size=c3_kernel_size, stride=c3_stride, ),
            nn.BatchNorm1d(c3_kernel_num),
            act_layer,
        )
        self.mlp = nn.Linear(c3_kernel_num, n_classes)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


    def forward(self, x: Tensor) -> Tensor:
        x = torch.mean(self.conv(x),2)
        features = x.view(x.size(0), -1) # 512 1 -》 512
        x = self.mlp(features)
        return x, F.normalize(features, p=2, dim=-1)  # 特征归一化

    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        std = x.std(dim=-1, keepdim=True) * std_scaling
        noise = torch.randn_like(mean) * std + mean
        x = x + noise
        return x


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
        
        self.clf = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, num_class),
        )
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
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
            branch_outputs.append(out)
        
        # 对齐特征尺度（上采样较小尺度分支）
        max_time = branch_outputs[0].shape[2]  # 最大尺度分支的时间维度
        aligned_outputs = [branch_outputs[0]]
        for i, out in enumerate(branch_outputs[1:]):
            aligned_outputs.append(F.interpolate(out, size=max_time, mode='linear'))
        aligned_outputs = torch.stack(aligned_outputs).permute(1,0,2,3)  # [batch, branches, channels, time]
        # 加权融合（归一化权重）
        # weights = F.softmax(aligned_outputs.mean(3), dim=0)
        features  = aligned_outputs.mean((1,3))
        out = self.clf(features)
        return out, F.normalize(features, p=2, dim=-1)
    
class MultiScale1DCNN_v3(MultiScale1DCNN):
    def __init__(self, in_channels, out_channels=128, scales=[5, 10, 25], num_class=4):
        super().__init__(in_channels, out_channels, scales, num_class)
        self.clf = nn.Sequential(
            nn.Linear(out_channels*3, out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, num_class),
        )
    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            branch_outputs.append(out)
        
        features = torch.concat([i.mean(-1) for i in branch_outputs], dim=1)
        out = self.clf(features)
        return out, F.normalize(features, p=2, dim=-1)
    

model = MultiScale1DCNN_v3(in_channels=64, out_channels=128)
input_tensor = torch.randn(32, 64, 100)
output = model(input_tensor)  # 输出: [32, 128,
# a = ResNetv2(128,256)
# input = torch.randn(64, 128, 256)
# a(input)
class TimeAttention(nn.Module):
    def __init__(self, channels, num_classes, num_heads=4, attn_dim=None, max_len=512):
        super().__init__()
        self.channels = channels
        self.attn_dim = attn_dim or channels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.attn_dim))
        self.proj = nn.Linear(channels, self.attn_dim) if self.attn_dim != channels else nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dim=self.attn_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(self.attn_dim)
        self.head = nn.Linear(self.attn_dim, num_classes)
        self.max_len = max_len
        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, self.attn_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


    def forward(self, x):
        x = x.transpose(1, 2)  # [batch, time, channels]
        x = self.proj(x)       # [batch, time, attn_dim]
        B, T, _ = x.shape
        cls_token = self.cls_token.expand(B, -1, -1)  # [batch, 1, attn_dim]
        x = torch.cat([cls_token, x], dim=1)          # [batch, 1+time, attn_dim]
        # 截断或补齐位置编码
        pos = self.pos_embed[:, :x.size(1), :]        # [1, 1+time, attn_dim]
        x, _ = self.attn(x+pos, x, x)                   # [batch, 1+time, attn_dim]
        x = self.norm(x)
        cls_token = x[:, 0]                                 # [batch, attn_dim]
        out = self.head(cls_token)                          # [batch, num_classes]
        return out, cls_token
    


class Query_bank(nn.Module):
    def __init__(self, num_query, query_dim, num_heads=4):
        super().__init__()
        # 验证参数有效性
        assert query_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        # 创建固定的query参数（128个256维向量）
        self.query_params = nn.Parameter(torch.randn(num_query, query_dim))
        
        # PyTorch内置的多头注意力层
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            batch_first=True  # 使用(batch, seq, feature)格式
        )

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状为(batch_size, channel, time) = (B, 128, 256)
        :return: 输出张量，形状为(batch_size, channel, time) = (B, 128, 256)
        """
        batch_size, channel_num, time_len = x.shape
        
        # 调整输入维度：(B, 128, 256) -> (B, 128, 256)
        # 这里channel=128作为序列长度，time=256作为特征维度
        k = v = x  # 输入作为K和V
        
        # 创建query：重复固定的query参数到batch中
        # query_params形状: (128, 256) -> 扩展为: (B, 128, 256)
        q = self.query_params.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 应用多头注意力机制
        attn_output, attn_weights = self.multihead_attn(
            query=q,    # (B, num_query, time_len)
            key=k,      # (B, channel_num, time_len)  
            value=v,    # (B, channel_num, time_len)
        )
        return attn_output  # 输出形状: (B, num_query, time_len)

class Conv1dQbank(ResNetv2):
    def __init__(self, 
                 in_channels: int,
                 conv1_channels: int,
                 num_query:int = 8, 
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 ):
        super(Conv1dQbank, self).__init__(in_channels, conv1_channels)
        self.qbank = Query_bank(num_query, 256)
    def forward(self, x: Tensor) -> Tensor:
        x = self.qbank(x)  # B, 8, 256
        x = torch.mean(self.conv1(x),2)
        features = x.view(x.size(0), -1) # 512 1 -》 512
        x = self.mlp(features)
        if not self.training:
            x = F.softmax(x, dim=-1)
            # x = F.sigmoid(x)
        return x, F.normalize(features, p=2, dim=-1)  # 特征归一化




a = Query_bank(8, 256)
input = torch.randn(64, 128, 256)
a(input)





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

        # 对齐特征尺度（上采样较小尺度分支）
        # max_time = branch_outputs[0].shape[2]
        # aligned_outputs = [branch_outputs[0]]
        # for out in branch_outputs[1:]:
        #     aligned_outputs.append(F.interpolate(out, size=max_time, mode='linear'))
        # [num_branches, batch, channels, time] -> [batch, num_branches, channels, time]
        # features = torch.stack(aligned_outputs, 1).mean((1,3))

    
        outs = [self.clf(i)for i in branch_outputs]
        features = torch.stack(branch_outputs).mean(0)
        out = self.clf(features)
        outs.append(out)
        return outs, F.normalize(features, p=2, dim=-1)

model = MultiScale1DCNN_v2(in_channels=64, out_channels=128)
input_tensor = torch.randn(32, 64, 100)
output = model(input_tensor)







class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=128, kernel_size=25):
        super().__init__()
        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1),
                nn.BatchNorm1d(out_channels),
                # nn.Dropout1d(0.2),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=1),
                nn.BatchNorm1d(out_channels),
                nn.Dropout1d(0.2),
                nn.ReLU()
            )
        self.conv2 = nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2),
                nn.BatchNorm1d(out_channels),
                # nn.Dropout1d(0.2),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=1),
                nn.BatchNorm1d(out_channels),
                nn.Dropout1d(0.2),
                nn.ReLU()
            )


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = x1.mean(-1) + x2.mean(-1)
        return x

model = ResBlock(in_channels=64, out_channels=128)
input_tensor = torch.randn(32, 64, 100)
output = model(input_tensor)

class MultiScale1DCNN_v4(nn.Module):
    def __init__(self, in_channels, out_channels=128, scales=[5, 15, 25], num_class=4):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in scales:
            branch = ResBlock(in_channels=in_channels,out_channels=out_channels,kernel_size=k)
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
            branch_outputs.append(out)  # 对每个分支的输出在时间维度上取平均

        # 对齐特征尺度（上采样较小尺度分支）
        # max_time = branch_outputs[0].shape[2]
        # aligned_outputs = [branch_outputs[0]]
        # for out in branch_outputs[1:]:
        #     aligned_outputs.append(F.interpolate(out, size=max_time, mode='linear'))
        # [num_branches, batch, channels, time] -> [batch, num_branches, channels, time]
        # features = torch.stack(aligned_outputs, 1).mean((1,3))

    
        outs = [self.clf(i)for i in branch_outputs]
        features = torch.stack(branch_outputs).mean(0)
        out = self.clf(features)
        outs.append(out)
        return outs, F.normalize(features, p=2, dim=-1)

# model = MultiScale1DCNN_v4(in_channels=64, out_channels=128)
# input_tensor = torch.randn(32, 64, 100)
# output = model(input_tensor)
















class CNN_all_step(nn.Module):
    def __init__(self, in_channels: int,
                 conv1_channels: int,
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 ):
        super(CNN_all_step, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, conv1_channels, kernel_size=25, stride=1, padding=0),
            nn.BatchNorm1d(conv1_channels),
            nn.ELU(inplace=False),
            nn.Dropout1d(0.1),
        )
        self.mlp = Mlp(conv1_channels,
                       hidden_features=512,
                       out_features=n_classes,
                       act_layer=act_layer,
                       drop=drop_out,
                       norm_layer=nn.LayerNorm)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


    def forward(self, x: Tensor) -> Tensor:
        # if self.training:
        #     x = self.add_noise(x)
        x = self.conv1(x).transpose(1, 2)
        # x2 = self.conv2(x)
        # x3 = self.conv3(x2)
        # x = torch.stack([torch.mean(i,(2)) for i in [x, x2, x3]], dim=1)
        # x = x*self.weight[None,:,None]
        features = x[:,-1,:] # 512 1 -》 512
        x = self.mlp(rearrange(x[:,:,:],'b t c -> (b t) c'))
        x = F.softmax(x, dim=-1)
        # if not self.training:
            # x = F.softmax(x, dim=-1)
            # x = F.sigmoid(x)
        return x, F.normalize(features, p=2, dim=-1)  # 特征归一化

    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        std = x.std(dim=-1, keepdim=True) * std_scaling
        noise = torch.randn_like(mean) * std + mean
        x = x + noise
        return x

    def update_class_weights(self, y=None, weights=None):
        """更新类别权重"""
        if weights is not None:
            self._class_weights = weights.to(self.classifier.fc1.weight.device)
        elif y is not None:
            self.configure_loss(y)

    def configure_loss(self, y: torch.Tensor):
        """动态计算类别权重"""
        counts = torch.bincount(y)
        weights = 1.0 / (counts.float() + 1e-8)
        weights /= weights.sum()
        # self._class_weights = weights.to(self.classifier.fc1.weight.device)

    def compute_loss(self, outputs, targets):
        # 加权损失计算
        loss_func = nn.CrossEntropyLoss()
        weighted_loss = loss_func(outputs, targets)

        return weighted_loss


a = CNN_all_step(128,256)
input = torch.randn(64, 128, 256)
a(input)

class CNN_LSTM(nn.Module):
    def __init__(self, in_channels: int,
                 conv1_channels: int,
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 decode_all_step = True,
                 ):
        super(CNN_LSTM, self).__init__()
        self.decode_all_step = decode_all_step
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, conv1_channels, kernel_size=25, stride=3, padding=0),
            nn.BatchNorm1d(conv1_channels),
            nn.ELU(inplace=False),
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=512, num_layers=2, batch_first=True, dropout=drop_out, bidirectional=False)

        self.mlp = Mlp(512,
                       hidden_features=512,
                       out_features=n_classes,
                       act_layer=act_layer,
                       drop=drop_out,
                       norm_layer=nn.LayerNorm)
        # self.mlp = nn.Linear(128, n_classes)
        # self.weight = nn.Parameter(torch.ones(3), requires_grad=True)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                # 输入到隐藏的权重（W_ii, W_if, W_ig, W_io）
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param)  # Xavier均匀分布[1,2](@ref)
                
                # 隐藏到隐藏的权重（W_hi, W_hf, W_hg, W_ho）
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param)  # 正交初始化保持梯度稳定性[4](@ref)
                
                # 偏置项（b_i, b_f, b_g, b_o）
                elif "bias" in name:
                    nn.init.zeros_(param)  # 先全置零
                    # 特殊处理遗忘门偏置（b_f的后1/4部分）
                    n = param.size(0)
                    param.data[n//4 : n//2].fill_(1)  # 遗忘门偏置置1[1,4](@ref)


    def forward(self, x: Tensor) -> Tensor:
        # if self.training:
        #     x = self.add_noise(x)
        x = self.conv1(x)
        output, (h_n, c_n) = self.lstm(x.transpose(1, 2))  # LSTM expects input shape (batch, seq_len, features)
        if self.decode_all_step:
            x = self.mlp(rearrange(output[:,:,:],'b t c -> (b t) c'))  # Flatten the output for MLP
            # x = rearrange(x,'(b t) c -> b t c',b=x.shape[0])  # Reshape back to (batch, seq_len, features)
            features = h_n[-1]
            if not self.training:
                # x = F.softmax(x, dim=-1)
                x = F.sigmoid(x)
            return x, F.normalize(features, p=2, dim=-1)  # 特征归一化
        else:
            features = h_n[-1]  # Get the last hidden state as features
            x = self.mlp(features)
            if not self.training:
                # x = F.softmax(x, dim=-1)
                x = F.sigmoid(x)
            return x, F.normalize(features, p=2, dim=-1)  # 特征归一化

    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        std = x.std(dim=-1, keepdim=True) * std_scaling
        noise = torch.randn_like(mean) * std + mean
        x = x + noise
        return x

    def update_class_weights(self, y=None, weights=None):
        """更新类别权重"""
        if weights is not None:
            self._class_weights = weights.to(self.classifier.fc1.weight.device)
        elif y is not None:
            self.configure_loss(y)

    def configure_loss(self, y: torch.Tensor):
        """动态计算类别权重"""
        counts = torch.bincount(y)
        weights = 1.0 / (counts.float() + 1e-8)
        weights /= weights.sum()
        # self._class_weights = weights.to(self.classifier.fc1.weight.device)

    def compute_loss(self, outputs, targets):
        # 加权损失计算
        loss_func = nn.CrossEntropyLoss()
        weighted_loss = loss_func(outputs, targets)

        return weighted_loss

a = CNN_LSTM(128,256)
input = torch.randn(64, 128, 256)
a(input)




import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeTrans(nn.Module):
    def __init__(self, input_channels, num_classes, d_model=64, nhead=4, 
                 num_layers=3, dim_feedforward=256, dropout=0.1):
        super(TimeTrans, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model)
        
        # 创建Transformer编码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 使用batch_first=True更符合常见习惯
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, d_model, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(d_model),
            nn.ELU(inplace=False),
            nn.Conv1d(d_model, d_model, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(d_model),
            nn.ELU(inplace=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, d_model, kernel_size=15, stride=3, padding=0),
            nn.BatchNorm1d(d_model),
            nn.ELU(inplace=False),
            nn.Conv1d(d_model, d_model, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(d_model),
            nn.ELU(inplace=False),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, d_model, kernel_size=25, stride=3, padding=0),
            # nn.BatchNorm1d(d_model),
            # nn.ELU(inplace=False),
            # nn.Conv1d(d_model, d_model, kernel_size=2, stride=1, padding=0),
            # nn.BatchNorm1d(d_model),
            # nn.ELU(inplace=False),
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 可学习的分类token[5](@ref)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 分类器
        self.classifier = nn.Sequential(
            # nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes)
        )
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
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
        # x形状: (batch, channels, time) -> (batch, time, channels)
        x = self.conv3(x).transpose(1, 2)
        # x1 = self.conv1(x).transpose(1, 2)
        # x2 = self.conv2(x).transpose(1, 2)
        # x3 = self.conv3(x).transpose(1, 2)
        # x = torch.concat([x1,x2,x3],dim=1)
        # 添加位置编码
        # x = self.pos_encoder(x)
        
        # # 添加可学习的cls token到序列开头[5](@ref)
        # batch_size = x.size(0)
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)  # (batch, time+1, d_model)
        
        # # 通过Transformer编码器
        x = self.transformer_decoder(tgt=x, memory=self.pos_encoder(x))  # (batch, time+1, d_model)
        
        # # 提取cls token的输出用于分类[5](@ref)
        # cls_output = x[:, 0, :]  # (batch, d_model)
        cls_output = x.mean(1) 
        # 分类
        logits = self.classifier(cls_output)
        
        return logits, F.normalize(cls_output, p=2, dim=-1)  # 返回归一化的特征
# model= TimeTrans(
#             input_channels=128,
#                 num_classes=4,      # 假设有5个类别
#                 d_model=128,
#                 nhead=8,
#                 num_layers=1,
#                 dim_feedforward=256,
#                 dropout=0.1
#             )
# input = torch.randn(64, 128, 256)
# model(input)
# 使用示例
# if __name__ == "__main__":
#     # 假设输入维度: (batch=32, channels=10, time=50)
#     model = TimeTrans(
#         input_channels=10,
#         num_classes=5,      # 假设有5个类别
#         d_model=64,
#         nhead=4,
#         num_layers=3,
#         dim_feedforward=256,
#         dropout=0.1
#     )
    
#     # 示例输入
#     input_tensor = torch.randn(32, 10, 50)  # (batch, channels, time)
#     output = model(input_tensor)
#     print(f"输出形状: {output.shape}")  # 应该是 (32, 5)



def batch_ecog_to_spectrogram(x, n_fft=128, hop_length=128, win_length=None, window=None, normalize=True):
    """
    x: torch.Tensor, shape = (batch, channels, t)
    returns: spec: torch.Tensor, shape = (batch, channels, freq_bins, time_frames)
             (power spectrogram, i.e., |STFT|^2)
    """
    if win_length is None:
        win_length = n_fft
    B, C, T = x.shape
    # reshape to (B*C, T) to compute STFT in batch
    x_reshaped = x.reshape(B*C, T)
    # compute STFT -> shape (B*C, freq_bins, time_frames), complex if return_complex=True
    stft = torch.stft(
        x_reshaped,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=False,
        pad_mode='reflect'
    )
    power = torch.abs(stft) ** 2  # power spectrogram
    # reshape back to (B, C, freq_bins, time_frames)
    freq_bins, time_frames = power.shape[-2], power.shape[-1]
    power = power.reshape(B, C, freq_bins, time_frames)
    if normalize:
        # channel-wise z-score normalization over time+freq dims
        mean = power.mean(dim=(-2, -1), keepdim=True)
        std = power.std(dim=(-2, -1), keepdim=True) + 1e-8
        power = (power - mean) / std
    return power #(batch, channels, freq_bins, time_frames)


class Spatial2dconvSpectro(Spatial2dconv):
    # Spatial2dconv
    def __init__(self, 
                 in_channels: 3,
                 out_channels: 3,
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 1,
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 pool_type='avg',
                 ):
        super(Spatial2dconv, self).__init__()
        self.electrod_index = electrod_index
        self.valid_pos = (self.electrod_index!=torch.inf)
        self.conv2d = nn.Sequential(
                    nn.Conv2d(
                        in_channels=128, 
                        out_channels=64,  # 扩展特征通道
                        kernel_size=(5, 3),  # H×W×T卷积核
                        stride=(3, 1),  # 保持空间尺寸不变
                        
                    ),
                    nn.BatchNorm2d(64),
                    nn.ELU(inplace=True),
                    nn.Dropout2d(0.1),
                    nn.Conv2d(
                        in_channels=64, 
                        out_channels=64,  # 扩展特征通道
                        kernel_size=(3, 2),  # H×W×T卷积核
                        stride=(2, 1),  # 保持空间尺寸不变
                        
                    ),
                    nn.BatchNorm2d(64),
                    nn.Dropout2d(0.2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=64, 
                        out_channels=128,  # 扩展特征通道
                        kernel_size=(3, 2),  # H×W×T卷积核
                        stride=(2, 1),  # 保持空间尺寸不变
                    
                    ),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )
            
        self.clf = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        self.initialize_weights()
        
    def forward(self, x):
        x = batch_ecog_to_spectrogram(x=x,n_fft=64, hop_length=25)
        # x = self.restore2d(x)  # 恢复为2D形状 [B, 1, 11, 12, 256]
        # x: [B, 1, 11, 12, 256]
        B, C, H, W = x.shape
        x = x.contiguous()  # [B, T, H, W]
        x = self.conv2d(x)
        features = torch.mean(x, dim=(2, 3,))
        x = self.clf(features)
        if not self.training:
            x = F.softmax(x, dim=-1)
            # x = F.sigmoid(x)
        return x, F.normalize(features, p=2, dim=-1)  # 特征归一化   
      
    
    def restore2d(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        x_2d = torch.zeros((B, 132, T), device=x.device, dtype=x.dtype)
        x_2d[:, self.valid_pos.flatten()] = x[:,(self.electrod_index[self.valid_pos]-1).long()]
        return x_2d.view(B,11,12,T).permute(0,3,1,2)


# a = Spatial2dconvSpectro(3,3)
# input = torch.randn(64, 128, 256)
# a(input)















class RegressionResNet(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 first_kernel_size: int = 25,
                 padding: Any = 'same',
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 n_regression: int = 2,
                 ):
        super(RegressionResNet, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=first_kernel_size, padding=padding),
            norm_layer(out_channels) if norm_layer == nn.BatchNorm1d else norm_layer(out_channels // 4, out_channels),
            act_layer()
        )
        layers = []
        for i in range(n_layers):
            layers.append(
                ConvBlock1D(out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            act_layer=act_layer,
                            norm_layer=norm_layer))

        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # self.linear = nn.Linear(out_channels, out_channels // 2)
        # out_channels = out_channels // 2
        # self.layer_norm = nn.LayerNorm(out_channels)
        self.mlp = Mlp(out_channels,
                       hidden_features=out_channels,
                       out_features=n_classes,
                       drop=drop_out,
                       act_layer=act_layer,
                       norm_layer=nn.LayerNorm)
        self.regression = Mlp(out_channels,
                              hidden_features=out_channels,
                              out_features=n_regression,
                              drop=0.0,
                              act_layer=act_layer,
                              norm_layer=nn.LayerNorm
                              )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # if self.training:
        #     x = self.add_noise(x)
        x = self.first_layer(x)
        x = self.layers(x)
        x = self.avg_pool(x)
        features = x.view(x.size(0), -1)
        # features = self.linear(features)  # 特征降维
        # features = self.layer_norm(features)  # 特征归一化
        x = self.mlp(features)
        position = self.regression(features)
        if not self.training:
            # x = F.softmax(x, dim=-1)
            x = F.sigmoid(x)
        return x, position, features, None  # 特征归一化

    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        std = x.std(dim=-1, keepdim=True) * std_scaling
        noise = torch.randn_like(mean) * std + mean
        x = x + noise
        return x

    def update_class_weights(self, y=None, weights=None):
        """更新类别权重"""
        if weights is not None:
            self._class_weights = weights.to(self.classifier.fc1.weight.device)
        elif y is not None:
            self.configure_loss(y)

    def configure_loss(self, y: torch.Tensor):
        """动态计算类别权重"""
        counts = torch.bincount(y)
        weights = 1.0 / (counts.float() + 1e-8)
        weights /= weights.sum()
        # self._class_weights = weights.to(self.classifier.fc1.weight.device)

    def compute_loss(self, outputs, targets):
        # 加权损失计算
        loss_func = nn.CrossEntropyLoss()
        weighted_loss = loss_func(outputs, targets)

        return weighted_loss


class TimeFFTResNet(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 first_kernel_size: int = 15,
                 padding: Any = 'same',
                 act_layer: Optional[LayerType] = nn.SiLU,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 ):
        super(TimeFFTResNet, self).__init__()

        self.time_first_layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=first_kernel_size, padding=padding),
            # nn.GroupNorm(4, out_channels),
            nn.BatchNorm1d(out_channels),
            act_layer()
        )
        layers = []
        for i in range(n_layers):
            layers.append(
                ConvBlock1D(out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            act_layer=act_layer))

        self.time_layers = nn.Sequential(*layers)

        self.fft_first_layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=first_kernel_size, padding=padding),
            # nn.GroupNorm(4, out_channels),
            nn.BatchNorm1d(out_channels),
            act_layer()
        )
        layers = []
        for i in range(n_layers):
            layers.append(
                ConvBlock1D(out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            act_layer=act_layer))

        self.fft_layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = Mlp(out_channels * 2, hidden_features=out_channels, out_features=n_classes, drop=drop_out)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            x = self.add_noise(x)
        length = x.size(-1) // 2 - 1
        fft = torch.abs(torch.fft.fft(x, dim=-1))[..., :length]  # 计算FFT并取前半部分

        x = self.time_first_layer(x)
        x = self.time_layers(x)
        x = self.avg_pool(x)

        time_features = x.view(x.size(0), -1)

        fft = self.fft_first_layer(fft)
        fft = self.fft_layers(fft)
        fft = self.avg_pool(fft)
        fft_features = fft.view(fft.size(0), -1)

        features = torch.cat((time_features, fft_features), dim=-1)
        contra_features = torch.cat((time_features.unsqueeze(1), fft_features.unsqueeze(1)), dim=1)

        x = self.mlp(features)
        if not self.training:
            # x = F.softmax(x, dim=-1)
            x = F.sigmoid(x)
        return x, F.normalize(contra_features, p=2, dim=-1)  # 特征归一化

    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        std = x.std(dim=-1, keepdim=True) * std_scaling
        noise = torch.randn_like(mean) * std + mean
        x = x + noise
        return x


class RegressionTimeFFTResNet(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 first_kernel_size: int = 15,
                 padding: Any = 'same',
                 act_layer: Optional[LayerType] = nn.SiLU,
                 norm_layer: nn.Module = nn.GroupNorm,
                 drop_out: float = 0.0,
                 n_layers: int = 8,
                 n_classes: int = 4,
                 n_regression: int = 2,
                 ):
        super(RegressionTimeFFTResNet, self).__init__()

        self.time_first_layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=first_kernel_size, padding=padding),
            norm_layer(out_channels) if norm_layer == nn.BatchNorm1d else norm_layer(out_channels // 4, out_channels),
            act_layer()
        )
        layers = []
        for i in range(n_layers):
            layers.append(
                ConvBlock1D(out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            norm_layer=norm_layer,
                            act_layer=act_layer))

        self.time_layers = nn.Sequential(*layers)

        self.fft_first_layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=first_kernel_size, padding=padding),
            norm_layer(out_channels) if norm_layer == nn.BatchNorm1d else norm_layer(out_channels // 4, out_channels),
            act_layer()
        )
        layers = []
        for i in range(n_layers):
            layers.append(
                ConvBlock1D(out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            norm_layer=norm_layer,
                            act_layer=act_layer))

        self.fft_layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.linear = nn.Linear(out_channels * 2, out_channels)

        self.mlp = Mlp(out_channels,
                       hidden_features=out_channels,
                       out_features=n_classes,
                       drop=drop_out,
                       norm_layer=nn.LayerNorm)
        self.regression = Mlp(out_channels,
                              hidden_features=out_channels,
                              out_features=n_regression,
                              drop=drop_out,
                              norm_layer=nn.LayerNorm
                              )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # if self.training:
        #     x = self.add_noise(x)
        length = x.size(-1) // 2 - 1
        fft = torch.abs(torch.fft.fft(x, dim=-1))[..., :length]  # 计算FFT并取前半部分

        x = self.time_first_layer(x)
        x = self.time_layers(x)
        x = self.avg_pool(x)

        time_features = x.view(x.size(0), -1)

        fft = self.fft_first_layer(fft)
        fft = self.fft_layers(fft)
        fft = self.avg_pool(fft)
        fft_features = fft.view(fft.size(0), -1)

        features = torch.cat((time_features, fft_features), dim=-1)
        features = self.linear(features)  # 特征降维
        # features = self.layer_norm(features)  # 特征归一化

        contra_features = torch.cat((time_features.unsqueeze(1), fft_features.unsqueeze(1)), dim=1)

        x = self.mlp(features)
        pos = self.regression(features)
        if not self.training:
            # x = F.softmax(x, dim=-1)
            x = F.sigmoid(x)
        return x, pos, features, F.normalize(contra_features, p=2, dim=-1)  # 特征归一化

    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        std = x.std(dim=-1, keepdim=True) * std_scaling
        noise = torch.randn_like(mean) * std + mean
        x = x + noise
        return x


class FourierUnit1D(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.5, groups=1):
        """
        Args:
            alpha (float): 使用FFC处理通道的比例 (0 < alpha <= 1)
        """
        super().__init__()
        self.alpha = alpha

        # FFC路径参数
        self.ffc_in = int(in_channels * alpha)  # FFC处理通道数
        self.ffc_out = int(out_channels * alpha)  # FFC输出通道数

        # 原始路径参数
        self.raw_in = in_channels - self.ffc_in  # 保留通道数
        self.raw_out = out_channels - self.ffc_out  # 原始路径输出通道数

        # FFC路径组件
        if self.ffc_in > 0:
            self.ffc_conv = nn.Conv1d(
                self.ffc_in * 2,  # 实虚拼接
                self.ffc_out * 2,  # 输出实虚分离
                kernel_size=1,
                groups=groups
            )
            self.ffc_norm = nn.BatchNorm1d(self.ffc_out * 2)
            self.ffc_act = nn.ReLU(inplace=True)

        # 原始路径组件
        if self.raw_in > 0:
            self.raw_conv = nn.Conv1d(
                self.raw_in,
                self.raw_out,
                kernel_size=1
            )
            self.raw_norm = nn.BatchNorm1d(self.raw_out)

    def forward(self, x):
        # 输入形状: (B, C_in, L)
        batch_size, _, seq_len = x.shape

        # 通道分割
        x_ffc, x_raw = torch.split(
            x,
            [self.ffc_in, self.raw_in],
            dim=1
        )  # x_ffc: (B, C_ffc_in, L), x_raw: (B, C_raw_in, L)

        # FFC路径处理
        if self.ffc_in > 0:
            # 步骤1: 傅里叶变换
            x_fft = torch.fft.rfft(x_ffc, norm='ortho')  # (B, C_ffc_in, L//2+1)

            # 步骤2: 实虚分离与拼接
            x_fft_real_imag = torch.view_as_real(x_fft)  # (B, C_ffc_in, L//2+1, 2)
            real_imag_combined = x_fft_real_imag.permute(0, 1, 3, 2).contiguous()
            real_imag_combined = real_imag_combined.view(
                batch_size,
                -1,
                x_fft.size(-1)
            )  # (B, C_ffc_in*2, L//2+1)

            # 步骤3: 频域卷积
            ffc_processed = self.ffc_conv(real_imag_combined)
            ffc_processed = self.ffc_norm(ffc_processed)
            ffc_processed = self.ffc_act(ffc_processed)

            # 步骤4: 分割实虚部分
            real_out, imag_out = torch.chunk(ffc_processed, 2, dim=1)
            complex_feat = torch.stack([real_out, imag_out], dim=-1)
            complex_feat = torch.view_as_complex(complex_feat.contiguous())

            # 步骤5: 逆变换恢复时域
            restored_ffc = torch.fft.irfft(
                complex_feat,
                n=seq_len,
                norm='ortho'
            )  # (B, C_ffc_out, L)
        else:
            restored_ffc = torch.zeros(
                batch_size, self.ffc_out, seq_len,
                device=x.device
            )

        # 原始路径处理
        if self.raw_in > 0:
            raw_processed = self.raw_conv(x_raw)
            raw_processed = self.raw_norm(raw_processed)
        else:
            raw_processed = torch.zeros(
                batch_size, self.raw_out, seq_len,
                device=x.device
            )

        # 双路径特征合并
        output = torch.cat([restored_ffc, raw_processed], dim=1)
        return output  # (B, C_out, L)


class ResNet_FFC(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: Any = 'same',
                 act_layer: Optional[LayerType] = nn.GELU,
                 n_layers: int = 8,
                 n_classes: int = 5,
                 sampling_enabled=True,
                 fintune=False,  # 微调模式，微调最后一层
                 ):
        super(ResNet_FFC, self).__init__()
        self.sampling_enabled = sampling_enabled
        self.resampling = Resample(orig_freq=600, new_freq=256)
        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=padding),
            nn.GroupNorm(4, out_channels),
            act_layer()
        )
        self.fouri1 = FourierUnit1D(in_channels, in_channels)
        self.fouri2 = FourierUnit1D(out_channels, out_channels)
        layers = []
        for i in range(n_layers):
            layers.append(
                ConvBlock1D(out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            act_layer=act_layer))

        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = Mlp(out_channels, hidden_features=out_channels, out_features=n_classes)
        self.vec = nn.Linear(out_channels, 2)

        self.initialize_weights()

        if fintune:
            self.fintuned_enable()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            x = self.add_noise(x)
        if x.shape[-1] != 256 and self.sampling_enabled:
            x = self.resampling(x)
        x = self.fouri1(x)
        x = self.first_layer(x)
        x = self.fouri2(x)
        x = self.layers(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        latent = x
        vec = self.vec(x)
        x = self.mlp(x)

        if not self.training:
            x = F.sigmoid(x)
        return x, latent, vec

    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        std = x.std(dim=-1, keepdim=True) * std_scaling
        noise = torch.randn_like(mean) * std + mean
        x = x + noise
        return x

    def fintuned_enable(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.avg_pool.parameters():
            param.requires_grad = True
        for param in self.vec.parameters():
            param.requires_grad = True
        for param in self.mlp.parameters():
            param.requires_grad = True


class FilterBank(nn.Module):
    def __init__(self, freqs: dict = None,
                 sfreq: float = 256.0):
        super(FilterBank, self).__init__()
        if freqs is None:
            freqs = {'alpha': (8, 12), 'beta': (12, 25), 'gamma': (30, 50)}
        self.freqs = freqs
        self.sfreq = sfreq
        self.register_buffer('filters', self.create_filters())
        self.reflection_pad = nn.ReflectionPad1d((self.filters.shape[-1] // 2, self.filters.shape[-1] // 2))

    def create_filters(self):
        filters = []
        max_length = 0
        for key, (l_freq, h_freq) in self.freqs.items():
            f = create_filter(None, self.sfreq, l_freq=l_freq, h_freq=h_freq)
            f = f.astype(np.float32)
            if f.shape[0] > max_length:
                max_length = f.shape[0]
            filters.append(torch.from_numpy(f).float())
        # Pad filters to the same length with two-sided padding
        filters = [F.pad(f, ((max_length - f.shape[0]) // 2, (max_length - f.shape[0]) // 2), mode='constant', value=0)
                   for f in filters]
        return torch.stack(filters, dim=0).unsqueeze(1).unsqueeze(1)

    def forward(self, x: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        x = self.reflection_pad(x)
        x = x.unsqueeze(1)  # 添加通道维度
        x = F.conv2d(x, self.filters, padding='valid')
        return x


class FeatureResNet(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 first_kernel_size: int = 3,
                 kernel_size: int = 3,
                 padding: Any = 'same',
                 act_layer: Optional[LayerType] = nn.SiLU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 n_layers: int = 1,
                 n_classes: int = 4,
                 decimation: int = 32,
                 ):
        super(FeatureResNet, self).__init__()

        # widths = np.logspace(0.001, 1, 12) * 5
        widths = np.array([8, 12, 20, 30, 50, 100, 200])
        num_width = len(widths)
        filters = torch.ones(1, 1, 1, 1, decimation) / decimation
        self.decimation = decimation
        self.register_buffer('filters', filters)
        self.morlet = Morlet(freqs=widths, sfreq=512)

        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels * num_width, out_channels, kernel_size=first_kernel_size,
                      padding=first_kernel_size // 2,
                      stride=1),
            norm_layer(out_channels) if norm_layer == nn.BatchNorm1d else norm_layer(out_channels // 4, out_channels),
            act_layer()
        )

        self.se_block = SqueezeExcitation1D(in_channels * num_width, reduction=4)

        layers = []
        for i in range(n_layers):
            layers.append(
                ConvBlock1D(out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            act_layer=act_layer,
                            norm_layer=norm_layer))

        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.out = Mlp(out_channels, hidden_features=out_channels, act_layer=act_layer, out_features=n_classes)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.morlet(x)
        x = F.conv3d(x.unsqueeze(1), self.filters, stride=(1, 1, self.decimation))
        # x = torch.log10(x)
        x = x.squeeze(1).permute(0, 2, 1, 3)  # (B, C, T, F)
        x = (x - x.mean(dim=(-2, -1), keepdim=True)) / x.std(dim=(-2, -1), keepdim=True)
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, C * F, T)

        # excitation = self.se_block(x)
        # x = x * excitation  # Squeeze-and-Excitation
        x = self.first_layer(x)

        x = self.layers(x)
        x = self.avg_pool(x)
        features = x.view(x.size(0), -1)
        x = self.out(features)
        return x, features

    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        std = x.std(dim=-1, keepdim=True) * std_scaling
        noise = torch.randn_like(mean) * std + mean
        x = x + noise
        return x


class AdaptiveFilterBankResNet(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 first_kernel_size: int = 15,
                 kernel_size: int = 3,
                 filter_kernel_size: int = 65,
                 padding: Any = 'same',
                 act_layer: Optional[LayerType] = nn.ELU,
                 norm_layer: nn.Module = nn.BatchNorm1d,
                 n_layers: int = 1,
                 n_classes: int = 4,
                 n_channels: int = 128,

                 sfreq: int = 256,
                 decimation: int = 2,
                 freqs: dict = None,
                 adaptive: bool = True,
                 ):
        super(AdaptiveFilterBankResNet, self).__init__()
        if freqs is None:
            freqs = {'alpha': (8, 12), 'beta': (12, 25), 'gamma': (30, 50),
                     'gamma2': (50, 100)}

        self.decimation = decimation

        if adaptive:
            self.filter_bank = SincConv_fast(out_channels=in_channels, kernel_size=filter_kernel_size,
                                             sample_rate=sfreq,
                                             min_low_hz=1, min_band_hz=2)
        else:
            self.filter_bank = SincConv(out_channels=in_channels, kernel_size=filter_kernel_size, sample_rate=sfreq, )

        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(n_channels, first_kernel_size),
                      padding=(0, first_kernel_size // 2), stride=(n_channels, 1)),
            nn.BatchNorm2d(out_channels),
            act_layer(),
        )

        self.layers = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=first_kernel_size, padding=padding),
            norm_layer(out_channels) if norm_layer == nn.BatchNorm1d else norm_layer(
                out_channels // 4, out_channels),
            act_layer(),
            ResidualBlock1D(out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                            n_layers=n_layers, ),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.mlp = Mlp(out_channels,
                       hidden_features=out_channels,
                       out_features=n_classes,
                       act_layer=act_layer,
                       norm_layer=nn.LayerNorm)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.filter_bank(x)
        x = self.first_layer(x)
        x = x.squeeze(-2)
        x = self.layers(x)
        x = self.avg_pool(x)
        features = x.view(x.size(0), -1)  # (B, C, T*F)
        x = self.mlp(features)
        return x, features

    def add_noise(self, x, mean_scaling=0.3, std_scaling=0.7):
        mean = x.mean(dim=-1, keepdim=True) * mean_scaling
        std = x.std(dim=-1, keepdim=True) * std_scaling
        noise = torch.randn_like(mean) * std + mean
        x = x + noise
        return x


if __name__ == "__main__":
    # model = ResNet(128, 512, )
    # model = TimeFFTResNet(128, 256)
    # model = FilterBank(sfreq=256)
    model = FilterBankResNet(
        128, 512
    )
    # model = FeatureResNet(128, 512, n_chs=128, emb_dim=64)
    # model = ResNet2D(1, 128, )
    # model = ResNet2(1, 64, first_kernel_size=15, kernel_size=3, stride=1, downsample=2, n_layers=3, is_residual=True)
    # model = DownSampleResnet(1, 8, n_chs=128, emb_dim=64)
    x = torch.randn((3, 128, 512))
    model.eval()
    z = model(x)
    # random data
    x = torch.randn((3, 32, 128, 32))
    x = x.float()

    # test decoder
    decoder = ConvUpSample2D(32, 16, stride=2, padding=1)
    decoder_2 = ConvUpSample2D(16, 8, stride=2, padding=1)
    decoder_out = decoder(x)
    decoder_out = decoder_2(decoder_out)
    print('Dncoder out shape:', decoder_out.shape)
