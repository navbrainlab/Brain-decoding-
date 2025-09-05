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

    def forward(self, x: Tensor) -> tuple[Union[Tensor, Any], Any, Any, None]:
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

    def forward(self, x: Tensor) -> tuple[Union[Tensor, Any], Any, Any, Tensor]:
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
