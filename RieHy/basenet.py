from einops.layers.torch import Rearrange
import numpy as np
from torch import nn
import torch


class _InputBlock(nn.Module):
    def __init__(
            self,
            n_channels: int = 22,
            n_temporal_filters: int = 40,
            temporal_filter_length: int = 25,
            spatial_expansion: int = 1,
            pool_length: int = 75,
            pool_stride: int = 15,
            dropout: float = 0.5
    ):
        super(_InputBlock, self).__init__()
        self.rearrange_input = Rearrange("b c t -> b 1 c t")
        self.temporal_conv = nn.Conv2d(1, n_temporal_filters,
                                       kernel_size=(1, temporal_filter_length),
                                       padding=(0, temporal_filter_length // 2),
                                       bias=False)
        self.intermediate_bn = nn.BatchNorm2d(n_temporal_filters)
        self.spatial_conv = nn.Conv2d(n_temporal_filters,
                                      n_temporal_filters * spatial_expansion,
                                      kernel_size=(n_channels, 1),
                                      groups=n_temporal_filters, bias=False)
        self.bn = nn.BatchNorm2d(n_temporal_filters * spatial_expansion)
        self.nonlinearity = nn.ELU()
        self.pool = nn.AvgPool2d((1, pool_length), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.rearrange_input(x)
        x = self.temporal_conv(x)
        x = self.intermediate_bn(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.nonlinearity(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class _ChannelAttentionBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 16,
            temp_filter_length: int = 15,
            pool_length: int = 8,
            pool_stride: int = 8,
            dropout: float = 0.5
    ):
        super(_ChannelAttentionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, temp_filter_length),
                      padding=(0, temp_filter_length // 2),
                      bias=False, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, (1, 1), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ELU())

        self.pool = nn.AvgPool2d((1, pool_length), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        out = self.dropout(out)
        return out


class BaseNet(nn.Module):
    def __init__(
            self,
            input_window_samples: int = 256,
            n_channels: int = 128,
            n_temporal_filters: int = 320,
            temp_filter_length_inp: int = 8,
            spatial_expansion: int = 1,
            pool_length_inp: int = 16,
            pool_stride_inp: int = 4,
            dropout_inp: float = 0.25,
            ch_dim: int = 64,
            temp_filter_length: int = 4,
            pool_length: int = 2,
            pool_stride: int = 2,
            dropout: float = 0.25,
            n_classes: int = 4,
            use_feedforward: bool = False,
            return_features: int = 0
    ):
        super(BaseNet, self).__init__()
        self.return_features = return_features
        self.use_feedforward = use_feedforward
        self.input_block = _InputBlock(
            n_channels=n_channels, n_temporal_filters=n_temporal_filters,
            temporal_filter_length=temp_filter_length_inp,
            spatial_expansion=spatial_expansion,
            pool_length=pool_length_inp, pool_stride=pool_stride_inp,
            dropout=dropout_inp)
        self.channel_expansion = nn.Sequential(
            nn.Conv2d(
                n_temporal_filters * spatial_expansion, ch_dim, (1, 1), bias=False),
            nn.BatchNorm2d(ch_dim),
            nn.ELU())

        seq_lengths = self._calculate_sequence_lengths(
            input_window_samples, [temp_filter_length_inp, temp_filter_length],
            [pool_length_inp, pool_length], [pool_stride_inp, pool_stride])

        self.channel_attention_block = _ChannelAttentionBlock(
            in_channels=ch_dim, temp_filter_length=temp_filter_length,
            pool_length=pool_length, pool_stride=pool_stride, dropout=dropout)
        if self.use_feedforward:
            self.feedforward = nn.Sequential(
                nn.Flatten(),
                nn.Linear(seq_lengths[-1] * ch_dim, seq_lengths[-1] * ch_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_lengths[-1] * ch_dim, n_classes))
    
    def forward(self, x):
        x1 = self.input_block(x)
        x2 = self.channel_expansion(x1)
        x3 = self.channel_attention_block(x2)
        if self.use_feedforward:
            x3 = self.feedforward(x3)
        x = self.classifier(x3)
        if self.return_features:
            if self.return_features == 1:
                feature = x3.flatten(1)  # shape: (batch_size, feature_dim)
            else:
                feature = torch.concat([x1.flatten(1), x2.flatten(1), x3.flatten(1)], dim=1) # shape: (batch_size, feature_dim)
            return x, feature
        else:
            return x

    @staticmethod
    def _calculate_sequence_lengths(input_window_samples: int, kernel_lengths: list,
                                    pool_lengths: list, pool_strides: list):
        seq_lengths = []
        out = input_window_samples
        for (k, pl, ps) in zip(kernel_lengths, pool_lengths, pool_strides):
            out = np.floor(out + 2 * (k // 2) - k + 1)
            out = np.floor((out - pl) / ps + 1)
            seq_lengths.append(int(out))
        return seq_lengths
