import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import glob
import numpy as np
import math
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import *
import os
from swanlab.integration.pytorch_lightning import SwanLabLogger
from joblib import delayed, Parallel
from einops import rearrange


class SEModel(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        y = x * y.expand_as(x)
        return y


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv3 = nn.Conv1d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv15 = nn.Conv1d(
            in_channels,
            out_channels // 4,
            kernel_size=15,
            stride=stride,
            padding=7,
            bias=False,
        )
        self.conv31 = nn.Conv1d(
            in_channels,
            out_channels // 4,
            kernel_size=31,
            stride=stride,
            padding=15,
            bias=False,
        )

    def forward(self, x):
        return torch.cat([self.conv3(x), self.conv15(x), self.conv31(x)], dim=1)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        downsample=False,
        dropout_rate=0.3,
    ):
        super(BasicBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = MultiScaleConv(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.SiLU()  # nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEModel(channel=out_channels)

        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet1D_most_mini(pl.LightningModule):
    def __init__(
        self,
        input_channels=128,
        num_classes=10,
        num_blocks=[1],
        dropout_rate=0.3,
        contrastive_temperature=0.1,
        max_epochs=100,
        lr=1e-4,
    ):
        super(ResNet1D_most_mini, self).__init__()

        self.conv1 = nn.Conv1d(
            input_channels, 128, kernel_size=32, padding="same", bias=False
        )  # kernel_size=3
        self.conv1_2 = nn.Conv1d(
            input_channels, 128, kernel_size=64, padding="same", bias=False
        )
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.SiLU()  # nn.ReLU(inplace=True)
        self.weight_schedule = "fixed"
        self.final_alpha = 0.5  # 0.05
        self.initial_alpha = 0.5
        self.current_alpha = self.initial_alpha
        self.max_epochs = max_epochs
        self.lr = lr

        self.layer1 = self._make_layer(
            256, 512, num_blocks[0], stride=2, dropout_rate=dropout_rate
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128)
        )
        self.fnn = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )
        self.fc = nn.Linear(512, num_classes)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        layers = [
            BasicBlock(
                in_channels,
                out_channels,
                downsample=True,
                stride=stride,
                dropout_rate=dropout_rate,
            )
        ]
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(out_channels, out_channels, dropout_rate=dropout_rate)
            )
        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).to(x.dtype)  # [B, 1, T_orig]

        x0 = self.conv1(x)
        x1 = self.conv1_2(x)
        x = torch.concatenate((x0, x1), axis=1)
        x = self.bn1(x)
        x = self.relu(x)

        feature = self.layer1(x)

        x = self.avgpool(feature)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fnn(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        predict = torch.argmax(logits, axis=1)
        acc = self.acc(predict, y)
        self.log("train loss", loss)
        self.log("train acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        predict = torch.argmax(logits, axis=1)
        acc = self.acc(predict, y)
        self.log("val loss", loss)
        self.log("val acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        predict = torch.argmax(logits, axis=1)
        acc = self.acc(predict, y)
        self.log("test loss", loss)
        self.log("test acc", acc)

        self.y_pred = predict
        self.y_true = y
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


class DataModule(pl.LightningDataModule):
    def __init__(
        self, train_dataset, test_dataset, val_dataset, batch_size=32, num_workers=4
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.bacth_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.bacth_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.bacth_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.bacth_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    batch_size = 256
    dropout = 0.6
    lr = 3e-4
    weight_decay = 1e-4
    channel_num = 80
    session_train = [
        # "20250323",
        # "20250324",
        "20250325",
        "20250326",
        "20250327",
        "20250329",
        "20250331",
    ]
    session_test = ["20250401"]

    feature_dim = 126
    num_classes = 4
    num_epochs = 200
    window_size = 600
    step = 60

    root = "/media/ubuntu/Storage/ecog_data/daily_bdy_new"
    all_neu, all_label = read_data_motion_session(root=root, session_used=session_train)
    x_train, x_test, y_train, y_test = train_test_split(
        all_neu, all_label, test_size=0.2, random_state=42, stratify=all_label
    )
    x_train, y_train = slide_window(
        x_train,
        y_train,
        windows_size=window_size,
        step=step,
        max_trial_length=1200,
        trial_handle_method="remove",
    )
    x_test, y_test = slide_window(
        x_test,
        y_test,
        windows_size=window_size,
        step=step,
        max_trial_length=1200,
        trial_handle_method="remove",
    )
    test_neu, test_label = read_data_motion_session(
        root=root, session_used=session_test
    )
    test_neu, test_label = slide_window(
        test_neu,
        test_label,
        windows_size=window_size,
        step=step,
        max_trial_length=1200,
        trial_handle_method="remove",
    )

    # for riemann channel select
    chs_select = channel_contribution(
        neu=x_train, label=y_train, channel_num=channel_num
    )
    feature_dim = len(chs_select)
    print("riemann selected chans:", chs_select)
    x_train = x_train[:, chs_select, :]
    x_test = x_test[:, chs_select, :]
    test_neu = test_neu[:, chs_select, :]

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(test_neu, dtype=torch.float32),
        torch.tensor(test_label, dtype=torch.long),
    )
    data_module = DataModule(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
    )

    model = ResNet1D_most_mini(
        input_channels=feature_dim,
        num_classes=num_classes,
        num_blocks=[1],
        dropout_rate=dropout,
        max_epochs=num_epochs,
        lr=lr,
    )

    model = model.to(torch.float32)
    logger = SwanLabLogger(
        project="resnet_most_mini",
        experiment_name="resnet_most_mini_200",
        config={
            "architecture": "CNN-ResNet(32)(64)-fnn",
            "lr": str(lr),
            "weight_decay": str(weight_decay),
            "dropout": str(dropout),
            "batch_size": str(batch_size),
            "session_train": str(session_train),
            "session_test": str(session_test),
            "selected_chs": str(chs_select),
        },
    )

    output_path = "./outputs"
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=1,
        default_root_dir=output_path,
        logger=logger,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint(f"{output_path}/last_model.ckpt")
    trainer.test(model=model, dataloaders=DataLoader(test_dataset), ckpt_path="last")
