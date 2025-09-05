import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import pickle
from motion_decoding import ResNet1D_most_mini
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from utils import *
import os

def load_model(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    return model


def infer(model, test_loader, criterion, device):
    model.eval() 
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, label = batch
            x = x.to(device)
            label = label.to(device)

            output = model(x)
            loss = criterion(output, label)
            total_loss += loss.item() * x.size(0)

            all_preds.append(output.argmax(dim=1).cpu()) 
            all_labels.append(label.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(test_loader.dataset), acc, all_preds, all_labels

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

# 主推理函数
def run_inference(model, test_loader, criterion, checkpoint_path, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model = load_model(model, checkpoint_path, device)
    test_loss, test_acc, test_preds, test_labels = infer(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # 保存混淆矩阵
    save_path = Path(save_dir) / 'confusion_matrix.png'
    plot_confusion_matrix(test_labels, test_preds, save_path)
    
    # 保存预测结果为 JSON
    json_path = Path(save_dir) / 'test_results.json'
    save_prediction_results(test_labels, test_preds, test_acc, json_path)

def save_prediction_results(y_true, y_pred, acc, save_path):
    results = {
        "accuracy": round(acc, 4),
        "samples": []
    }

    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        results["samples"].append({
            "index": i,
            "true_label": int(true),
            "pred_label": int(pred)
        })

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

# 示例用法
if __name__ == "__main__":

    # === 基础参数 ===
    batch_size = 256
    dropout = 0.6
    lr = 3e-4
    weight_decay = 1e-4
    feature_dim = 126
    num_classes = 4
    num_epochs = 200
    window_size = 600
    step = 60

    # === 加载测试集 ===
    root = 'E:\\motion_decoding\\daily_bdy'
    session_test = ['20250401']

    test_neu, test_label = read_data_motion_session(root=root, session_used=session_test)
    test_neu, test_label = slide_window(test_neu, test_label, windows_size=window_size, step=step, max_trial_length=1200, trial_handle_method='remove')


    # chs_select = [27, 5, 63, 94, 61, 73, 29, 125, 78, 93, 6, 32, 3, 31, 95, 22, 109, 71, 88, 1, 25, 47, 
    #  119, 99, 54, 7, 55, 92, 0, 85, 96, 23, 50, 8, 122, 42, 53, 117, 30, 51, 106, 64, 34, 
    #  18, 70, 39, 69, 33, 20, 10, 102, 98, 123, 24, 124, 13, 72, 114, 17, 38, 56, 26, 101, 
    #  28, 81, 97, 65, 52, 12, 43, 15, 110, 48, 68, 90, 120, 105, 21, 35, 14]
    chs_select = [27, 64, 79, 62, 74, 5, 29, 96, 95, 32, 1, 31, 72, 90, 121, 127, 22, 54, 111, 97, 25, 3, 61, 
    6, 47, 55, 98, 101, 51, 0, 94, 23, 50, 18, 108, 119, 53, 8, 86, 126, 33, 42, 65, 24, 70, 34, 71, 15, 73, 
    30, 39, 20, 10, 100, 104, 116, 56, 124, 103, 28, 17, 38, 52, 105, 125, 66, 7, 13, 26, 46, 112, 43, 117, 
    12, 14, 48, 92, 122, 77, 19]

    feature_dim = len(chs_select)
    test_neu = test_neu[:, chs_select, :]

    # === 加载模型 ===
    model = ResNet1D_most_mini(
        input_channels=feature_dim,
        num_classes=num_classes,
        num_blocks=[1],
        dropout_rate=dropout,
        max_epochs=num_epochs,
        lr=lr
    )
    model = model.to(torch.float32)
    test_dataset = TensorDataset(
        torch.tensor(test_neu, dtype=torch.float32),
        torch.tensor(test_label, dtype=torch.long),
    )

    trainer = pl.Trainer(devices=1)

    ckpt_path = 'resnet_most_mini/2c32rjdxj1a8un6b7cc42/checkpoints/epoch=199-step=55000.ckpt'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    # test_loss, test_acc, test_preds, test_labels = infer(model, DataLoader(test_dataset), criterion, device)
    
    save_dir = 'resnet_most_mini_Adam_riemann_80'
    run_inference(model, DataLoader(test_dataset), criterion, ckpt_path, device, save_dir)