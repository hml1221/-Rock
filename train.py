import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
import seaborn as sns



# --------------------------------------
# 1. 基本设置
# --------------------------------------
data_root =  "/home/user2/HML/new-rock"
batch_size = 32
num_epochs = 100
lr = 1e-3
input_size = 64  # 推荐尺寸

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------
# 2. 数据增强 & 预处理
# --------------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),

    # 数据增强
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),

    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
# --------------------------------------
# 3. 加载数据集（修正版，训练和验证分开 transform）
# --------------------------------------

# 1) 加载不带 transform 的原始数据
full_dataset = datasets.ImageFolder(root=data_root)

# 2) 随机划分索引（完全等价 random_split）
dataset_size = len(full_dataset)
from sklearn.model_selection import train_test_split

# 1) 获取所有样本的标签（ImageFolder 每张图都可读取 label）
targets = [full_dataset[i][1] for i in range(len(full_dataset))]

# 2) 分层划分
train_indices, val_indices = train_test_split(
    np.arange(len(full_dataset)),
    test_size=0.3,
    shuffle=True,
    stratify=targets,   # 核心：分层抽样
    random_state=42
)

print("分层采样完成：")
print("训练集:", len(train_indices), "验证集:", len(val_indices))


# 3) 创建带 transform 的训练集与验证集
train_dataset = torch.utils.data.Subset(
    datasets.ImageFolder(root=data_root, transform=train_transforms),
    train_indices
)

val_dataset = torch.utils.data.Subset(
    datasets.ImageFolder(root=data_root, transform=val_transforms),
    val_indices
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

print("训练集:", len(train_dataset), "验证集:", len(val_dataset))
print("类别:", full_dataset.classes)

# --------------------------------------
# 4. 创建模型（ResNet18）
# --------------------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 6)  # 五分类
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# --------------------------------------
# 5. 训练函数


# --------------------------------------
# 6. 验证函数
# --------------------------------------
def evaluate_model():
    model.eval()
    correct = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)

    return correct.double() / len(val_loader.dataset)


# --------------------------------------
# 7. 混淆矩阵 & 分类报告
# --------------------------------------


    # --------------------------------------
    # 5. 训练函数（增加准确率记录）
# --------------------------------------
# 训练函数（记录准确率并绘图）
# --------------------------------------
def train_model():
    best_acc = 0
    train_acc_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels)

        scheduler.step()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct.double() / len(train_loader.dataset)
        val_acc = evaluate_model()

        train_acc_list.append(train_acc.item())
        val_acc_list.append(val_acc.item())

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_rock_model.pth")

    print("训练完成，最优验证准确率:", best_acc)

    # 绘制准确率曲线并保存
    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), train_acc_list, label="Train Acc")
    plt.plot(range(1, num_epochs+1), val_acc_list, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("")
    plt.legend()
    plt.grid(True)
    plt.ylim(0.6, 1.05)
    plt.show()  # 可选，在有 GUI 的环境下显示

def test_report():
    class_names = full_dataset.classes

    model.load_state_dict(torch.load("best_rock_model.pth"))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n=== 分类报告 ===")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    print("混淆矩阵：")
    print(cm)
    # -----------------------------
    # 混淆矩阵热力图
    # -----------------------------
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix Heatmap")
    plt.tight_layout()
    plt.show()

# 主程序入口
# --------------------------------------
if __name__ == "__main__":
    train_model()
    test_report()
