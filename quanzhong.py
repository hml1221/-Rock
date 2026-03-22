import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


# =========================
# 1) 基本设置（按需改）
# =========================
data_root = "/home/user2/HML/new-rock"
batch_size = 32
num_epochs = 100
lr = 1e-3
input_size = 64

# ✅ 最优权重保存路径（建议绝对路径）
BEST_MODEL_PATH = "/home/user2/HML/shibie/best_rock_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# 2) 数据预处理（尽量保持与你原训练一致）
# =========================
train_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    # 你原来写的是 [0.5]，[0.5]；RGB更规范是3通道：
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

val_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# =========================
# 3) 构建分层划分的数据集与Loader
# =========================
# 先读取不带transform的dataset来拿targets
full_dataset_raw = datasets.ImageFolder(root=data_root)
targets = [full_dataset_raw[i][1] for i in range(len(full_dataset_raw))]

train_indices, val_indices = train_test_split(
    np.arange(len(full_dataset_raw)),
    test_size=0.3,
    shuffle=True,
    stratify=targets,
    random_state=42
)

# 再分别构建带不同transform的dataset
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

# 类别名（用于写进checkpoint，保证推理时class_id不会错位）
class_names = full_dataset_raw.classes
num_classes = len(class_names)

print("训练集:", len(train_dataset), "验证集:", len(val_dataset))
print("类别:", class_names)


# =========================
# 4) 模型/损失/优化器
# =========================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# =========================
# 5) 验证函数
# =========================
@torch.no_grad()
def evaluate_model():
    model.eval()
    correct = 0
    total = 0

    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    return correct / max(total, 1)


# =========================
# 6) 训练 + 保存最优权重
# =========================
def train_and_save_best():
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)

    best_acc = -1.0
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.numel()

        scheduler.step()

        train_loss = running_loss / max(len(train_loader.dataset), 1)
        train_acc = running_correct / max(running_total, 1)
        val_acc = evaluate_model()

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # ✅ 保存最优
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

            checkpoint = {
                "epoch": best_epoch,
                "val_acc": float(best_acc),
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "input_size": input_size,
            }
            torch.save(checkpoint, BEST_MODEL_PATH)
            print(f"✅ Saved best checkpoint -> {BEST_MODEL_PATH} (epoch={best_epoch}, val_acc={best_acc:.4f})")

    print(f"\n🎯 Done. Best epoch={best_epoch}, best val_acc={best_acc:.4f}")
    print(f"Best weight file: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_best()
