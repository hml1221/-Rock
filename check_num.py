import os
from torchvision import datasets

data_root = "/home/user2/HML/new-rock"

# 支持的图片格式
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.pgm')

# 创建 ImageFolder
dataset = datasets.ImageFolder(root=data_root)

print("类别列表:", dataset.classes)
print("总样本数:", len(dataset))

# 检查每类数量
for cls_idx, cls_name in enumerate(dataset.classes):
    cls_count = sum(1 for item in dataset.samples if item[1] == cls_idx)
    print(f"{cls_name} 类样本数: {cls_count}")

# 可选：列出没有被识别的文件
for root, dirs, files in os.walk(data_root):
    for f in files:
        if not f.lower().endswith(IMG_EXTENSIONS):
            print("未识别的文件:", os.path.join(root, f))
