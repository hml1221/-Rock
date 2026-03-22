import os
import cv2
import numpy as np

# 根目录
root = "/home/user2/HML/new-rock"

sizes = []

# 遍历目录
for cls in os.listdir(root):
    cls_path = os.path.join(root, cls)
    if not os.path.isdir(cls_path):
        continue

    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print("读取失败:", img_path)
            continue

        h, w = img.shape[:2]
        sizes.append((w, h))

sizes = np.array(sizes)
ws = sizes[:, 0]
hs = sizes[:, 1]

print("\n========= 数据集尺寸统计结果 =========")
print(f"总图像数量: {len(sizes)}")
print(f"最小尺寸: 宽={ws.min()}  高={hs.min()}")
print(f"最大尺寸: 宽={ws.max()}  高={hs.max()}")
print(f"平均尺寸: 宽={ws.mean():.1f}  高={hs.mean():.1f}")
print(f"中位数尺寸: 宽={np.median(ws)}  高={np.median(hs)}")

# 自动推荐输入尺寸
median_w = int(np.median(ws))
median_h = int(np.median(hs))

# 取两者平均作为参考
ref_size = int((median_w + median_h) / 2)

# 自动推荐一个更标准的深度学习尺寸
# 选择接近 ref_size 的常用值
candidate_sizes = [64, 96, 128, 160, 192, 224]
best = min(candidate_sizes, key=lambda x: abs(x - ref_size))

print("\n========= 自动推荐输入尺寸 =========")
print(f"建议模型输入尺寸：{best} × {best}")
print("（基于中位数尺寸自动计算）")
