import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import measure, draw, morphology, filters
from skimage.segmentation import find_boundaries
from sklearn.decomposition import PCA
import json
import os
from datetime import datetime
import copy
from torchvision import models, transforms
from PIL import Image
import torchvision.transforms.functional as TF
from collections import OrderedDict
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: {device}")
# 检查可用的GPU数量
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
# 输入图像的维度
input_channels = 1
input_height = 256
input_width = 256
# Transformer 模型的参数
num_layers = 4
num_heads = 8
hidden_dim = 2048
dropout_rate = 0.1
class TransformerModel(nn.Module):
    def __init__(self, input_channels, input_height, input_width, num_layers, num_heads, hidden_dim, dropout_rate):
        super(TransformerModel, self).__init__()

        # 特征提取层
        resnet50 = models.resnet50(pretrained=False)
        resnet50.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet50.fc = nn.Identity()
        self.feature_extractor = resnet50

        # 计算特征维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            features = self.feature_extractor(dummy_input)
            feature_dim = features.shape[1]

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 参数生成层
        self.parameter_generator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softplus()  # 确保输出参数为正数
        )

    def forward(self, x):
        # 提取特征 [batch_size, feature_dim]
        features = self.feature_extractor(x)

        # 调整维度 [batch_size, 1, feature_dim]
        features = features.unsqueeze(1)

        # Transformer 编码器
        transformer_output = self.transformer_encoder(features)

        # 全局平均池化 [batch_size, feature_dim]
        pooled_output = transformer_output.mean(dim=1)

        # 生成参数 [batch_size, 4]
        parameters = self.parameter_generator(pooled_output)

        return parameters
def DeleteSmall(contours, min_area=200):
    """删除面积过小的轮廓"""
    filtered = []
    for contour in contours:
        # 计算轮廓面积
        poly = np.round(contour).astype(int)
        area = cv2.contourArea(poly[:, ::-1])  # 注意坐标顺序
        if area >= min_area:
            filtered.append(contour)
    return filtered
def guassian_blur(img, kernel_size=3, sigma=1.0):
    """高斯模糊实现"""
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.dim() == 3:
        img = img.unsqueeze(0)

    # 创建高斯核
    x = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
    y = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
    x, y = torch.meshgrid(x, y, indexing='ij')

    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    # 应用卷积
    padding = kernel_size // 2
    blurred = F.conv2d(img, kernel, padding=padding)

    return blurred.squeeze()


def torch_gradient(tensor):
    """更精确的梯度计算"""
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D")

    # 使用中心差分法
    dy = torch.zeros_like(tensor)
    dx = torch.zeros_like(tensor)

    # y方向梯度
    dy[1:-1, :] = (tensor[2:, :] - tensor[:-2, :]) / 2.0
    dy[0, :] = tensor[1, :] - tensor[0, :]  # 前向差分
    dy[-1, :] = tensor[-1, :] - tensor[-2, :]  # 后向差分

    # x方向梯度
    dx[:, 1:-1] = (tensor[:, 2:] - tensor[:, :-2]) / 2.0
    dx[:, 0] = tensor[:, 1] - tensor[:, 0]  # 前向差分
    dx[:, -1] = tensor[:, -1] - tensor[:, -2]  # 后向差分

    return dy, dx


def GLFIF(Img, LImg, u0, sigma, lambda1, lambda2, alpha1, alpha2, g):
    """水平集演化函数"""
    u1 = u0 ** 2
    u2 = (1 - u0) ** 2

    # 计算c1, c2
    Iu1 = Img * u1
    Iu2 = Img * u2
    c1 = torch.sum(Iu1) / (torch.sum(u1) + 1e-8)
    c2 = torch.sum(Iu2) / (torch.sum(u2) + 1e-8)

    # 高斯模糊
    Ku1 = guassian_blur(u1, 3, sigma)
    Ku2 = guassian_blur(u2, 3, sigma)
    KI1 = guassian_blur(Iu1, 3, sigma)
    KI2 = guassian_blur(Iu2, 3, sigma)

    s1 = KI1 / (Ku1 + 1e-8)
    s2 = KI2 / (Ku2 + 1e-8)

    # 计算能量项
    kim = c1 * u1 + c2 * u2
    DcH = (LImg - kim) * LImg
    F3_old = DcH

    sim = s1 * u1 + s2 * u2
    DsH = (LImg - sim) * LImg
    F4_old = DsH

    # 更新水平集函数
    denominator = (lambda2 * (Img - c2) ** 2 + (alpha1 * s1 + alpha2 * c1) + 1e-8)
    un = 1 / (1 + (lambda1 * (Img - c1) ** 2 + (alpha1 * s2 + alpha2 * c2)) / denominator)

    # 平滑处理
    un = guassian_blur(un, 3, sigma)

    return un


def change_lsf(Img, initial_lsf, iter_num, sigma, lambda1, lambda2, alpha1, alpha2):
    """水平集演化主函数"""
    if Img.dim() != 2:
        raise ValueError("Input must be a 2D grayscale image")

    if Img.shape != initial_lsf.shape:
        raise ValueError("Image and initial_lsf must have the same shape")
    # 图像平滑
    img_smooth = guassian_blur(Img, 3, sigma)
    # 计算梯度
    dy, dx = torch_gradient(img_smooth)
    f = dy ** 2 + dx ** 2
    g = 1 / (1 + f)  # 边缘指示函数

    # 水平集演化
    phi = initial_lsf.clone()
    for n in range(iter_num):
        phi = GLFIF(Img, Img, phi, sigma, lambda1, lambda2, alpha1, alpha2, g)

    return phi


def BLS(img, parameters):
    """主水平集分割函数"""
    parameters = parameters.squeeze()

    # 创建初始水平集
    initial_lsf = torch.ones_like(img) * 0.3
    initial_lsf[0:5, 0:5] = 0.7  # 设置小的初始区域

    # 水平集演化
    phi = change_lsf(
        img,
        initial_lsf,
        iter_num=20,
        sigma=0.1,
        lambda1=parameters[0].float(),
        lambda2=parameters[1].float(),
        alpha1=parameters[2].float(),
        alpha2=parameters[3].float()
    )

    return phi
def extract_segments_from_contours(image, contours, phi_np):
    """从轮廓中提取分割区域"""
    segments = []
    original_image = image.copy()

    for i, contour in enumerate(contours):
        try:
            # 创建轮廓掩码
            mask = np.zeros_like(phi_np, dtype=np.uint8)
            contour_int = np.round(contour).astype(int)

            # 确保轮廓点在图像范围内
            contour_int[:, 0] = np.clip(contour_int[:, 0], 0, mask.shape[0] - 1)
            contour_int[:, 1] = np.clip(contour_int[:, 1], 0, mask.shape[1] - 1)

            # 填充轮廓
            cv2.fillPoly(mask, [contour_int[:, ::-1]], 1)  # OpenCV使用(y,x)格式

            # 获取边界框
            y, x = np.where(mask)
            if len(x) > 0 and len(y) > 0:
                x_min, x_max = np.min(x), np.max(x)
                y_min, y_max = np.min(y), np.max(y)

                # 创建分割结果
                segment_image = original_image.copy()
                segment_image[mask == 0] = 0  # 将非区域部分设为黑色

                # 裁剪区域
                cropped_segment = segment_image[y_min:y_max + 1, x_min:x_max + 1]

                segments.append({
                    'id': i + 1,
                    'contour': contour,
                    'mask': mask,
                    'segment': cropped_segment,
                    'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                    'area': int(np.sum(mask))
                })

        except Exception as e:
            print(f"提取第 {i + 1} 个区域时出错: {e}")

    return segments

# 创建模型
net = TransformerModel(input_channels, input_height, input_width, num_layers, num_heads, hidden_dim, dropout_rate).to(
    device)
# 多GPU支持
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    net = nn.DataParallel(net)
else:
    print("Using single GPU or CPU")
# 加载模型权重
try:
    checkpoint_path = "/home/user2/HML/model_epoch_0.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 处理可能的DataParallel包装
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 移除module.前缀（如果使用多GPU训练但单GPU推理）
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict)
        print("Model weights loaded successfully")
    else:
        print("Checkpoint file not found, using random initialization")
except Exception as e:
    print(f"Error loading model weights: {e}")
    print("Using randomly initialized model")
class LevelSetProcessor:
    def __init__(self):
        self.device = device
        self.net = net
        print("LevelSetProcessor initialized with trained model")

    def process_image(self, image_path, visualize=True):
        """处理图像的主函数"""
        return process_single_image_segmentation(image_path, self.net, visualize)

    def save_segmentation_result(self, image_path, output_dir="./results"):
        """保存分割结果"""
        os.makedirs(output_dir, exist_ok=True)
        # 处理图像
        phi_result, contours, segment_count, segments = self.process_image(image_path, visualize=False)
        if phi_result is not None:
            # 创建可视化结果
            image = cv2.imread(image_path)
            image = cv2.resize(image, (input_width, input_height))
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            # 原始图像
            ax1.imshow(img_rgb)
            ax1.set_title('Original Image')
            ax1.axis('off')
            # 分割结果
            ax2.imshow(img_rgb)
            for segment in segments:
                contour = segment['contour']
                ax2.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')
            ax2.set_title(f'Segmentation Result - {segment_count} regions')
            ax2.axis('off')
            # 保存图像
            output_path = os.path.join(output_dir, f"segmentation_{os.path.basename(image_path)}")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"结果已保存到: {output_path}")
            return segment_count
        return 0
def process_single_image_segmentation(image_path, net, visualize=True):
    """处理单张图像的分割"""
    print(f"\n开始处理图像: {os.path.basename(image_path)}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, [], 0, []
    image = cv2.resize(image, (input_width, input_height))
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ======== 图像预处理 - 在 Lab 空间进行 PCA 降维 ========
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    h, w, c = img_lab.shape
    data = img_lab.reshape(-1, c).astype(np.float32)
    pca = PCA(n_components=1)
    reduced = pca.fit_transform(data)
    reduced_img = reduced.reshape(h, w)
    normalized_img = (reduced_img - reduced_img.min()) / (reduced_img.max() - reduced_img.min())
    normalized_img = normalized_img.astype(np.float32)

    # 转换为 PyTorch 张量
    input_tensor = torch.tensor(normalized_img).unsqueeze(0).unsqueeze(0).float().to(device)

    # 模型推理
    with torch.no_grad():
        parameters = net(input_tensor)
        phi = BLS(torch.tensor(normalized_img).to(device), parameters)

    phi_np = phi.cpu().detach().numpy().squeeze()

    # 提取轮廓
    contours = measure.find_contours(phi_np, 0.5)
    contours = DeleteSmall(contours, min_area=image.shape[0] * image.shape[1] * 0.0005)
    segments = extract_segments_from_contours(image, contours, phi_np)
    segment_count = len(segments)

    print(f"图像分割成了 {segment_count} 个部分")

    if visualize:
        fig, (ax1) = plt.subplots(1, figsize=(6, 6))
        ax1.imshow(img_rgb)
        for segment in segments:
            contour = segment['contour']
            ax1.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')
        ax1.set_title('Segmented Regions (Lab-PCA Input)')
        ax1.axis('off')
        plt.tight_layout()
        plt.show()

    return phi_np, contours, segment_count, segments

def main():
    """主函数 - 单张图片处理"""
    print("开始单张岩石图像分割处理...")
    # 初始化处理器
    processor = LevelSetProcessor()
    # 单张图片路径 - 修改为您要处理的图片路径
    image_path = "/home/user2/HML/4image/Pic_1036.jpg"  # 请替换为实际图片路径

    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return

    print(f"📁 处理图像: {os.path.basename(image_path)}")
    try:
        # 处理单张图片并显示结果
        phi_result, contours, segment_count, segments = processor.process_image(image_path, visualize=True)

        if phi_result is not None:
            print(f"✅ 处理完成，共分割出 {segment_count} 个区域")
            # 可选：保存结果
            save_results = input("是否保存结果？(y/n): ").lower().strip()
            if save_results == 'y':
                output_dir = "./results"
                segment_count = processor.save_segmentation_result(image_path, output_dir=output_dir)
                print(f"💾 结果已保存到: {output_dir}")
    except Exception as e:
        print(f"⚠️ 处理图像时出错: {e}")
        import traceback
        traceback.print_exc()
    print("\n🎉 图像处理完成！")
if __name__ == "__main__":
    main()