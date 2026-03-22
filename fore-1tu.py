import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
import matplotlib.pyplot as plt
from skimage import measure
from collections import OrderedDict
from torchvision import models
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# 图像输入尺寸
input_channels = 1
input_height = 256
input_width = 256
# Transformer 模型参数
num_layers = 4
num_heads = 8
hidden_dim = 2048
dropout_rate = 0.1


class TransformerModel(nn.Module):
    def __init__(self, input_channels, input_height, input_width, num_layers, num_heads, hidden_dim, dropout_rate):
        super().__init__()
        resnet50 = models.resnet50(pretrained=False)
        resnet50.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)
        resnet50.fc = nn.Identity()
        self.feature_extractor = resnet50

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            feat = self.feature_extractor(dummy)
            feature_dim = feat.shape[1]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout_rate, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.parameter_generator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softplus()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.unsqueeze(1)
        transformer_output = self.transformer_encoder(features)
        pooled = transformer_output.mean(dim=1)
        params = self.parameter_generator(pooled)
        return params


def DeleteSmall(contours, min_area=200):
    filtered = []
    for contour in contours:
        poly = np.round(contour).astype(int)
        area = cv2.contourArea(poly[:, ::-1])
        if area >= min_area:
            filtered.append(contour)
    return filtered


def guassian_blur(img, kernel_size=3, sigma=1.0):
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.dim() == 3:
        img = img.unsqueeze(0)
    x = torch.arange(kernel_size, device=img.device) - kernel_size // 2
    y = torch.arange(kernel_size, device=img.device) - kernel_size // 2
    x, y = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    blurred = F.conv2d(img, kernel, padding=padding)
    return blurred.squeeze()
def torch_gradient(tensor):
    dy = torch.zeros_like(tensor)
    dx = torch.zeros_like(tensor)
    dy[1:-1, :] = (tensor[2:, :] - tensor[:-2, :]) / 2.0
    dy[0, :] = tensor[1, :] - tensor[0, :]
    dy[-1, :] = tensor[-1, :] - tensor[-2, :]
    dx[:, 1:-1] = (tensor[:, 2:] - tensor[:, :-2]) / 2.0
    dx[:, 0] = tensor[:, 1] - tensor[:, 0]
    dx[:, -1] = tensor[:, -1] - tensor[:, -2]
    return dy, dx


def GLFIF(Img, LImg, u0, sigma, lambda1, lambda2, alpha1, alpha2, g):
    u1 = u0 ** 2
    u2 = (1 - u0) ** 2
    Iu1 = Img * u1
    Iu2 = Img * u2
    c1 = torch.sum(Iu1) / (torch.sum(u1) + 1e-8)
    c2 = torch.sum(Iu2) / (torch.sum(u2) + 1e-8)
    Ku1 = guassian_blur(u1, 3, sigma)
    Ku2 = guassian_blur(u2, 3, sigma)
    KI1 = guassian_blur(Iu1, 3, sigma)
    KI2 = guassian_blur(Iu2, 3, sigma)
    s1 = KI1 / (Ku1 + 1e-8)
    s2 = KI2 / (Ku2 + 1e-8)
    kim = c1 * u1 + c2 * u2
    sim = s1 * u1 + s2 * u2
    denominator = (lambda2 * (Img - c2) ** 2 + (alpha1 * s1 + alpha2 * c1) + 1e-8)
    un = 1 / (1 + (lambda1 * (Img - c1) ** 2 + (alpha1 * s2 + alpha2 * c2)) / denominator)
    un = guassian_blur(un, 3, sigma)
    return un


def change_lsf(Img, initial_lsf, iter_num, sigma, lambda1, lambda2, alpha1, alpha2):
    img_smooth = guassian_blur(Img, 3, sigma)
    dy, dx = torch_gradient(img_smooth)
    f = dy ** 2 + dx ** 2
    g = 1 / (1 + f)
    phi = initial_lsf.clone()
    for _ in range(iter_num):
        phi = GLFIF(Img, Img, phi, sigma, lambda1, lambda2, alpha1, alpha2, g)
    return phi


def extract_segments_from_contours(image, contours, phi_np):
    segments = []
    for i, contour in enumerate(contours):
        try:
            mask = np.zeros_like(phi_np, dtype=np.uint8)
            contour_int = np.round(contour).astype(int)
            contour_int[:, 0] = np.clip(contour_int[:, 0], 0, mask.shape[0] - 1)
            contour_int[:, 1] = np.clip(contour_int[:, 1], 0, mask.shape[1] - 1)
            cv2.fillPoly(mask, [contour_int[:, ::-1]], 1)
            y, x = np.where(mask)
            if len(x) > 0 and len(y) > 0:
                x_min, x_max = np.min(x), np.max(x)
                y_min, y_max = np.min(y), np.max(y)
                segments.append({
                    'id': i + 1,
                    'contour': contour,
                    'mask': mask,
                    'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                    'area': int(np.sum(mask))
                })
        except:
            continue
    return segments


def extract_background_mask(segments, image_shape, phi_np=None, image_gray=None):
    """
    提取背景区域：第一次分割后没有被分割到的区域

    Args:
        segments: 前景区域列表（第一次分割的结果）
        image_shape: 图像形状
        phi_np: 水平集函数值（可选，用于辅助清理）
        image_gray: 灰度图像（可选，用于辅助清理）
    """
    h, w = image_shape[:2]

    # 合并所有前景区域的掩码
    foreground_mask = np.zeros((h, w), dtype=np.uint8)
    for seg in segments:
        foreground_mask = cv2.bitwise_or(foreground_mask, seg['mask'])

    # 背景 = 不在前景中的区域（简单取反）
    background_mask = 1 - foreground_mask

    # 形态学操作清理背景掩码
    # 填充前景区域内部的小孔洞（这些孔洞应该也是背景的一部分）
    kernel_size = max(3, min(h, w) // 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 闭运算：填充背景区域的小孔洞
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 确保前景区域不是背景（双重检查）
    background_mask[foreground_mask > 0] = 0

    return background_mask.astype(np.uint8)
def process_single_image_segmentation(image_path, net, visualize=True):
    print(f"\n开始处理图像: {os.path.basename(image_path)}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, [], 0, [], [], None

    # 调整图像尺寸
    image = cv2.resize(image, (input_width, input_height))
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_img = img_gray.astype(np.float32) / 255.0
    input_tensor = torch.tensor(normalized_img).unsqueeze(0).unsqueeze(0).float().to(device)

    # 模型推理
    net.eval()
    with torch.no_grad():
        parameters = net(input_tensor)
    parameters = parameters.squeeze()
    if parameters.dim() == 0: parameters = parameters.unsqueeze(0)
    if parameters.numel() != 4:
        raise ValueError(f"Expected 4 parameters, got {parameters.numel()}")
    print(f"Predicted parameters: {parameters.cpu().numpy()}")

    # 自适应 sigma
    sigma = max(np.std(normalized_img) * 2, 0.1)
    print(f"Adaptive sigma: {sigma:.3f}")

    # 初始化一次分割水平集
    h, w = normalized_img.shape
    initial_lsf = torch.ones_like(torch.tensor(normalized_img)).to(device) * 0.3
    initial_lsf[h // 4:h * 3 // 4, w // 4:w * 3 // 4] = 0.7

    # 一次分割
    phi = change_lsf(
        torch.tensor(normalized_img).to(device),
        initial_lsf,
        iter_num=20,
        sigma=sigma,
        lambda1=parameters[0].float(),
        lambda2=parameters[1].float(),
        alpha1=parameters[2].float(),
        alpha2=parameters[3].float()
    )
    phi_np = phi.cpu().detach().numpy()
    contours = measure.find_contours(phi_np, 0.5)
    contours = DeleteSmall(contours, min_area=image.shape[0] * image.shape[1] * 0.0005)
    segments = extract_segments_from_contours(image, contours, phi_np)
    segment_count = len(segments)
    print(f"第一次分割: {segment_count} 个区域")

    # ----------------------------
    # 前景二次分割（改为灰度图像）
    # ----------------------------
    # 二次分割（方案 2：使用第一次分割结果作为初始轮廓）
    second_segmentation_results = []

    if segment_count > 0:

        # foreground_mask：一次分割所有区域合并
        foreground_mask = np.zeros((h, w), dtype=np.uint8)
        for seg in segments:
            foreground_mask = cv2.bitwise_or(foreground_mask, seg['mask'])

        foreground_area = np.sum(foreground_mask)

        if foreground_area > 500:

            # ====== (1) 使用灰度图（你要求的）======
            fg_gray = img_gray.astype(np.float32) / 255.0
            fg_gray[foreground_mask == 0] = 0  # 非前景置 0
            fg_tensor = torch.tensor(fg_gray).to(device)

            # ====== (2) 构造二次分割的初始化 ======
            # --- 核心！直接使用第一次 phi 的区域作为二次分割初始轮廓 ---
            initial_lsf_sec = torch.tensor(phi_np, device=device)

            # 限制二次分割只在前景内部演化
            mask_torch = torch.tensor(foreground_mask, device=device)
            initial_lsf_sec[mask_torch == 0] = 0.3  # 背景降权
            initial_lsf_sec[mask_torch == 1] = 0.7  # 前景增强

            # ====== (3) 二次分割使用完全相同参数 ======
            lambda1 = parameters[0].float()
            lambda2 = parameters[1].float()
            alpha1 = parameters[2].float()
            alpha2 = parameters[3].float()

            # ====== (4) 执行二次水平集分割 ======
            phi_sec = change_lsf(
                fg_tensor,
                initial_lsf_sec,
                iter_num=20,  # 二次分割不要太多，否则破坏形状
                sigma=sigma,  # 与第一次完全一致
                lambda1=lambda1,
                lambda2=lambda2,
                alpha1=alpha1,
                alpha2=alpha2
            )

            phi_sec_np = phi_sec.cpu().detach().numpy()

            # ====== (5) 找二次分割轮廓 ======
            contours_sec = measure.find_contours(phi_sec_np, level=0.5)
            min_area = image.shape[0] * image.shape[1] * 0.00001
            filtered_contours_sec = DeleteSmall(contours_sec, min_area=min_area)

            if filtered_contours_sec:
                second_segmentation_results = [{
                    'contours': filtered_contours_sec,
                    'second_phi': phi_sec_np,
                    'num_sub_segments': len(filtered_contours_sec)
                }]
                print(f"二次分割得到 {len(filtered_contours_sec)} 个子区域")
            else:
                print("二次分割没有得到子区域")

    if visualize:
        # 保持和原代码一致的大小
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)

        # 绘制一次分割轮廓（第一级）
        for seg in segments:
            plt.plot(seg['contour'][:, 1], seg['contour'][:, 0], color='red', linewidth=2.0)

        # 绘制二次分割轮廓（第二级）
        for result in second_segmentation_results:
            for contour in result['contours']:
                plt.plot(contour[:, 1], contour[:, 0], color='yellow', linewidth=1.5)

        plt.axis('off')
        plt.tight_layout(pad=0)  # 去掉白边
        plt.show()

    return phi_np, contours, segment_count, segments, second_segmentation_results, foreground_mask

# 主函数
def main_single_image():
    print("开始单张岩石图像鲁棒分割处理...")
    image_path = "/home/user2/HML/4image/Pic_1083.jpg"

    if not os.path.exists(image_path):
        print(f"❌ 图像不存在: {image_path}")
        return
    try:
        phi_result, contours, segment_count, segments, second_results, background_mask = process_single_image_segmentation(
            image_path, net, visualize=True
        )
        total_secondary = sum([res['num_sub_segments'] for res in second_results]) if second_results else 0
        print(f"✅ 处理完成:")
        print(f"   第一次分割: {segment_count} 个区域")
        print(f"   背景二次分割: {total_secondary} 个子区域")
        print(f"   背景像素总数: {np.sum(background_mask)}")
    except Exception as e:
        print(f"⚠️ 处理图像时出错: {e}")
        import traceback;
        traceback.print_exc()
if __name__ == "__main__":
    net = TransformerModel(input_channels, input_height, input_width, num_layers, num_heads, hidden_dim,
                           dropout_rate).to(device)
    checkpoint_path = "/home/user2/HML/model_epoch_0.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_dict = OrderedDict()
        for k, v in state_dict.items(): new_dict[k.replace('module.', '')] = v
        model_dict = net.state_dict()
        for k, v in new_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape: model_dict[k] = v
        net.load_state_dict(model_dict)
        print("模型权重加载成功")
    main_single_image()
