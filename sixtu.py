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


def extract_foreground_mask(segments, image_shape):
    """提取前景掩码"""
    foreground_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for seg in segments:
        foreground_mask = cv2.bitwise_or(foreground_mask, seg['mask'])
    return foreground_mask


def extract_background_mask(segments, image_shape):
    foreground_mask = extract_foreground_mask(segments, image_shape)
    return 1 - foreground_mask


def preprocess_for_otsu(roi_b):
    """
    为OTSU算法预处理图像
    """
    # 确保数据类型正确
    if roi_b.dtype != np.uint8:
        if roi_b.max() <= 1.0:
            roi_b = (roi_b * 255).astype(np.uint8)
        else:
            roi_b = roi_b.astype(np.uint8)

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(roi_b, (5, 5), 0)

    # 对比度增强（OTSU对对比度敏感）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    return enhanced


def postprocess_binary(binary):
    """
    对OTSU二值化结果进行后处理
    """
    # 形态学开运算去除小噪声
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 闭运算填充小孔洞
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def extract_contours_from_binary(binary, min_area_threshold):
    """
    从二值图像中提取轮廓
    """
    contours = []

    # 查找轮廓
    contour_result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contour_result) == 2:
        contour_candidates = contour_result[0]
    else:
        contour_candidates = contour_result[1]

    for contour in contour_candidates:
        if len(contour) >= 3:  # 确保是多边形
            # 计算面积
            area = cv2.contourArea(contour)

            # 面积过滤
            if area >= min_area_threshold:
                # 转换轮廓格式（从OpenCV格式到skimage格式）
                contour_skimage = contour.reshape(-1, 2)
                contour_skimage = contour_skimage[:, ::-1]  # 转换坐标 (x,y) -> (y,x)
                contours.append(contour_skimage)

    return contours


def perform_single_region_otsu(roi_b, parent_mask_roi, parent_segment, offset):
    """
    单个区域的OTSU分割（用于回退）
    """
    segments = []
    try:
        # 预处理B通道图像
        roi_processed = preprocess_for_otsu(roi_b)

        # 应用OTSU全局阈值
        _, binary = cv2.threshold(roi_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 后处理：形态学操作去除噪声
        cleaned_binary = postprocess_binary(binary)

        # 确保只在父区域内处理
        cleaned_binary[parent_mask_roi == 0] = 0

        # 动态计算最小面积阈值
        min_area_second = max(20, parent_segment['area'] * 0.01)

        # 提取轮廓
        child_contours = extract_contours_from_binary(cleaned_binary, min_area_second)

        # 创建子区域
        for j, contour in enumerate(child_contours):
            child_segment = create_child_segment(
                contour, parent_segment, offset, j, roi_b.shape
            )
            if child_segment:
                segments.append(child_segment)

    except Exception as e:
        print(f"OTSU回退分割失败: {e}")

    return segments


def levelset_with_heatmap_initialization(roi_b, parent_mask_roi, phi_parent):
    """
    使用父区域热力图作为第二次水平集的初始LSF
    """
    try:
        # 预处理B通道
        if np.sum(roi_b) == 0:
            return None

        normalized_b = (roi_b.astype(np.float32) - roi_b.min()) / (roi_b.max() - roi_b.min() + 1e-8)
        img_tensor = torch.tensor(normalized_b).to(device)

        # 使用父区域热力图作为初始LSF（在父区域内）
        initial_lsf = torch.ones_like(img_tensor) * 0.3
        foreground_indices = torch.tensor(parent_mask_roi, device=device) > 0

        # 将父区域热力图映射到初始LSF
        phi_parent_tensor = torch.tensor(phi_parent, device=device)
        initial_lsf[foreground_indices] = phi_parent_tensor[foreground_indices]

        # 为子区域调整参数（更敏感）
        lambda1 = lambda2 = 0.8  # 更小的lambda，更高灵敏度
        alpha1 = alpha2 = 0.1  # 更小的alpha，更精细分割

        # 基于父区域热力图方差进一步调整参数
        phi_values = phi_parent[parent_mask_roi == 1]
        if len(phi_values) > 0:
            phi_std = np.std(phi_values)
            # 热力图变化大时使用更敏感的参数
            if phi_std > 0.2:
                lambda1 = lambda2 = 0.6
                alpha1 = alpha2 = 0.05

        parameters_child = torch.tensor([lambda1, lambda2, alpha1, alpha2], device=device)

        # 执行水平集分割
        phi_child = change_lsf(
            img_tensor,
            initial_lsf,
            iter_num=25,  # 减少迭代次数
            sigma=0.8,  # 稍小的sigma保持细节
            lambda1=parameters_child[0].float(),
            lambda2=parameters_child[1].float(),
            alpha1=parameters_child[2].float(),
            alpha2=parameters_child[3].float()
        )

        phi_child_np = phi_child.cpu().detach().numpy()
        return phi_child_np

    except Exception as e:
        print(f"热力图初始化水平集失败: {e}")
        return None


def perform_foreground_heatmap_segmentation(segments, original_image, phi_np):
    """
    对前景区域进行热力图二次分割（使用水平集方法）
    """
    second_level_segments = []

    for i, parent_segment in enumerate(segments):
        # 只在足够大的区域上进行二次分割
        if parent_segment['area'] < 500:
            continue

        try:
            # 提取父区域ROI
            bbox = parent_segment['bbox']
            x_min, y_min, x_max, y_max = bbox
            parent_mask_roi = parent_segment['mask'][y_min:y_max + 1, x_min:x_max + 1]

            # 提取对应的热力图区域
            phi_roi = phi_np[y_min:y_max + 1, x_min:x_max + 1]

            # 创建只包含父区域的B通道图像
            roi_b = original_image[y_min:y_max + 1, x_min:x_max + 1, 0].copy()
            roi_b[parent_mask_roi == 0] = 0

            print(
                f"🔍 处理父区域 {parent_segment['id']}, ROI尺寸: {roi_b.shape}, 热力图均值: {np.mean(phi_roi[parent_mask_roi == 1]):.3f}")

            # 使用热力图指导的水平集分割
            phi_child_np = levelset_with_heatmap_initialization(roi_b, parent_mask_roi, phi_roi)

            if phi_child_np is not None:
                # 从子水平集提取轮廓
                child_contours = measure.find_contours(phi_child_np, 0.5)

                # 动态计算最小面积阈值
                min_area_second = max(20, parent_segment['area'] * 0.01)
                filtered_contours = DeleteSmall(child_contours, min_area_second)

                # 创建子区域
                for j, contour in enumerate(filtered_contours):
                    child_segment = create_child_segment(
                        contour, parent_segment, (x_min, y_min), j, roi_b.shape
                    )
                    if child_segment:
                        second_level_segments.append(child_segment)

                print(f"✅ 父区域 {parent_segment['id']} 热力图分割出 {len(filtered_contours)} 个子区域")
            else:
                print(f"⚠️ 父区域 {parent_segment['id']} 热力图分割失败，使用OTSU回退")
                # 回退到OTSU方法
                otsu_segments = perform_single_region_otsu(roi_b, parent_mask_roi, parent_segment, (x_min, y_min))
                second_level_segments.extend(otsu_segments)

        except Exception as e:
            print(f"❌ 父区域 {parent_segment['id']} 热力图分割失败: {e}")
            continue

    return second_level_segments


def create_child_segment(contour, parent_segment, offset, child_index, roi_shape):
    """
    从分割创建子区域并转换坐标
    """
    x_offset, y_offset = offset

    try:
        # 创建局部mask
        mask_local = np.zeros(roi_shape, dtype=np.uint8)
        contour_int = np.round(contour).astype(int)
        contour_int[:, 0] = np.clip(contour_int[:, 0], 0, mask_local.shape[0] - 1)
        contour_int[:, 1] = np.clip(contour_int[:, 1], 0, mask_local.shape[1] - 1)

        # 填充轮廓
        cv2.fillPoly(mask_local, [contour_int[:, ::-1]], 1)

        # 转换到全局坐标
        contour_global = contour.copy()
        contour_global[:, 0] += x_offset  # x坐标（列方向）
        contour_global[:, 1] += y_offset  # y坐标（行方向）

        # 创建全局mask
        mask_global = np.zeros((input_height, input_width), dtype=np.uint8)
        h, w = mask_local.shape

        # 确保不越界
        y_end = min(y_offset + h, input_height)
        x_end = min(x_offset + w, input_width)
        h_valid = y_end - y_offset
        w_valid = x_end - x_offset

        if h_valid > 0 and w_valid > 0:
            mask_global[y_offset:y_end, x_offset:x_end] = mask_local[:h_valid, :w_valid]

        # 计算边界框
        y_local, x_local = np.where(mask_local > 0)
        if len(x_local) == 0 or len(y_local) == 0:
            return None

        x_min_local, x_max_local = np.min(x_local), np.max(x_local)
        y_min_local, y_max_local = np.min(y_local), np.max(y_local)

        bbox_global = (
            int(x_min_local + x_offset),
            int(y_min_local + y_offset),
            int(x_max_local + x_offset),
            int(y_max_local + y_offset)
        )

        child_segment = {
            'id': f"{parent_segment['id']}-{child_index + 1}",
            'parent_id': parent_segment['id'],
            'contour': contour_global,
            'mask': mask_global,
            'bbox': bbox_global,
            'area': int(np.sum(mask_local)),
            'level': 2  # 标记为第二级分割
        }

        return child_segment

    except Exception as e:
        print(f"创建子区域失败: {e}")
        return None


def process_single_image_segmentation(image_path, net, visualize=True):
    print(f"\n开始处理图像: {os.path.basename(image_path)}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, [], 0, [], [], None, []

    image = cv2.resize(image, (input_width, input_height))
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_img = img_gray.astype(np.float32) / 255.0
    input_tensor = torch.tensor(normalized_img).unsqueeze(0).unsqueeze(0).float().to(device)

    net.eval()
    with torch.no_grad():
        parameters = net(input_tensor)
    parameters = parameters.squeeze()
    if parameters.dim() == 0: parameters = parameters.unsqueeze(0)
    if parameters.numel() != 4:
        raise ValueError(f"Expected 4 parameters, got {parameters.numel()}")
    print(f"Predicted parameters: {parameters.cpu().numpy()}")

    sigma = max(np.std(normalized_img) * 2, 0.1)
    print(f"Adaptive sigma: {sigma:.3f}")

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
    contours = DeleteSmall(contours, min_area=image.shape[0] * image.shape[1] * 0.0001)
    segments = extract_segments_from_contours(image, contours, phi_np)
    segment_count = len(segments)
    print(f"第一次分割: {segment_count} 个区域")

    # 提取前景掩码
    foreground_mask = extract_foreground_mask(segments, image.shape)
    background_mask = 1 - foreground_mask

    background_area = np.sum(background_mask)
    total_area = image.shape[0] * image.shape[1]
    print(f"背景区域面积: {background_area}/{total_area} ({background_area / total_area * 100:.1f}%)")

    # 二次分割 - 背景区域（保持原来的水平集方法）
    second_segmentation_results = []
    if background_area > 500:
        background_image = image.copy()
        background_image[background_mask == 0] = 0
        lab_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2LAB)
        b_channel = lab_image[:, :, 2]
        b_bg = b_channel[background_mask == 1]
        normalized_b = (b_channel.astype(np.float32) - b_bg.min()) / (b_bg.max() - b_bg.min() + 1e-8)
        img_tensor = torch.tensor(normalized_b).to(device)

        initial_lsf_sec = torch.ones_like(img_tensor) * 0.3
        background_indices = torch.tensor(background_mask, device=device) > 0
        initial_lsf_sec[background_indices] = 0.7

        var_b = np.var(b_bg)
        lambda1 = lambda2 = 1.0 + var_b * 5;
        alpha1 = alpha2 = 0.5
        parameters_sec = torch.tensor([lambda1, lambda2, alpha1, alpha2], device=device)

        phi_sec = change_lsf(
            img_tensor,
            initial_lsf_sec,
            iter_num=30,
            sigma=sigma,
            lambda1=parameters_sec[0].float(),
            lambda2=parameters_sec[1].float(),
            alpha1=parameters_sec[2].float(),
            alpha2=parameters_sec[3].float()
        )
        phi_sec_np = phi_sec.cpu().detach().numpy()
        contours_sec = measure.find_contours(phi_sec_np, level=0.4)
        min_area = image.shape[0] * image.shape[1] * 0.0005
        filtered_contours_sec = DeleteSmall(contours_sec, min_area=min_area)
        if filtered_contours_sec:
            second_segmentation_results = [{
                'contours': filtered_contours_sec,
                'second_phi': phi_sec_np,
                'num_sub_segments': len(filtered_contours_sec)
            }]
            print(f"背景二次分割为 {len(filtered_contours_sec)} 个子区域")
        else:
            print("背景区域未找到有效的二次分割结果")

    # 前景二次分割 - 使用热力图方法
    foreground_second_segments = []
    foreground_area = np.sum(foreground_mask)
    if foreground_area > 500:
        print("开始前景区域热力图二次分割...")
        foreground_second_segments = perform_foreground_heatmap_segmentation(segments, image, phi_np)
        print(f"前景热力图二次分割完成: {len(foreground_second_segments)} 个子区域")
    else:
        print("前景区域太小，跳过热力图二次分割")

    # 可视化六图
    if visualize:
        fig, axes = plt.subplots(2, 3, figsize=(25, 10))

        # 第一行
        # 图1 原图
        axes[0, 0].imshow(img_rgb);
        axes[0, 0].set_title('Original Image');
        axes[0, 0].axis('off')
        # 图2 一次分割热力图
        axes[0, 1].imshow(phi_np, cmap='viridis')
        for contour in contours: axes[0, 1].plot(contour[:, 1], contour[:, 0], color='red')
        axes[0, 1].set_title('Primary Segmentation');
        axes[0, 1].axis('off')
        # 图3 仅一次分割轮廓
        axes[0, 2].imshow(img_rgb)
        for seg in segments: axes[0, 2].plot(seg['contour'][:, 1], seg['contour'][:, 0], color='red')
        axes[0, 2].set_title('Primary Contours Only');
        axes[0, 2].axis('off')

        # 第二行
        # 图4 背景高亮 + 一次分割轮廓
        background_overlay = img_rgb.copy().astype(np.float32)
        alpha = 0.4;
        yellow = np.array([255, 255, 0], dtype=np.float32)
        mask = background_mask == 1
        background_overlay[mask] = (1 - alpha) * background_overlay[mask] + alpha * yellow
        background_overlay = np.clip(background_overlay, 0, 255).astype(np.uint8)
        axes[1, 0].imshow(background_overlay)
        for seg in segments: axes[1, 0].plot(seg['contour'][:, 1], seg['contour'][:, 0], color='red')
        axes[1, 0].set_title('Background Highlighted + Primary Contours');
        axes[1, 0].axis('off')

        # 图5 一次 + 二次分割轮廓
        axes[1, 1].imshow(img_rgb)
        for seg in segments: axes[1, 1].plot(seg['contour'][:, 1], seg['contour'][:, 0], color='red', linewidth=1.5)
        for result in second_segmentation_results:
            for contour in result['contours']:
                axes[1, 1].plot(contour[:, 1], contour[:, 0], color='yellow', linewidth=2.0)
        axes[1, 1].set_title('Primary (red) + Background Secondary (yellow)');
        axes[1, 1].axis('off')

        # 图6 一次 + 背景二次分割 + 前景二次分割轮廓
        axes[1, 2].imshow(img_rgb)
        for seg in segments:
            axes[1, 2].plot(seg['contour'][:, 1], seg['contour'][:, 0], color='red', linewidth=1.5)
        # 添加背景二次分割的黄线
        for result in second_segmentation_results:
            for contour in result['contours']:
                axes[1, 2].plot(contour[:, 1], contour[:, 0], color='yellow', linewidth=2.0)
        # 添加前景热力图分割的青线
        for seg in foreground_second_segments:
            axes[1, 2].plot(seg['contour'][:, 1], seg['contour'][:, 0], color='cyan', linewidth=2, linestyle='-')
        axes[1, 2].set_title('Primary (red) + Background Secondary (yellow) + Foreground Heatmap (cyan)');
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

    total_background_secondary = sum(
        [res['num_sub_segments'] for res in second_segmentation_results]) if second_segmentation_results else 0
    total_foreground_secondary = len(foreground_second_segments)

    print(f"✅ 处理完成:")
    print(f"   第一次分割: {segment_count} 个区域")
    print(f"   背景二次分割: {total_background_secondary} 个子区域")
    print(f"   前景热力图二次分割: {total_foreground_secondary} 个子区域")
    print(f"   前景像素总数: {np.sum(foreground_mask)}")

    return phi_np, contours, segment_count, segments, second_segmentation_results, background_mask, foreground_second_segments


# 主函数
def main_single_image():
    print("开始单张岩石图像鲁棒分割处理...")
    image_path = "/home/user2/HML/4image/Pic_1005.jpg"

    if not os.path.exists(image_path):
        print(f"❌ 图像不存在: {image_path}")
        return
    try:
        phi_result, contours, segment_count, segments, second_results, background_mask, foreground_second_results = process_single_image_segmentation(
            image_path, net, visualize=True
        )
        total_background_secondary = sum([res['num_sub_segments'] for res in second_results]) if second_results else 0
        total_foreground_secondary = len(foreground_second_results)

        print(f"✅ 处理完成:")
        print(f"   第一次分割: {segment_count} 个区域")
        print(f"   背景二次分割: {total_background_secondary} 个子区域")
        print(f"   前景热力图二次分割: {total_foreground_secondary} 个子区域")
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