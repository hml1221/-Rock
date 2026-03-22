import cv2
import numpy as np
import matplotlib.pyplot as plt


def kmeans_gray_refined(img_gray, K=3, blur=3, min_area_ratio=0.001):
    """
    改良版灰度 K-means 岩石分割：
    - 在灰度空间做 K-means
    - 先做轻微高斯平滑
    - 再用连通域去噪 + 闭运算补小孔

    参数:
        img_gray       : 输入灰度图 (H, W), uint8
        K              : 聚类个数，默认 3
        blur           : 高斯核大小（必须为奇数），默认 3
        min_area_ratio : 最小保留区域面积 = ratio * (H*W)

    返回:
        mask_bin       : uint8, (H, W)，前景=255，背景=0
    """
    if img_gray.ndim == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    h, w = img_gray.shape[:2]
    min_area = int(h * w * min_area_ratio)

    # ① 轻微平滑，减少颗粒内部纹理噪声
    img_blur = cv2.GaussianBlur(img_gray, (blur, blur), 0)

    # ② 灰度向量化
    X = img_blur.reshape((-1, 1)).astype(np.float32)

    # ③ K-means 聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                200, 0.2)

    _, labels, centers = cv2.kmeans(
        X, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape(h, w)
    centers = centers.squeeze()

    # ④ 假设亮度最大的类是颗粒前景
    fg_id = np.argmax(centers)
    mask = (labels == fg_id).astype(np.uint8)   # 0/1

    # ⑤ 连通域去掉小区域
    num, comp = cv2.connectedComponents(mask)
    mask_clean = np.zeros_like(mask)

    for i in range(1, num):
        area = np.sum(comp == i)
        if area >= min_area:
            mask_clean[comp == i] = 1

    # ⑥ 闭运算补小孔，使颗粒更连贯
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE,
                                  kernel, iterations=1)

    mask_bin = mask_clean.astype(np.uint8) * 255
    return mask_bin


def demo_single_image(image_path,
                      K=3,
                      blur=3,
                      min_area_ratio=0.001,
                      contour_color=(255, 0, 0),
                      contour_thickness=1):
    """
    单张岩石图像 K-means 分割演示：
    显示 原图 / 分割 mask / 轮廓叠加。
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 分割
    mask = kmeans_gray_refined(
        img_gray,
        K=K,
        blur=blur,
        min_area_ratio=min_area_ratio
    )

    # 轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    img_contour = img_rgb.copy()
    cv2.drawContours(img_contour, contours, -1,
                     contour_color, contour_thickness)

    plt.figure(figsize=(5, 5))
    plt.imshow(img_contour)
    plt.axis("off")  # 去坐标轴
    plt.margins(0, 0)  # 去内容外边距
    plt.subplots_adjust(0, 0, 1, 1)  # 填满画布
    plt.show()


if __name__ == "__main__":
    # 换成你的图像路径
    img_path = "/home/user2/HML/4image/Pic_1068.jpg"

    demo_single_image(
        image_path=img_path,
        K=3,
        blur=3,             # 想更光滑可以试 5
        min_area_ratio=0.00001,  # 想更干净可以调大一点 0.002
        contour_color=(255, 0, 0),
        contour_thickness=1
    )
