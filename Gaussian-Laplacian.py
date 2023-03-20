import cv2
import numpy as np

def image_decomposition(image_path, ksize=5, k=4):
    # 读取图像并转换为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 使用高斯滤波器平滑图像
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # 计算拉普拉斯算子
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)

    # 将拉普拉斯算子转换为与平滑后图像相同的数据类型
    laplacian = np.asarray(laplacian, dtype=blurred_image.dtype)

    # 计算结构层
    structure_layer = cv2.addWeighted(blurred_image, 1, laplacian, k, 0)

    return structure_layer

# 示例
image_path = "000012.png"
structure_layer, texture_layer = image_decomposition(image_path)

cv2.imwrite("structure_layer.jpg", structure_layer)
