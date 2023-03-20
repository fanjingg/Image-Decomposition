import numpy as np
import cv2

def decompose_image(image):
    # 将彩色图像转为灰度图像
    gray = image
    
    # 对灰度图像进行高斯滤波处理
    blur = cv2.GaussianBlur(gray, (15,15), 0)#15为高斯模糊的窗口大小，可以调节
    
    # 计算灰度图像与模糊图像之间的差值
    diff = cv2.subtract(gray, blur)
    
    # 将差值图像进行拉普拉斯变换，得到图像的结构层
    laplacian = cv2.Laplacian(diff, cv2.CV_64F,ksize = 5)#ksize=1、3、5、7可调节
    laplacian = cv2.convertScaleAbs(laplacian)
    #print(laplacian)
    struct = np.uint8(laplacian)
    

    # 将灰度图像减去结构层即可得到图像的纹理层
    texture = cv2.subtract(gray, struct)

    # 返回得到的结构层和纹理层
    return struct, texture

# 读取待处理的图像
image = cv2.imread("2.jpg")
b,g,r = cv2.split(image)
# 调用图像分解函数进行分解操作
b_struct, b_texture = decompose_image(b)
g_struct, g_texture = decompose_image(g)
r_struct, r_texture = decompose_image(r)

struct = cv2.merge((b_struct,g_struct,r_struct))
texture = cv2.merge((b_texture,g_texture,r_texture))
# 显示结构层和纹理层
cv2.imshow("Structural layer", struct)
cv2.imshow("Textural layer", texture)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 在这个程序中，我们首先将彩色图像转换为灰度图像，然后使用高斯滤波对灰度图像进行模糊处理，
# 得到一张模糊的图像。接着，我们将灰度图像和模糊图像作差，得到一个差值图像，然后将差值图像
# 进行拉普拉斯变换，得到图像的结构层。最后，我们将灰度图像减去结构层，即得到图像的纹理层。
