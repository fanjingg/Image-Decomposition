import cv2 
import numpy as np 

# 单通道
# f为相似窗口的半径, t为搜索窗口的半径, h为高斯函数平滑参数(一般取为相似窗口的大小)
def make_kernel(f):
    kernel = np.zeros((2*f+1, 2*f+1), np.float32)
    for d in range(1, f+1):
        kernel[f-d:f+d+1, f-d:f+d+1] += (1.0/((2*d+1)**2))

    return kernel/kernel.sum()

def NLmeans_filter(src, f, t, h):
    H, W = src.shape
    out = np.zeros((H, W), np.uint8)
    pad_length = f+t
    src_padding = np.pad(src, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    kernel = make_kernel(f)
    h2 = h*h

    for i in range(0, H):
        for j in range(0, W):
            i1 = i + f + t
            j1 = j + f + t
            W1 = src_padding[i1-f:i1+f+1, j1-f:j1+f+1] # 领域窗口W1
            w_max = 0
            aver = 0
            weight_sum = 0
            # 搜索窗口
            for r in range(i1-t, i1+t+1):
                for c in range(j1-t, j1+t+1):
                    if (r==i1) and (c==j1):
                        continue
                    else:
                        W2 = src_padding[r-f:r+f+1, c-f:c+f+1] # 搜索区域内的相似窗口
                        Dist2 = (kernel*(W1-W2)*(W1-W2)).sum()
                        w = np.exp(-Dist2/h2)
                        if w > w_max:
                            w_max = w
                        weight_sum += w
                        aver += w*src_padding[r, c]
            aver += w_max*src_padding[i1, j1] # 自身领域取最大的权重
            weight_sum += w_max
            out[i, j] = aver/weight_sum

    return out

img = cv2.imread("lena256gs10.bmp", 0)
out = NLmeans_filter(img, 2, 5, 10)
cv2.imwrite("result.bmp", out)
————————————————
版权声明：本文为CSDN博主「羊同学学Python」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/yy0722a/article/details/113924087
