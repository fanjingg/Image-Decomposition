import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2 as cv
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
qj = None
norm_name = {"bn": nn.BatchNorm2d}
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "relu6": nn.ReLU6,
}

class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).ta
class Swish(nn.Module):  #
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))
        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))
    def forward(self, x):
        return self.F.apply(x)

class Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride, pad, groups=1, dila=1, norm=None, activate='relu'):
        super(Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        self.__conv = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                                stride=stride, padding=pad, bias=not norm, groups=groups, dilation=dila)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "Mish":
                self.__activate = activate_name[activate]()
            if activate == "Swish":
                self.__activate = activate_name[activate]()
            if activate == "MEMish":
                self.__activate = activate_name[activate]()
            if activate == "MESwish":
                self.__activate = activate_name[activate]()
            if activate == "FReLu":
                self.__activate = activate_name[activate]()

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x


def zh(img,str1,str2):
    toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素
    pic = toPIL(img)
    pic.save(str1 + '/' + str2 +'.jpg')

class Sobel_Edge_Block(nn.Module):
    def __init__(self, channel_in, alpha=0.5, sigma=4):
        super(Sobel_Edge_Block, self).__init__()
        self.__down0 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.__sobelconv = Sobel_conv(channel_in, channel_in, alpha, sigma)
        self.__down1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.__conv0 = Convolutional(channel_in, 2, kernel_size=1, stride=1, pad=0, norm='bn', activate='relu')

    def forward(self, x):
        # x_down0 = self.__down0(x)
        x_sobel = self.__sobelconv(x)
        # x_down1 = self.__down1(self.__down1(x_sobel))
        # x_conv0 = self.__conv0(x_down1)
        return x_sobel


class Sobel_conv(nn.Module):
    def __init__(self, channel_in, channel_out, alpha=0.5, sigma=4, stride=1, padding=1):
        super(Sobel_conv, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.sigma = sigma
        self.alpha = alpha
        self.__conv_weight = Convolutional(channel_out * 4, 4, kernel_size=1, stride=1, pad=0, norm='bn', activate='leaky')
        self.theta = nn.Parameter(torch.sigmoid(torch.randn(channel_out) * 1.0) + self.alpha, requires_grad=True)

    def forward(self, x):
        # [channel_out, channel_in, kernel, kernel]
        kernel0, kernel45, kernel90, kernel135 = sobel_kernel(self.channel_in, self.channel_out, self.theta)
        kernel0 = kernel0.float()
        kernel45 = kernel45.float()
        kernel90 = kernel90.float()
        kernel135 = kernel135.float()

        out0 = F.conv2d(x, kernel0, stride=self.stride, padding=self.padding)
        out45 = F.conv2d(x, kernel45, stride=self.stride, padding=self.padding)
        out90 = F.conv2d(x, kernel90, stride=self.stride, padding=self.padding)
        out135 = F.conv2d(x, kernel135, stride=self.stride, padding=self.padding)
        zh(out0[0],'1',qj)
        zh(out45[0],'2',qj)
        zh(out90[0],'3',qj)
        zh(out135[0],'4',qj)

        out_cat = torch.cat((out0, out45, out90, out135),1)
        out_cat_conv = self.__conv_weight(out_cat)
        out_weight = F.softmax(out_cat_conv, dim=1)

        # out = torch.abs(out0)* out_weight[:,0:1,:,:] + torch.abs(out45)*out_weight[:,1:2,:,:]\
        #       + torch.abs(out90)*out_weight[:,2:3,:,:] + torch.abs(out135)*out_weight[:,3:,:,:]
        # out = torch.abs(out0) * 0.25 + torch.abs(out45) * 0.25\
        #       + torch.abs(out90) * 0.25+ torch.abs(out135) * 0.25
        out = torch.abs(out0) * 0.25 + torch.abs(out45) * 0.25 + torch.abs(out90) * 0.25+ torch.abs(out135) * 0.25
        out = (out * self.sigma)
        # zh(out[0], '1', '5')
        return out



def sobel_kernel(channel_in, channel_out, theta):
    sobel_kernel0 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel0 = sobel_kernel0.reshape((1, 1, 3, 3))
    sobel_kernel0 = Variable(torch.from_numpy(sobel_kernel0))
    sobel_kernel0 = sobel_kernel0.repeat(channel_out, channel_in, 1, 1).float()
    sobel_kernel0 = sobel_kernel0.cuda()*theta.view(-1, 1, 1, 1).cuda()

    sobel_kernel45 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel45 = sobel_kernel45.reshape((1, 1, 3, 3))
    sobel_kernel45 = Variable(torch.from_numpy(sobel_kernel45))
    sobel_kernel45 = sobel_kernel45.repeat(channel_out, channel_in, 1, 1).float()
    sobel_kernel45 = sobel_kernel45.cuda()*theta.view(-1, 1, 1, 1).cuda()

    sobel_kernel90 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32')
    sobel_kernel90 = sobel_kernel90.reshape((1, 1, 3, 3))
    sobel_kernel90 = Variable(torch.from_numpy(sobel_kernel90))
    sobel_kernel90 = sobel_kernel90.repeat(channel_out, channel_in, 1, 1).float()
    sobel_kernel90 = sobel_kernel90.cuda()*theta.view(-1, 1, 1, 1).cuda()

    sobel_kernel135 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel135 = sobel_kernel135.reshape((1, 1, 3, 3))
    sobel_kernel135 = Variable(torch.from_numpy(sobel_kernel135))
    sobel_kernel135 = sobel_kernel135.repeat(channel_out, channel_in, 1, 1).float()
    sobel_kernel135 = sobel_kernel135.cuda()*theta.view(-1, 1, 1, 1).cuda()

    return sobel_kernel0, sobel_kernel45, sobel_kernel90, sobel_kernel135

import random
p = random.uniform(-0.3,0.1)
import torchvision.transforms as transforms
img = cv.imread('000004.png')

import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
root = 'G:\\exdark\\fog\\FOG1\\fog\\'
g = os.listdir(root)
for i in g:
    qj = i
    tpt = root + i
    img = cv2.imread(tpt)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
    transf = transforms.ToTensor()
    img_tensor = transf(img).to(torch.float32) # tensor数据格式是torch(C,H,W)
    img_tensor =  img_tensor.unsqueeze(0).cuda()
    print(img_tensor.size())
    model =Sobel_Edge_Block(3).cuda()
    k = model(img_tensor)
    zh(k[0],'5',i)
    zh(img_tensor[0] - k[0],'6',i)
