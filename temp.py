# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/24 16:48
@Auth ： Yin yanquan
@File ：temp.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st

# def gkern(kernlen=15, nsig=3):
#     x = np.linspace(-nsig, nsig, kernlen)
#     kern1d = st.norm.pdf(x)
#     kernel_raw = np.outer(kern1d, kern1d)
#     kernel = kernel_raw / kernel_raw.sum()
#     return kernel
#
# kernel_size = 5
# kernel = gkern(kernel_size, 3).astype(np.float32)
# gaussian_kernel = np.stack([kernel, kernel, kernel])
# gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
# gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
#
# # print(gaussian_kernel)
#
#
#
#
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# X_ori = torch.zeros(1, 3, 28, 28).to(device)
# delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
#
# class Normalize(nn.Module):
#     def __init__(self, mean, std):
#         super(Normalize, self).__init__()
#         self.mean = torch.Tensor(mean)
#         self.std = torch.Tensor(std)
#
#     def forward(self, x):
#         return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]
# norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
# ##define DI
# def DI(X_in):
#     rnd = np.random.randint(299, 330, size=1)[0]
#     h_rem = 330 - rnd
#     w_rem = 330 - rnd
#     pad_top = np.random.randint(0, h_rem, size=1)[0]
#     pad_bottom = h_rem - pad_top
#     pad_left = np.random.randint(0, w_rem, size=1)[0]
#     pad_right = w_rem - pad_left
#
#     # print(rnd,pad_left,pad_top,pad_right,pad_bottom)
#     c = np.random.rand(1)
#     if c <= 0.7:
#         X_out = F.pad(F.interpolate(X_in, size=(rnd, rnd)), (pad_left, pad_top, pad_right, pad_bottom), mode='constant',
#                       value=0)
#         return X_out
#     else:
#         return X_in
#
# logits = norm(X_ori + delta)
# print(DI(X_ori + delta).shape)
# print(logits.shape)

# np.set_printoptions(suppress = True)
# np.set_printoptions(threshold=np.inf)
#
# gt = cv2.imread('example_gt.png')
# with open('temp.txt','w') as f:
#     f.write(str(gt))

# import matplotlib.pyplot as plt
# import math
#
# x = np.linspace(-5, 5, 200)
# y = 1 / (1 + np.exp(-x))
# y1 = 1 / (1 + np.exp(-x / 0.5))
# y2 = 1 / (1 + np.exp(-x / 2))
# ax = plt.subplot(111)
# # 坐标轴
# ax = plt.gca()  # get current axis 获得坐标轴对象
# ax.spines['right'].set_color('none')  # 将右边 边沿线颜色设置为空 其实就相当于抹掉这条边
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# # 设置中心的为（0，0）的坐标轴
# ax.spines['bottom'].set_position(('data', 0))  # 指定 data 设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
# ax.spines['left'].set_position(('data', 0))
#
# ax.plot(x, y, label="T=1", color="blueviolet")
# ax.plot(x, y1, label="T=0.5", color="red")
# ax.plot(x, y2, label="T=2", color="cyan")
# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def result_plot():
    x = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    y0 = np.array([0.49, 0.31, 0.19, 0.14, 0.12, 0.10])
    y1 = np.array([0.46, 0.27, 0.19, 0.13, 0.10, 0.10])
    y2 = np.array([0.38, 0.24, 0.16, 0.12, 0.10, 0.10])
    y3 = np.array([0.34, 0.21, 0.14, 0.11, 0.10, 0.09])
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y0_smooth = make_interp_spline(x, y0)(x_smooth)
    y1_smooth = make_interp_spline(x, y1)(x_smooth)
    y2_smooth = make_interp_spline(x, y2)(x_smooth)
    y3_smooth = make_interp_spline(x, y3)(x_smooth)

    plt.scatter(x, y0, c='black', alpha=0.5)
    plt.scatter(x, y1, c='black', alpha=0.5)
    plt.scatter(x, y2, c='black', alpha=0.5)
    plt.scatter(x, y3, c='black', alpha=0.5)

    plt.plot(x_smooth, y0_smooth, label='T=0.5', c='black')
    plt.plot(x_smooth, y1_smooth, label='T=1', c='deepskyblue')
    plt.plot(x_smooth, y2_smooth, label='T=2', c='red')
    plt.plot(x_smooth, y3_smooth, label='T=3', c='green')

    plt.title('The relationship between average F-scores and eps for different T')
    plt.xlabel('epsilon')
    plt.ylabel('f_score')
    plt.legend(loc="upper right")
    plt.savefig("./output0/result.png", dpi=300)
    plt.show()


result_plot()


# import cv2
# for i in range(100):
#     num = i+1
#     image_ori = './UAP/data/SAN/%s.jpg' % num
#     image_ori = cv2.resize(cv2.imread(image_ori, 1), (256, 256))
#     print(image_ori.shape)
#     print(num)
#     cv2.imwrite('./UAP/data/re/%s.jpg' % num , image_ori)


# import matplotlib.pyplot as plt
# from scipy.interpolate import make_interp_spline
# loss = []
#
# for line in open('./WSL/log/loss.log','r'):
#     line = float(line.replace('\n',''))
#     loss.append(line)
#
# print(loss)
#
# x = np.array(range(1,3233))
# y = np.array(loss)
# # x_smooth = np.linspace(x.min(), x.max(), 300)
# # y_smooth = make_interp_spline(x, y)(x_smooth)
# x1 = np.array([0,500,1000,1500,2000,2500,3000,3232])
# y1 = np.array([0.05,0.80,0.98,0.99,0.99,0.99,0.99,0.99])
# x1_smooth = np.linspace(x1.min(), x1.max(), 3000)
# y1_smooth = make_interp_spline(x1, y1)(x1_smooth)
#
# plt.scatter(x, y, c='black', alpha=1, s=2, label='loss')
# # for xy in zip(x1, y1):
# #     plt.annotate("(%.2f,%.2f)" % xy,xy=xy)
# plt.plot(x1_smooth, y1_smooth, c='deepskyblue',label='F score')
#
# # plt.title('')
# plt.xlabel('batch num')
# # plt.ylabel('loss')
# plt.legend(loc="upper left")
# plt.savefig('./WSL/log/loss.png', dpi=300)
# plt.show()