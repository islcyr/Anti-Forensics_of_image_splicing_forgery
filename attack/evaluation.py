# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/5 15:04
@Auth ： Yin yanquan
@File ：evaluation.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""

import math
import numpy as np
import cv2


def PSNR(img1, img2):
    """
    计算PSNR值
    :param img1: 图片1
    :param img2: 图片2
    :return: 两张图片的PSNR值
    """
    mse = np.mean((img1 - img2) ** 2)
    pixel_max = 255.0
    if mse == 0:
        return float('inf')
    psnr = 10 * math.log10(pixel_max / math.sqrt(mse))
    return psnr


def SSIM(img1, img2):
    """
    计算SSIM值（单通道）
    :param img1: 图片1
    :param img2: 图片2
    :return: 两张图片的SSIM值（单通道）
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """
    计算SSIM值（多通道）
    :param img1: 图片1
    :param img2: 图片2
    :return: 两张图片的SSIM值（多通道）
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return SSIM(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(SSIM(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return SSIM(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def cal_ssim_psnr(image1, image2):
    """
    测试PSNR与SSIM的数值
    :param img1: 图片1
    :param img2: 图片2
    :return: PSNR,SSIM
    """
    img1 = np.array(cv2.imread(image1, 0))
    img2 = np.array(cv2.imread(image2, 0))

    ss = calculate_ssim(img2, img1)
    # np.int8转np.float64,提升精度
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    ps = PSNR(img2, img1)
    return ps, ss

    # 直接调包
    # from skimage.measure import compare_ssim, compare_psnr
    # from skimage.metrics import peak_signal_noise_ratio as psnr
    # ssim = compare_ssim(img1, img2,data_range=255,multichannel=True)
    # psnr = compare_psnr(img1, img2, 255)


def BGR_to_RGB(cvimg):
    """
    将cv读入的图片由BGR转为RGB
    :param cvimg: 对应图片
    :return: 转换后的图片
    """
    pilimg = cvimg.copy()
    pilimg[:, :, 0] = cvimg[:, :, 2]
    pilimg[:, :, 2] = cvimg[:, :, 0]
    return pilimg
