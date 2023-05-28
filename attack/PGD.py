# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/5 15:04
@Auth ： Yin yanquan
@File ：PGD.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def pgd_attack(image, ori_image, data_grad, eps, alpha):
    """
    按照PGD方法制作对抗样本
    :param image: 输入图像
    :param ori_image: 输入图像的具体数据
    :param data_grad: 输入图像的梯度数据
    :param eps: 限制添加的扰动固定在一定范围内的值
    :param alpha: 添加扰动的程度的值
    :return: 添加扰动后生成的图像
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image + alpha * sign_data_grad
    eta = torch.clamp(perturbed_image - ori_image, min=-eps, max=eps)
    perturbed_image = torch.clamp(ori_image + eta, min=0, max=1)
    return perturbed_image


def pgd_attack_restricted(image, ori_image, result, data_grad, eps, alpha):
    value = 10

    image = image.cpu().detach().numpy()
    result = result.cpu().detach().numpy()
    perturbed_image = image

    sign_data_grad = data_grad.sign()
    sign_data_grad = sign_data_grad.cpu().detach().numpy()

    label_image = np.zeros_like(result[0])

    for i in range(image.shape[2]):
        for j in range(image.shape[3]):
            if result[0][i][j] > 0.5:
                for k in range(max(i - value, 0), min(i + value, image.shape[2])):
                    for l in range(max(j - value, 0), min(j + value, image.shape[3])):
                        label_image[k][l] = 1

    for m in range(image.shape[0]):
        for n in range(image.shape[1]):
            for i in range(image.shape[2]):
                for j in range(image.shape[3]):
                    if label_image[i][j] == 0:
                        perturbed_image[m][n][i][j] = image[m][n][i][j] + alpha * sign_data_grad[m][n][i][j]

    perturbed_image = torch.from_numpy(perturbed_image)
    ori_image = ori_image.cpu().detach().numpy()
    ori_image = torch.from_numpy(ori_image)
    eta = torch.clamp(perturbed_image - ori_image, min=-eps, max=eps)
    perturbed_image = torch.clamp(ori_image + eta, min=0, max=1)
    return perturbed_image
