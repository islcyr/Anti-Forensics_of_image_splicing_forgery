# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/5 15:00
@Auth ： Yin yanquan
@File ：FGSM.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fgsm_attack(image, epsilon, data_grad):
    """
    按照FGSM方法制作对抗样本
    :param image: 输入图像
    :param epsilon: 沿梯度正方向添加扰动的eps的值
    :param data_grad: 输入图像的梯度数据
    :return: 添加扰动后的图像
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def fgsm_attack_restricted(image, result, epsilon, data_grad):
    value = 10

    perturbed_image = image + 0 * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    image = image.cpu().detach().numpy()
    result = result.cpu().detach().numpy()
    perturbed_image = perturbed_image.cpu().detach().numpy()

    sign_data_grad = data_grad.sign()
    sign_data_grad = sign_data_grad.cpu().detach().numpy()

    # result.shape = (1,1,256,384)
    # result.shape = (1,256,384)
    # sign_data_grad.shape = (1,3,256,384)
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
                        perturbed_image[m][n][i][j] = image[m][n][i][j] + epsilon * sign_data_grad[m][n][i][j]

    perturbed_image = torch.from_numpy(perturbed_image)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
