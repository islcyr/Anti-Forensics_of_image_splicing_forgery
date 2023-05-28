# -*- coding: utf-8 -*-
"""
@Time ： 2023/1/24 0:08
@Auth ： Yin yanquan
@File ：utils.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from attack import *


def perturbed_plot_fgsm(epsilons):
    """
    模型输入FGSM对抗样本后的结果绘图
    :param epsilons: 对抗样本对应的eps
    :return: None
    """
    plt.suptitle("FGSM Attack (PSNR/SSIM)")
    plt.subplot(3, 3, 1)
    plt.axis("off")
    image_ori = './output/fgsm/1_perturbed_output_eps_0.jpg'
    image_data_ori = np.array(BGR_to_RGB(cv2.imread(image_ori, 1)))
    plt.imshow(image_data_ori)
    plt.title('origin image')

    for i in range(1, len(epsilons)):
        num = i + 1
        plt.subplot(3, 3, i + 3)
        plt.axis("off")
        image = './output/fgsm/%%s_perturbed_output_eps_%s.jpg' % epsilons[i] % num
        image_data = np.array(BGR_to_RGB(cv2.imread(image, 1)))
        plt.imshow(image_data)
        psnr, ssim = cal_ssim_psnr(image_ori, image)
        plt.title('%.2f/%%.2f' % psnr % ssim)

    plt.subplots_adjust(left=0, right=1, bottom=0.1, top=0.9, wspace=0, hspace=0.3)
    plt.show()


def perturbed_plot_pgd(epsilons):
    """
    模型输入PGD对抗样本后的结果绘图
    :param epsilons: 对抗样本对应的eps
    :return: None
    """
    plt.suptitle("PGD Attack (PSNR/SSIM)")
    plt.subplot(3, 3, 1)
    plt.axis("off")
    image_ori = './output/pgd/1_perturbed_output_eps_0.jpg'
    image_data_ori = np.array(BGR_to_RGB(cv2.imread(image_ori, 1)))
    plt.imshow(image_data_ori)
    plt.title('origin image')

    for i in range(1, len(epsilons)):
        num = i + 1
        plt.subplot(3, 3, i + 3)
        plt.axis("off")
        image = './output/pgd/%%s_perturbed_output_eps_%s.jpg' % epsilons[i] % num
        image_data = np.array(BGR_to_RGB(cv2.imread(image, 1)))
        plt.imshow(image_data)
        psnr, ssim = cal_ssim_psnr(image_ori, image)
        plt.title('%.2f/%%.2f' % psnr % ssim)

    plt.subplots_adjust(left=0, right=1, bottom=0.1, top=0.9, wspace=0, hspace=0.3)
    plt.show()


def output_plot_fgsm():
    """
    FGSM对抗样本绘图
    :return: None
    """
    plt.suptitle("FGSM Attack")
    plt.subplot(3, 3, 1)
    plt.axis("off")
    image_ori = './output/fgsm/1_fgsm_output_eps_0.jpg'
    image_data_ori = np.array(BGR_to_RGB(cv2.imread(image_ori, 1)))
    plt.imshow(image_data_ori)
    plt.title('origin mask')

    for i in range(1, len(epsilons)):
        num = i + 1
        plt.subplot(3, 3, i + 3)
        plt.axis("off")
        image = './output/fgsm/%%s_fgsm_output_eps_%s.jpg' % epsilons[i] % num
        image_data = np.array(BGR_to_RGB(cv2.imread(image, 1)))
        plt.imshow(image_data)
        plt.title("eps=%s" % epsilons[i])

    plt.subplots_adjust(left=0, right=1, bottom=0.1, top=0.9, wspace=0, hspace=0.3)
    plt.show()


def output_plot_pgd():
    """
    PGD对抗样本绘图
    :return: None
    """
    plt.suptitle("PGD Attack")
    plt.subplot(3, 3, 1)
    plt.axis("off")
    image_ori = './output/pgd/1_pgd_output_eps_0.jpg'
    image_data_ori = np.array(BGR_to_RGB(cv2.imread(image_ori, 1)))
    plt.imshow(image_data_ori)
    plt.title('origin mask')

    for i in range(1, len(epsilons)):
        num = i + 1
        plt.subplot(3, 3, i + 3)
        plt.axis("off")
        image = './output/pgd/%%s_pgd_output_eps_%s.jpg' % epsilons[i] % num
        image_data = np.array(BGR_to_RGB(cv2.imread(image, 1)))
        plt.imshow(image_data)
        plt.title("eps=%s" % epsilons[i])

    plt.subplots_adjust(left=0, right=1, bottom=0.1, top=0.9, wspace=0, hspace=0.3)
    plt.show()


def output_plot_SENetwork():
    """
    PGD对抗样本输入SENetwork后结果绘图
    :return: None
    """
    plt.suptitle("PGD Attack: RRU2SE")
    plt.subplot(3, 3, 1)
    plt.axis("off")
    image_ori = './output/pgd/1_pgd_output_eps_0.jpg'
    image_ori = cv2.resize(cv2.imread(image_ori, 1), (384, 384))
    image_data_ori = np.array(BGR_to_RGB(image_ori))
    plt.imshow(image_data_ori)
    plt.title('origin mask')

    for i in range(1, len(epsilons)):
        num = i + 1
        plt.subplot(3, 3, i + 3)
        plt.axis("off")
        image = './output/SENetwork/%syu.jpg' % num
        image_data = np.array(BGR_to_RGB(cv2.imread(image, 1)))
        plt.imshow(image_data)

        pu_mask = './attack/IOU.jpg'
        Fscore, _, _ = usage('./output/pgd/1_pgd_output_eps_0.jpg', image, pu_mask)
        plt.title("eps=%s,F_score=%%.2f" % epsilons[i] % Fscore)

    plt.subplots_adjust(left=0, right=1, bottom=0.1, top=0.9, wspace=0, hspace=0.3)
    plt.show()


if __name__ == '__main__':
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    # perturbed_plot_pgd(epsilons)
    output_plot_fgsm()
    # output_plot_pgd()
    # output_plot_SENetwork()
