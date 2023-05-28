# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/13 21:13
@Auth ： Yin yanquan
@File ：ad_ex_test.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import cv2
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from eval import eval_net
from unet.unet_model import *
from utils import *
from attack import *
import ad_ex

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Ringed_Res_Unet(n_channels=3, n_classes=1)
model = net.to(device)
pretrain_model = torch.load(r'./best_model.pth')
model.load_state_dict(pretrain_model)
model.eval()


def result_plot(epsilons, list_, y_name, title):
    x = np.array(epsilons)
    y = np.array(list_)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, y)(x_smooth)

    plt.scatter(x, y, c='black', alpha=0.5)
    for xy in zip(x, y):
        plt.annotate("(%.2f,%.2f)" % xy, xy=xy, xytext=(-20, 10), textcoords='offset points')
    plt.plot(x_smooth, y_smooth, c='deepskyblue')

    plt.title(title)
    plt.xlabel('epsilon')
    plt.ylabel(y_name)
    plt.savefig("./output0/%s.png" % y_name, dpi=300)
    plt.show()


def adversarial_example_test(temperature=2.0, USE_FGSM_label=True):
    torch.cuda.empty_cache()

    epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    image_nums = 10
    PSNR_mean_list = []
    SSIM_mean_list = []
    f_score_mean_list = []
    ASR_list = []

    for i in range(len(epsilons)):
        PSNR_list = [[], [], [], [], [], []]
        SSIM_list = [[], [], [], [], [], []]
        f_score_list = [[], [], [], [], [], []]
        count = 0

        for file_index in range(image_nums):
            if (file_index + 1) % 5 == 0:
                print('epsilon:{}\tfile index:{}'.format(epsilons[i], file_index + 1))
            # read image
            origin_image_path = './dataset/SAN/%s.jpg' % (file_index + 1)
            origin_mask_path = './dataset/mask/%s_mask.png' % (file_index + 1)
            img, target = ad_ex.read_image(origin_image_path, origin_mask_path)

            # add adversarial example
            perturbed, result = ad_ex.test(model, device, img, target, epsilons[i],
                                           T=temperature, USE_FGSM=USE_FGSM_label)
            perturbed_data = perturbed.cpu().detach().numpy()
            perturbed_data = np.transpose(perturbed_data[0], (1, 2, 0))
            perturbed_image = Image.fromarray((perturbed_data * 255).astype(np.uint8))

            # save image
            perturbed_image_save_path = './output0/tmp/No.{}_{}_perturb_eps_{}.jpg'.format((file_index + 1), (i + 1),
                                                                                           epsilons[i])
            result_save_path = './output0/tmp/No.{}_{}_output_eps_{}.jpg'.format((file_index + 1), (i + 1), epsilons[i])
            perturbed_image.save(perturbed_image_save_path)
            result.save(result_save_path)

            # calculate psnr, ssim, f_score
            ps, ss = cal_ssim_psnr(origin_image_path, perturbed_image_save_path)
            PSNR_list[i].append(ps)
            SSIM_list[i].append(ss)
            f, _, _ = usage(origin_mask_path, result_save_path)
            f_score_list[i].append(f)
            if f < 0.5:
                count += 1

        PSNR_mean = np.mean(PSNR_list[i])
        SSIM_mean = np.mean(SSIM_list[i])
        f_score_mean = np.mean(f_score_list[i])

        PSNR_mean_list.append(PSNR_mean)
        SSIM_mean_list.append(SSIM_mean)
        f_score_mean_list.append(f_score_mean)
        ASR_list.append(count / image_nums)

    result_plot(epsilons, PSNR_mean_list, 'PSNR', 'average PSNR for eplisons')
    result_plot(epsilons, SSIM_mean_list, 'SSIM', 'average SSIM for eplisons')
    result_plot(epsilons, f_score_mean_list, 'f_score', 'average f_score for eplisons')
    result_plot(epsilons, ASR_list, 'ASR', 'average ASR for eplisons')
    print(PSNR_mean_list)
    print(SSIM_mean_list)
    print(f_score_mean_list)
    print(ASR_list)


adversarial_example_test(temperature=1, USE_FGSM_label=True)
