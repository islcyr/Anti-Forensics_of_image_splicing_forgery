# -*- coding: utf-8 -*-
"""
@Time ： 2023/2/27 20:33
@Auth ： Yin yanquan
@File ：ad_ex.py
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
# import matplotlib.pyplot as plt

from eval import eval_net
from unet.unet_model import *
from utils import *
from attack import *

os.environ["PYPYTORCH_CUDA_ALLOC_CONF"] = f'max_split_size_mb:{64}'



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Ringed_Res_Unet(n_channels=3, n_classes=1)
model = net.to(device)
pretrain_model = torch.load(r'./best_model.pth')
model.load_state_dict(pretrain_model)
model.eval()

epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def read_image(img_path, img_gt_path):
    img_fn = img_path
    img = Image.open(img_fn)
    img = resize_and_crop(img, scale=1).astype(np.float32)
    img = np.transpose(normalize(img), (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(dim=0)

    img_output = img_gt_path
    img_out = Image.open(img_output)
    # imgs_out = np.array(img_out)
    # imgs_out = imgs_out.astype(np.float32)/255
    imgs_out = resize_and_crop(img_out, scale=1).astype(np.float32) / 255
    # imgs_out = np.transpose(imgs_out, axes=[2, 0, 1])
    target = torch.from_numpy(imgs_out).unsqueeze(dim=0)

    return img, target


def test(model, device, data, target, epsilon, T=2, USE_FGSM=True):
    data = data.to(device)
    target = target.to(device)
    # data.requires_grad = True
    # output = model(data)
    #
    # outputs = torch.sigmoid(output)
    # outputs_flat = outputs.view(-1)

    if USE_FGSM:

        data.requires_grad = True
        output = model(data)

        outputs = torch.sigmoid(output / T)

        outputs_flat = outputs.view(-1)

        target_flat = target.view(-1)

        criterion = nn.BCELoss()
        loss = criterion(outputs_flat, target_flat).to(device)
        model.zero_grad()
        loss.backward()

        data_gard = data.grad.data
        # print(data_gard.sign().shape)
        # np.set_printoptions(suppress=True)
        # np.set_printoptions(threshold=np.inf)
        # with open('temp_out.txt', 'w') as f:
        #     f.write(str(data_gard.cpu().detach().numpy()))
        # with open('temp_outs.txt', 'w') as f:
        #     f.write(str(outputs.cpu().detach().numpy()))
        perturbed_data = fgsm_attack(data, epsilon, data_gard)
        # perturbed_data = fgsm_attack_restricted(data, target, epsilon, data_gard)

        output_fgsm_ = model(perturbed_data.to(device))
        output_fgsm_mask = torch.sigmoid(output_fgsm_).squeeze().cpu().detach().numpy()
        result_fgsm = Image.fromarray((output_fgsm_mask * 255).astype(np.uint8))
        return perturbed_data, result_fgsm
        # result_fgsm.save('fgsm_output.jpg')
    else:
        alpha = 2 / 255
        iters = 40

        ori_images = data.data

        data_clone = data
        rand_image = torch.rand(data_clone.size(), dtype=data_clone.dtype, device=data_clone.device)
        data_clone = data_clone + (rand_image - 0.5) * 2 * alpha
        data = torch.clamp(data_clone, 0, 1)

        # data_clone.requires_grad = True
        # output = model(data_clone)
        # outputs = torch.sigmoid(output)
        # outputs_flat = outputs.view(-1)
        target_flat = target.view(-1)

        criterion = nn.BCELoss()

        perturbed_data = result_pgd = 0

        for i in range(iters):
            data.requires_grad = True
            output = model(data)
            outputs = torch.sigmoid(output / T)
            outputs_flat = outputs.view(-1)

            model.zero_grad()
            cost = criterion(outputs_flat, target_flat).to(device)
            cost.backward(retain_graph=True)

            data_grad = data.grad.data
            perturbed_data = pgd_attack(data, ori_images, data_grad, eps=epsilon, alpha=alpha)
            # perturbed_data = pgd_attack_restricted(data, ori_images, target, data_grad, eps=epsilon, alpha=alpha)

            output_pgd_ = model(perturbed_data.to(device))
            output_pgd_mask = torch.sigmoid(output_pgd_).squeeze().cpu().detach().numpy()
            result_pgd = Image.fromarray((output_pgd_mask * 255).astype(np.uint8))

        return perturbed_data, result_pgd


def save_image():
    torch.cuda.empty_cache()
    USE_FGSM_label = True
    for i in range(len(epsilons)):
        img, target = read_image('./example.jpg', './example_gt.png')
        perturbed, result = test(model, device, img, target, epsilons[i], T=2, USE_FGSM=USE_FGSM_label)
        perturbed_data = perturbed.cpu().detach().numpy()
        # print(perturbed_data[0].shape)
        perturbed_data = np.transpose(perturbed_data[0], (1, 2, 0))
        perturbed_image = Image.fromarray((perturbed_data * 255).astype(np.uint8))
        num = i + 1
        if USE_FGSM_label:
            perturbed_image.save('./output/fgsm/%%s_perturbed_output_eps_%s.jpg' % epsilons[i] % num)
            result.save('./output/fgsm/%%s_fgsm_output_eps_%s.jpg' % epsilons[i] % num)
        else:
            print(num)
            perturbed_image.save('./output/pgd/%%s_perturbed_output_eps_%s.jpg' % epsilons[i] % num)
            result.save('./output/pgd/%%s_pgd_output_eps_%s.jpg' % epsilons[i] % num)


if __name__ == '__main__':
    save_image()
