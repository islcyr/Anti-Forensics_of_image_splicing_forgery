# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/5 16:41
@Auth ： Yin yanquan
@File ：criterion.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""

import numpy as np
import cv2
from PIL import Image
from PIL import ImageChops


def cal_IOU(image1, image2):
    """
    计算重叠度
    :param image1: 图片1
    :param image2: 图片2
    :return: None（TP）
    """
    gt_mask = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    out_mask = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)
    size = gt_mask.shape
    w = size[1]  # 宽度
    h = size[0]  # 高度
    IOU_mask = np.zeros((h, w), dtype=int)
    # print(size,w,h)
    for i in range(h):
        for j in range(w):
            if int(gt_mask[i][j]) > 240 and int(out_mask[i][j]) > 240:
                IOU_mask[i][j] = 255
    cv2.imwrite('./IOU.jpg', IOU_mask)
    return IOU_mask


def compare_images(image1, image2, save_path):
    """
    比较图片，如果有不同则生成展示不同的图片
    :param image1: 第一张图片的路径
    :param image2: 第二张图片的路径
    :param save_path: 不同图的保存路径
    :return: None （FP+FN）
    """
    image_one = Image.open(image1)
    image_two = Image.open(image2)
    try:
        diff = ImageChops.difference(image_one, image_two)

        if diff.getbbox() is None:
            print("No difference between the two pictures")
        else:
            diff.save(save_path)
    except ValueError as e:
        text = ("表示图片大小和box对应的宽度不一致，参考API说明：Pastes another image into this image."
                "The box argument is either a 2-tuple giving the upper left corner, a 4-tuple defining the left, upper, "
                "right, and lower pixel coordinate, or None (same as (0, 0)). If a 4-tuple is given, the size of the pasted "
                "image must match the size of the region.使用2纬的box避免上述问题")
        print("【{0}】{1}".format(e, text))


def cal_criterion(ground_truth_mask, output_mask):
    """
    计算对抗样本输入模型后的mask与ground truth的比值
    :param ground_truth_mask: 原图片对应的gt_mask
    :param output_mask: 对抗样本输入模型后的mask与gt_mask的差值
    :return: 两者的比率， output_mask的非黑像素数量，gt_mask的非黑像素数量，攻击是否成功
    """

    gt_mask = cv2.imread(ground_truth_mask, cv2.IMREAD_GRAYSCALE)
    gt_pixels = cv2.countNonZero(gt_mask)

    out_mask = cv2.imread(output_mask, cv2.IMREAD_GRAYSCALE)
    out_pixels = cv2.countNonZero(out_mask)

    attack_rate = out_pixels / gt_pixels

    return attack_rate, out_pixels, gt_pixels


def precision(public_mask, output_mask):
    """
    计算准确率
    :param public_mask: 图片公共部分
    :param output_mask: 模型输出图片
    :return: 准确率
    """
    # p_mask = cv2.imread(public_mask, cv2.IMREAD_GRAYSCALE)
    # p_pixels = cv2.countNonZero(p_mask)
    p_pixels = cv2.countNonZero(public_mask)

    out_mask = cv2.imread(output_mask, cv2.IMREAD_GRAYSCALE)
    out_pixels = cv2.countNonZero(out_mask)

    pre_rate = p_pixels / out_pixels
    return pre_rate


def recall(public_mask, ground_truth_mask):
    """
    计算召回率
    :param public_mask: 图片公共部分
    :param ground_truth_mask: 原图片对应的gt_mask
    :return: 召回率
    """
    # p_mask = cv2.imread(public_mask, cv2.IMREAD_GRAYSCALE)
    # p_pixels = cv2.countNonZero(p_mask)
    p_pixels = cv2.countNonZero(public_mask)

    gt_mask = cv2.imread(ground_truth_mask, cv2.IMREAD_GRAYSCALE)
    gt_pixels = cv2.countNonZero(gt_mask)

    re_rate = p_pixels / gt_pixels
    return re_rate


def f_score(precision_rate, recall_rate, beta=2):
    """
    计算F分数
    :param precision_rate: 准确率
    :param recall_rate: 召回率
    :param beta: 计算权重
    :return: F分数
    """
    numerator = precision_rate * recall_rate
    denominator = ((beta ** 2) * precision_rate) + recall_rate
    ratio = 1 + (beta ** 2)
    if denominator < 0.001:
        score = 0
    else:
        score = ratio * numerator / denominator
    return score


def usage(ground_truth_mask, output_mask, beta=2):
    """
    具体图片计算方法
    :param ground_truth_mask: 原图片对应的gt_mask
    :param output_mask: 模型输出图片
    :param beta: 计算权重
    :return: F分数，准确率，召回率
    """
    public_mask = cal_IOU(ground_truth_mask, output_mask)
    pre_rate = precision(public_mask, output_mask)
    re_rate = recall(public_mask, ground_truth_mask)
    # print(pre_rate, re_rate)
    F_score = f_score(pre_rate, re_rate, beta=beta)
    # print(F_score)
    return F_score, pre_rate, re_rate


if __name__ == '__main__':
    img1 = '../example_gt.png'
    img2 = '../output/fgsm/3_fgsm_output_eps_0.1.jpg'
    # cal_IOU(img1, img2)
    # pre_rate = precision(img3, img2)
    # re_rate = recall(img3, img1)
    # print(pre_rate, re_rate)
    # F_score = f_score(pre_rate, re_rate)
    # print(F_score)
    F_score, _, _ = usage(img1, img2, beta=2)
    print(F_score)
