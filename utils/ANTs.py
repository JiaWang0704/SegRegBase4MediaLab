# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Email:  alisonzli@tencent.com or alisonbrielee@gmail.com
""" Registration with ANTs """
# **********************************

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import ants
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from utils.Convert_rgb_hist import image_histogram_matching

def read_img2gray_pair_pad(source_image, target_image, ratio=10):
    source_img = cv2.imread('data/image/' + source_image, 0)
    target_img = cv2.imread('data/image/' + target_image, 0)

    source_img = cv2.resize(source_img, (int(source_img.shape[1] / ratio), int(source_img.shape[0] / ratio)),
                            interpolation=cv2.INTER_LINEAR)
    target_img = cv2.resize(target_img, (int(target_img.shape[1] / ratio), int(target_img.shape[0] / ratio)),
                            interpolation=cv2.INTER_LINEAR)

    l = max([source_img.shape[0], source_img.shape[1], target_img.shape[0], target_img.shape[1]])
    source_img = cv2.copyMakeBorder(source_img, 0, l - source_img.shape[0], 0, l - source_img.shape[1],
                                    cv2.BORDER_CONSTANT, value=int(source_img[0][0]))
    target_img = cv2.copyMakeBorder(target_img, 0, l - target_img.shape[0], 0, l - target_img.shape[1],
                                    cv2.BORDER_CONSTANT, value=int(target_img[0][0]))
    source_img = image_histogram_matching(
        source_img, target_img, use_color='rgb', norm_img_size=4096)
    return source_img.astype(np.float64), target_img.astype(np.float64)


if __name__ == '__main__':
    image_path_table = np.load('data/all_pairs_table_addnote.npz')['image_path']
    print(len(image_path_table))
    train_data = image_path_table
    transforms, transforms_inv = [], []
    i = 0
    for image_path in train_data:
        Source_image, Target_image, label, _ = image_path
        save_path = str(i) + '_' + Source_image[:-4].replace("/", "_") + Target_image[:-4].replace("/", "_") + '_' + label
        i += 1

        viz_path = 'results/ANTs-viz-10ratio-TRSAA/'
        warped_path = 'results/ANTs-warped-10ratio-TRSAA/'
        trans_path = 'results/ANTs-transforms-10ratio-TRSAA/'
        if not os.path.exists(viz_path):
            os.mkdir(viz_path)
        if not os.path.exists(warped_path):
            os.mkdir(warped_path)
        if not os.path.exists(trans_path):
            os.mkdir(trans_path)

        input_moving, input_fixed = read_img2gray_pair_pad(Source_image, Target_image)
        fixed = ants.from_numpy(input_fixed)
        moving = ants.from_numpy(input_moving)
        mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='TRSAA',
                                 initial_transform="identity",
                                 aff_iterations=(2100, 2100, 2100, 1000, 1000),
                                 aff_shrink_factors=(1, 1, 1, 1, 1),
                                 aff_smoothing_sigmas=(0, 0, 0, 0, 0))

        # Save warped image
        warped_moving = mytx['warpedmovout'].numpy()
        cv2.imwrite(warped_path + save_path + '_warped.jpg', warped_moving)

        # Save transforms
        transforms.append(mytx['fwdtransforms'])
        transforms_inv.append(mytx['invtransforms'])
        np.savez(trans_path + save_path + '.npz', transforms=transforms, transforms_inv=transforms_inv)

        # Save compare image
        fig, axes = plt.subplots(1, 3, figsize=(6, 6))
        ax0, ax1, ax2 = axes.ravel()

        ax0.imshow(input_fixed)
        ax0.set_title("Target image")

        ax1.imshow(input_moving)
        ax1.set_title("Source image")

        ax2.imshow(warped_moving)
        ax2.set_title("Warped image")

        for ax in axes.ravel():
            ax.axis('off')
        fig.tight_layout()

        plt.savefig(viz_path + save_path + '.jpg')
        print('Done!')





