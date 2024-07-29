# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Email:  alisonzli@tencent.com or alisonbrielee@gmail.com
""" Process images to 1/5-ratio """
# **********************************

import os
import cv2
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from utils.Convert_rgb_hist import image_histogram_matching


def normalize(img):
    img = img/255.0
    return img


def read_img2gray_pair_pad(source_image, target_image, ratio):
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
    image_path = []
    # i = 0
    for path in image_path_table:
        Source_image, Target_image, label, _ = path
        src_img, dst_img = read_img2gray_pair_pad(Source_image, Target_image, ratio=5)
        print(src_img.shape, dst_img.shape)

        new_save_path_m = 'data/image-processed/5-ratio/moving/'
        new_save_path_f = 'data/image-processed/5-ratio/fixed/'
        if not os.path.exists(new_save_path_m):
            os.mkdir(new_save_path_m)
        if not os.path.exists(new_save_path_f):
            os.mkdir(new_save_path_f)
        # Sou_path = new_save_path_m + str(i) + '_' + Source_image[:-4].replace('/', '_') + '_' + label + '.jpg'
        # Tar_path = new_save_path_f + str(i) + '_' + Target_image[:-4].replace('/', '_') + '_' + label + '.jpg'
        Sou_path = new_save_path_m + '_' + Source_image[:-4].replace('/', '_') + '_' + label + '.jpg'
        Tar_path = new_save_path_f + '_' + Target_image[:-4].replace('/', '_') + '_' + label + '.jpg'
        print(Sou_path, Tar_path)
        # i += 1

        cv2.imwrite(Sou_path, src_img)
        cv2.imwrite(Tar_path, dst_img)
        image_path.append((Sou_path, Tar_path))

    np.savez('data/all_pairs_table_addnote_processed_5-ratio.npz', image_path=image_path)
    print(len(image_path))




