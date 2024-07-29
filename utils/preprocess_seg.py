# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Email:  alisonzli@tencent.com or alisonbrielee@gmail.com
""" Process masks to 1/5-ratio """
# **********************************

import os
import cv2
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def normalize(img):
    img = img/255.0
    return img


def read_seg_pair(source_image, target_image, ratio):
    print(seg_path + source_image)
    source_img = cv2.imread('data/image/' + source_image, 0)
    target_img = cv2.imread('data/image/' + target_image, 0)

    source_seg = cv2.imread('data/image/' + source_image[:-4]+'_mask.png', 0)
    target_seg = cv2.imread('data/image/' + target_image[:-4]+'_mask.png', 0)

    source_seg = cv2.resize(source_seg, (int(source_img.shape[1] / ratio), int(source_img.shape[0] / ratio)),
                            interpolation=cv2.INTER_NEAREST)
    target_seg = cv2.resize(target_seg, (int(target_img.shape[1] / ratio), int(target_img.shape[0] / ratio)),
                            interpolation=cv2.INTER_NEAREST)

    l = max([source_seg.shape[0], source_seg.shape[1], target_seg.shape[0], target_seg.shape[1]])
    source_seg = cv2.copyMakeBorder(source_seg, 0, l - source_seg.shape[0], 0, l - source_seg.shape[1],
                                    cv2.BORDER_CONSTANT, value=int(source_seg[0][0]))
    target_seg = cv2.copyMakeBorder(target_seg, 0, l - target_seg.shape[0], 0, l - target_seg.shape[1],
                                    cv2.BORDER_CONSTANT, value=int(target_seg[0][0]))
    return source_seg, target_seg


if __name__ == '__main__':
    image_path_table = np.load('data/all_pairs_table_addnote.npz')['image_path']
    image_path = []
    # i = 0
    for path in image_path_table:
        Source_image, Target_image, label, _ = path
        src_img, dst_img = read_seg_pair(Source_image, Target_image, ratio=5)
        print(src_img.shape, dst_img.shape)

        new_save_path_m = 'data/image-processed/5-ratio/moving-seg/'
        new_save_path_f = 'data/image-processed/5-ratio/fixed-seg/'
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

    np.savez('data/all_pairs_table_addnote_processed_5-ratio_seg.npz', image_path=image_path)
    print(len(image_path))




