# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Email:  alisonzli@tencent.com or alisonbrielee@gmail.com
""" Applying euler and rigid transforms to images/masks and show  """
# **********************************

import matplotlib.pyplot as plt
from utils.utils import *
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    image_path_table = np.load('data/all_pairs_table_addnote.npz')['image_path']
    viz_path = 'results/images-using-euler-rigid/'
    if not os.path.exists(viz_path):
        os.mkdir(viz_path)
    # i = 0

    for image_path in image_path_table:
        Source_image, Target_image, label, _ = image_path
        new_save_path_m = 'data/image-processed/5-ratio/moving/'
        new_save_path_f = 'data/image-processed/5-ratio/fixed/'
        new_save_path_w = 'data/image-processed/5-ratio/moved/'

        # Sou_path = new_save_path_m + str(i) + '_' + Source_image[:-4].replace('/', '_') + '_' + label + '.jpg'
        # Tar_path = new_save_path_f + str(i) + '_' + Target_image[:-4].replace('/', '_') + '_' + label + '.jpg'
        Sou_path = new_save_path_m + '_' + Source_image[:-4].replace('/', '_') + '_' + label + '.jpg'
        Tar_path = new_save_path_f + '_' + Target_image[:-4].replace('/', '_') + '_' + label + '.jpg'
        print(Sou_path)
        src_img = sitk.ReadImage(Sou_path, sitk.sitkFloat32)
        tar_img = sitk.ReadImage(Tar_path, sitk.sitkFloat32)

        # Load Euler Transformation
        tranform1 = 'data/Archive/src_warp_reg_euler_seg_940/' + \
                   Source_image[:-4].replace("/", "--") + '--' + \
                   Target_image[:-4].replace("/", "--") + '--' + label + '.txt'
        tranform2 = 'data/Archive/src_warp_reg_rigid_seg_940/' + \
                    Source_image[:-4].replace("/", "--") + '--' + \
                    Target_image[:-4].replace("/", "--") + '--' + label + '.txt'
        transform_1 = sitk.ReadTransform(tranform1)
        transform_2 = sitk.ReadTransform(tranform2)

        # Warp Image
        warp_euler = warp_img(src_img, tar_img, transform_1)
        warp_rigid = warp_img(warp_euler, tar_img, transform_2)
        # warp_euler = warp_seg(src_img, tar_img, transform_1)
        # warp_rigid = warp_seg(warp_euler, tar_img, transform_2)


        # Save and Show
        # w_path = new_save_path_w + str(i) + '_' + Source_image[:-4].replace('/', '_') + '_' + label + '_moved.jpg'
        w_path = new_save_path_w + '_' + Source_image[:-4].replace('/', '_') + '_' + label + '_moved.jpg'
        cv2.imwrite(w_path, sitk.GetArrayViewFromImage(warp_rigid))

        fig, axes = plt.subplots(1, 3, figsize=(6, 6))
        ax0, ax1, ax2 = axes.ravel()
        ax0.imshow(sitk.GetArrayViewFromImage(tar_img))
        ax0.set_title("Target")
        ax1.imshow(sitk.GetArrayViewFromImage(src_img))
        ax1.set_title("Source")
        ax2.imshow(sitk.GetArrayViewFromImage(warp_rigid))
        ax2.set_title("Affine")
        for ax in axes.ravel():
            ax.axis('off')
        fig.tight_layout()
        save_path = Source_image[:-4].replace("/", "_") + Target_image[:-4].replace("/", "_") + label + '.png'
        plt.savefig(viz_path + save_path)
        print('Done!')

        # i += 1






