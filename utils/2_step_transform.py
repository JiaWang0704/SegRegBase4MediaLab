# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Email:  alisonzli@tencent.com or alisonbrielee@gmail.com
""" Applying euler and rigid transforms to landmarks and computing tre using train data """
# **********************************

from dataset import read_size
from utils.utils import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    image_path_table = np.load('data/all_pairs_table_addnote.npz')['image_path']
    tre_list0, tre_list1, tre_list2 = [], [], []
    csv_path = 'data/landmarks-using-euler-rigid/'
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    i = 0
    for image_path in image_path_table:
        Source_image, Target_image, label, _ = image_path

        # Load Landmark
        Source_point_path = Source_image[:-4] + '.csv'
        Target_point_path = Target_image[:-4] + '.csv'
        s_point = load_landmarks_csv('data/landmarks/' + Source_point_path) / 5
        Source_size = read_size(Source_image)
        norm_value = (Source_size[0] ** 2 + Source_size[1] ** 2) ** 0.5
        print(Source_image, Target_image)

        # Load Euler Transformation
        tranform1 = 'data/transforms/src_warp_reg_euler_seg_940/' + \
                   Source_image[:-4].replace("/", "--") + '--' + \
                   Target_image[:-4].replace("/", "--") + '--' + label + '.txt'
        tranform2 = 'data/transforms/src_warp_reg_rigid_seg_940/' + \
                    Source_image[:-4].replace("/", "--") + '--' + \
                    Target_image[:-4].replace("/", "--") + '--' + label + '.txt'

        transform_1 = sitk.ReadTransform(tranform1)
        transform_2 = sitk.ReadTransform(tranform2)

        # Warp Point
        transform_1.SetInverse()
        transform_2.SetInverse()

        src_points_euler, src_points_rigid = [], []
        for i in range(s_point.shape[0]):
            src_point = np.array((s_point[i, 0], s_point[i, 1]))
            src_point_euler = transform_point(transform_1, src_point)
            src_point_rigid = transform_point(transform_2, src_point_euler)
            src_points_euler.append(src_point_euler)
            src_points_rigid.append(src_point_rigid)

        euler_point = np.array(src_points_euler)
        rigid_point = np.array(src_points_rigid)
        if label == 'training':
            t_point = load_landmarks_csv('data/landmarks/' + Target_point_path) / 5
            _, dict_stat0 = compute_target_regist_error_statistic(t_point * 5, s_point * 5)
            _, dict_stat1 = compute_target_regist_error_statistic(t_point * 5, euler_point * 5)
            _, dict_stat2 = compute_target_regist_error_statistic(t_point * 5, rigid_point * 5)

            # Judge
            if dict_stat0['Mean'] < dict_stat2['Mean'] and dict_stat0['Mean'] < dict_stat1['Mean']:
                rigid_point = s_point
            if dict_stat1['Mean'] < dict_stat2['Mean'] and dict_stat1['Mean'] < dict_stat0['Mean']:
                rigid_point = euler_point
            _, dict_stat2 = compute_target_regist_error_statistic(t_point * 5, rigid_point * 5)

            # Print
            print('TRE:', dict_stat0['Mean'] / norm_value, dict_stat1['Mean'] / norm_value,
                  dict_stat2['Mean'] / norm_value)
            tre_list0.append(float(dict_stat0['Mean'] / norm_value))
            tre_list1.append(float(dict_stat1['Mean'] / norm_value))
            tre_list2.append(float(dict_stat2['Mean'] / norm_value))
            print('Done!')

        # Save
        path = Source_image[:-4].replace("/", "_") + Target_image[:-4].replace("/", "_") + '.csv'
        save_landmarks_csv(csv_path + path, rigid_point * 5)

        i += 1

    print('----------------TRE----------------')
    print(np.mean(tre_list0), np.std(tre_list0))
    print(np.mean(tre_list1), np.std(tre_list1))
    print(np.mean(tre_list2), np.std(tre_list2))
