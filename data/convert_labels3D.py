import numpy as np
import nibabel as nib
import os
import glob
import cv2

# labels = [0, 42, 41, 47, 46, 53, 63, 44, 43, 77, 51, 54, 62, 60,
#               49, 52, 50, 16, 58, 3, 24, 85, 15, 254, 4, 253, 252, 251,
#               255, 2, 14, 10, 8, 31, 28, 7, 11, 26, 13, 12, 17, 18, 5, 30, 80] # 45
labels = [0, 1, 11, 22, 33, 44, 46, 47, 48, 49, 50, 51, 52, 55] # 14 PPMI ABIDE 
# good_label = [0, 2, 3, 4, 41, 10, 42, 43, 77, 16, 17, 49, 53, 31] # 14 OASIS
            # 0  1   2   3   4   5   6   7  8    9  10  11  12  13
good_label = [0, 1, 11, 22, 33, 44, 46, 47, 48, 49, 50, 51, 52, 55] # 14 PPMI ABIDE 

# good_label = [0, 50] # 14 PPMI ABIDE 

# labels = [ 0.,  2.,  3.,  4.,  5.,  7.,  8., 10., 11., 12., 13., 14., 15., 16., 17., 18., 26., 28.,
#  41., 42., 43., 44., 46., 47., 49., 50., 51., 52., 53., 54., 58., 60.] # 14 SynthSeg 
# good_label = [ 0.,  2.,  3.,  4.,  5.,  7.,  8., 10., 11., 12., 13., 14., 15., 16., 17., 18., 26., 28.,
#  41., 42., 43., 44., 46., 47., 49., 50., 51., 52., 53., 54., 58., 60.]

extra_label = list(set(labels).difference(set(good_label)))
index = list(range(len(good_label)))


def convert(seg):
    output = np.copy(seg)
    for i in extra_label:
        output[seg == i] = 0
    # print(np.unique(output))
    for k, v in zip(good_label, index):
        output[seg == k ] = v
    # print(np.unique(output))
    return output

def inverse_convert(seg):
    output = np.copy(seg)
    # for i in extra_label:
    #     output[seg == i] = 0
    for k, v in zip(index, good_label):
        output[seg == k] = v
    return output
    
# nii = nib.load('data/0139_MR1-3_seg.nii.gz').get_data() # 44 labels
# label = []
# nii = convert(nii)
# for i in range(nii.shape[0]):
#     for j in range(nii.shape[1]):
#         for k in range(nii.shape[2]):
#             if nii[i][j][k] not in label:
#                 label.append(nii[i][j][k])
# print(label)
# print(len(label))

#
# nii = nib.load('data/atlas_seg.nii.gz').get_data() # 39 labels
# label = []
# print(nii.shape)
# for i in range(nii.shape[0]):
#     for j in range(nii.shape[1]):
#         for k in range(nii.shape[2]):
#             if nii[i][j][k] not in label:
#                 label.append(nii[i][j][k])
# print(label)
# print(len(label))

# image_path_table = np.load('data/Seg_train_data.npz')['arr_0']
# print(len(image_path_table))
# for (img_path, seg_path) in image_path_table:
#     label = []
#     img = cv2.imread(seg_path, 0)
#     img = convert(img)
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if img[i][j] not in label:
#                 label.append(img[i][j])
#     if len(label) != 14:
#         print(seg_path, len(label))
#         print(label)

# res = labels
# for (img_path, seg_path) in image_path_table:
#     label = []
#     img = cv2.imread(seg_path, 0)
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if img[i][j] not in label:
#                 label.append(img[i][j])
#     res = list(set(res).intersection(set(label)))
# print(len(image_path_table))
# print(res)
# print(len(res))


