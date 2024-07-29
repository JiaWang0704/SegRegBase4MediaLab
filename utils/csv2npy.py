# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Email:  alisonzli@tencent.com or alisonbrielee@gmail.com
""" From .csv to .npz files """
# **********************************

from utils.utils import *
import csv

# ******************** from dataset_medium.csv to .npz files **************

with open('data/dataset_medium_annotated.csv', 'r') as myFile:
    lines = csv.reader(myFile)
    Id, Image_diagonal, Image_size = [], [], []
    image_path, landmarks_path = [], []
    for line in lines:
        if line[7] == 'evaluation' or line[7] == 'training':
            # Image_diagonal.append(line[1])
            # Image_size.append(line[2])
            image_path.append((line[3], line[5], line[7], line[11]))

    np.savez('data/all_pairs_table_addnote.npz', image_path=image_path)
    print('everything finished!')
    print(len(image_path))
