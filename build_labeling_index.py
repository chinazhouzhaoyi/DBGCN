"""
paper: DBGCN: Dual-branch Graph Convolutional Network for organ instance inference on sparsely labeled 3D plant data
file: build_labeling_index.py
about: Calculate the index of the training point set in the point cloud,
simulate the random labeling strategy and extreme labeling strategy of manual annotation
author: Zhaoyi Zhou
date: 2025-6-19

It is worth noting that the stem organ in the plant point cloud used in this article are all labeled '0'
"""

import random
import numpy as np
import torch
from utils import load_data, accuracy, encode_onehot
import os
from tqdm import tqdm
import glob
import argparse

def get_file_names(folder):
    '''
    Get an absolute path list of all files in the specified folder
    '''
    file_names = glob.glob(folder + '/*')
    return file_names

def ComPerCent(new_counts, nodesNum):

    leafnodesNum = nodesNum
    new_counts_array = new_counts
    ans = sum(new_counts_array)
    new_counts_array = np.array(new_counts_array)
    leafPer = (leafnodesNum * new_counts_array / ans)

    for i in range(len(leafPer)):
        leafPer[i] = (round(leafPer[i]))
        if leafPer[i] < 1:
            leafPer[i] = 1
    sum_leafNode = sum(leafPer)
    max_index = np.argmax(leafPer)
    max_number = np.max(leafPer)

    if sum_leafNode < leafnodesNum:
        leafPer[max_index] = leafPer[max_index] + leafnodesNum - sum_leafNode
    else:
        leafPer[max_index] = leafPer[max_index] + (leafnodesNum - sum_leafNode)
    leafPer = leafPer.astype(int)
    return leafPer

def auto_RSassign(labels, trainNum, saven):
    unique, counts = torch.unique(labels, return_counts=True)
    idx_train=[]
    class_num = []
    for i in range(unique.numel()):
        class_num.append(counts[i].cpu().detach().numpy())
    class_rs_num_arr = ComPerCent(class_num, trainNum)

    #Random labeling strategy: allocate sampling points of each category according to the proportion of organ points,
    # conduct random sampling in each organ as the training point set
    for i in range(unique.numel()):
        class_label_idx = torch.where(labels == unique[i])[0]
        class_label_idx = class_label_idx.cpu().detach().numpy()

        rsNum = class_rs_num_arr[i]
        rs_sample_idx = random.sample(list(class_label_idx), rsNum)
        idx_train.extend(rs_sample_idx)
    print("idx_train:", idx_train)
    np.savetxt(saven, idx_train, fmt="%d")
    return idx_train

def auto_extreme_assign(points, labels, saven):
    unique, counts = torch.unique(labels, return_counts=True)
    idx_train=[]

    #Labeling for extreme strategy: take the first five points for the stem,
    # and the point closest to the median for the blade as the training point set
    for i in range(unique.numel()):
        class_label_idx = torch.where(labels == unique[i])[0]
        class_xyz=points[class_label_idx]
        if unique[i]==0:
            idx_train.extend(class_label_idx[:5])
        else:
            median_x=class_xyz[:,0].median()
            median_y = class_xyz[:, 1].median()
            median_z = class_xyz[:, 2].median()
            median_point=[median_x,median_y,median_z]
            median_point=np.array(median_point)
            # print("median_point:",median_point)
            minID = np.argmin(np.sum((class_xyz.data.cpu().numpy()- median_point) ** 2, axis=1))

            leaf_point = class_xyz[minID, :] # Take the point closest to the median point as the extreme labeled point of the leaf organ
            leaf_point_idx=[torch.where(points == leaf_point)[0][0]]
            idx_train.extend(leaf_point_idx)
    print("idx_train:", idx_train)
    np.savetxt(saven, idx_train, fmt="%d")

    return idx_train

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--Kum', type=int, default=20, help='neighbor size of KNN')
    parser.add_argument('--source_data', type=str, default='./source_data/soybean',
                        help='specify your point cloud path')
    parser.add_argument('--is_RS', type=bool, default=True,
                        help='specify your labeling strategy')
    parser.add_argument('--trainidx_store_fold', type=str, default='./examples/RS_train_idx',
                        help='specify your train index saved path')
    parser.add_argument('--trainNum', type=int, default='200',
                        help='specify your training set size')

    return parser.parse_args()

args = parse_args()
Knum = args.Kum
if args.is_RS:
    savedPath = os.path.join(args.trainidx_store_fold, 'trainNum' + str(args.trainNum))
else:
    savedPath = args.trainidx_store_fold
    os.makedirs(savedPath, exist_ok=True)

datapath = get_file_names(args.source_data)
print('The size of %s data is %d' % ("dataset", len(datapath)))

trainNum = args.trainNum
for index in tqdm(range(len(datapath)), total=len(datapath)):
    # Load data

    plant_name = datapath[index].split('/')[-1].split('.')[0] + '.txt'
    print("plant_name:", plant_name)
    P = np.loadtxt(datapath[index]).astype(np.float32)
    points = P[:, :3]
    labels = P[:, 3].astype(np.int)
    points = torch.LongTensor(points)
    labels = torch.LongTensor(labels)

    if args.is_RS:
        RS_times = 5
        for rr in range(RS_times):
            idx_folder = "idx" + str(rr + 1)
            # print(idx_folder)
            os.makedirs(os.path.join(savedPath, idx_folder), exist_ok=True)
            save = os.path.join(savedPath, idx_folder, plant_name)
            print("save path:", save)
            idx_train = auto_RSassign(labels, trainNum, save)
    else:
        save = os.path.join(savedPath, plant_name)

        idx_train = auto_extreme_assign(points, labels, save)
        print("save path:", save)

    print("get training set size:", len(idx_train))



