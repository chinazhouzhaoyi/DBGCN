"""
paper: DBGCN: Dual-branch Graph Convolutional Network for organ instance inference on sparsely labeled 3D plant data
file:  build_graph_representation.py
about: The labeled point cloud data is processed as the input of GCN network, and the adjacency matrix and graph feature CSV file are output.
     The first three dimensions of the input labeled point cloud file are XYZ coordinates, and the fourth dimension is the label.
author: Zhaoyi Zhou
date: 2025-6-19
"""

import numpy as np
from sklearn.neighbors import kneighbors_graph
import pandas as pd
import argparse
import os
from tqdm import tqdm
import glob
def get_file_names(folder):
    '''
    Get an absolute path list of all files in the specified folder
    '''
    file_names = glob.glob(folder + '/*')
    return file_names
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_data', type=str, default='./source_data/soybean',
                        help='specify your source data path')
    parser.add_argument('--Kum', type=int, default=20, help='neighbor size of KNN')
    parser.add_argument('--adj_store_fold', type=str, default='./examples/adj_folder_k',
                        help='specify your adj_folder path')
    parser.add_argument('--content_store_fold', type=str, default='./examples/content_folder_k',
                        help='specify your content_folder path')

    return parser.parse_args()

args = parse_args()
adjoutput_dir = args.adj_store_fold + str(args.Kum)
contentoutput_dir = args.content_store_fold + str(args.Kum)
datapath = args.source_data
Knum = args.Kum

os.makedirs(adjoutput_dir, exist_ok=True)
os.makedirs(contentoutput_dir, exist_ok=True)

plantpath = get_file_names(args.source_data)
print('The size of %s data is %d' % ("dataset", len(plantpath)))

for index in tqdm(range(len(plantpath)), total=len(plantpath)):
        fn = plantpath[index]
        plant_name = fn.split('/')[-1].split('.')[0]
        print(fn, plant_name)
        adjn = os.path.join(adjoutput_dir, plant_name)
        contentn = os.path.join(contentoutput_dir, plant_name)
        f = open(fn, 'r')

        data = f.readlines()
        f.close()
        arr = []
        label = []
        number = []

        n = 0
        for line in data:
            line = line.strip('\n')
            splitData = line.split()
            l = [eval(i) for i in splitData[3:4]]
            x, y, z = [eval(j) for j in splitData[:3]]
            arr.append([x, y, z])
            label.append(l)
            number.append(n)
            n += 1
        point_size = n
        print("the points number:", point_size)

        newLabel = [int(x) for item in label for x in item]
        arrMatrix = np.mat(arr)
        data_xyz_txt = arrMatrix
        print("data_xyz_txt.shape:", data_xyz_txt.shape)
        data_xyzDF = pd.DataFrame(data_xyz_txt, dtype=float)
        A = kneighbors_graph(arr, Knum, mode='distance', include_self=False) #Use the nearest distance to establish the adjacency matrix
        C = A.toarray()
        graph_test = []

        for i in range(point_size):
            for j, k in zip(A.getrow(i).indices, A.getrow(i).data):
                initial = i
                end = j
                weight = k
                graph_test.append([initial, end, weight])

        #-----Adjacency matrix storage-----
        f2 = open(adjn + ".txt", 'w+')
        for knnnum in graph_test:
            print(knnnum[0], '=', knnnum[1], end='\n', file=f2, sep='')
        f2.close()
        print("Adjacency matrix storage completed")

        # #-----do graph feature CSV file processing---------
        weight_arr = []
        for i in C:
            for j in i:
                if(j != 0):
                    weight_arr.append(j)
        K = Knum
        #Calculate the bandwidth of Gaussian kernel function
        h = sum(weight_arr) / (point_size * K)
        # -------------------------------------------------------------------------------
        row_weight = 0
        weight = 0

        # Calculate similarity matrix
        for i in range(point_size):
            for j, k in zip(A.getrow(i).indices, A.getrow(i).data):
                initial = i
                end = j
                weight = np.exp(-k / (2 * h))
                row_weight += weight

            for j, k in zip(A.getrow(i).indices, A.getrow(i).data):
                initial = i
                end = j
                C[i][j] = np.exp(-k / (2 * h)) / row_weight
            row_weight = 0

        print(np.sum(C, axis=1)) #Verify the normalized calculation results of row direction

        data_txtDF = pd.DataFrame(C, dtype=float)
        #
        data_txtDF = pd.concat([data_txtDF,data_xyzDF], axis=1) # [N, N+3] Concate similarity distance and XYZ coordinates

        # Add serial number and label
        data_txtDF.insert(loc=0, column="number", value=number)
        data_txtDF.insert(loc=point_size + 4, column="label", value=newLabel)
        data_txtDF.to_csv(contentn + ".csv", index=False, header=None, sep=' ')
        print("Graph feature CSV file completed")

