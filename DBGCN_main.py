"""
paper: DBGCN: Dual-branch Graph Convolutional Network for organ instance inference on sparsely labeled 3D plant data
file:  DBGCN_main.py
about: perform inference and training on crop point clouds with sparsely annotated organ labels,
 automatically propagating the labels to all points in the point cloud, and conduct label propagation accuracy evaluation
author: Zhaoyi Zhou
date: 2025-6-19
"""

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from utils import load_data, accuracy
from models import DBGCN
import os
from tqdm import tqdm
import glob

def get_file_names(folder):
    '''
    Get an absolute path list of all files in the specified folder
    '''
    file_names = glob.glob(folder + '/*')
    return file_names
def train(epoch,idx_train,saven,max_epoch,best_epoch,best_acc):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, cross_out = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train, pred_train= accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test, _ = accuracy(output[idx_test], labels[idx_test])
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_test: {:.4f}'.format(loss_test.item()),
    #       'acc_test: {:.4f}'.format(acc_test.item()),
    #       'time: {:.4f}s'.format(time.time() - t))
    if epoch > max_epoch - 20:
        if acc_train >= best_acc:
            best_acc = acc_train
            best_epoch = epoch
            torch.save(model.state_dict(), saven+'best_model.pth')

    return pred_train, acc_train, best_epoch,best_acc

def test(saven,all_acc):
    model.load_state_dict(torch.load(saven+"best_model.pth",
                                     map_location=torch.device('cpu')))
    model.cuda()
    model.eval()
    output, cross_out= model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test, preds_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    all_acc = all_acc+acc_test

    # save testing set results
    xyz = features[idx_test,-3:].data.cpu().numpy()
    pred_label = preds_test.data.cpu().numpy()
    gt_label = labels[idx_test].data.cpu().numpy()
    outputtxt = np.ones((output[idx_test].shape[0],5))
    outputtxt[:,:3] = xyz
    outputtxt[:, 3:4] = pred_label.reshape(output[idx_test].shape[0],1)
    outputtxt[:, 4:5] = gt_label.reshape(output[idx_test].shape[0],1)
    filename = saven + "test.txt"
    np.savetxt(filename, outputtxt, fmt="%5f %5f %5f %d %d")

    # save training set results
    xyz = features[idx_train,-3:].data.cpu().numpy()
    _, preds_train = accuracy(output[idx_train], labels[idx_train])
    pred_label = preds_train.data.cpu().numpy()
    gt_label = labels[idx_train].data.cpu().numpy()
    outputtxt = np.ones((output[idx_train].shape[0], 5))
    outputtxt[:,:3] = xyz
    outputtxt[:, 3:4] = pred_label.reshape(output[idx_train].shape[0],1)
    outputtxt[:, 4:5] = gt_label.reshape(output[idx_train].shape[0],1)
    filename = saven + "train.txt"
    np.savetxt(filename, outputtxt, fmt="%5f %5f %5f %d %d")

    return all_acc, acc_test

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005, #learning rate
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default= 128, #number of hidden feature
                    help='Number of hidden units.')
parser.add_argument('--trainNum', type=int, default=200, metavar='N',
                    help='RS init idx')
parser.add_argument('--idx', type=int, default=3, metavar='N',
                    help='RS init idx')
parser.add_argument('--ks', type=int, default=20, metavar='N',
                    help='Num of static_k')
parser.add_argument('--kd', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--layer', type=int, default=2, help='Number of layers by 2 or 4.') #the depth of DBGCN
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--adj_folder', type=str, default='./examples/adj_folder_k', help='the storage folder of adjacency matrixs')
parser.add_argument('--content_folder', type=str, default='./examples/content_folder_k', help='the storage folder of similarity matrixs')
parser.add_argument('--train_idx_folder', type=str, default='./examples/RS_train_idx', help='the storage folder of model results')
parser.add_argument('--result_folder', type=str, default='./examples/results', help='the storage folder of model results')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
# ---random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if __name__ == '__main__':
    adjstore = os.path.join(args.adj_folder) + str(args.ks)
    contentstore = os.path.join(args.content_folder) + str(args.ks)
    foldname = "idx" + str(args.idx) + "ks" + str(args.ks) + 'kd' + str(args.kd)
    savedPath = os.path.join(args.result_folder + "/layers" + str(args.layer) + "/trainNum" + str(args.trainNum), foldname)
    train_idxPath = os.path.join(args.train_idx_folder, "trainNum" + str(args.trainNum) + '/idx' + str(args.idx))
    os.makedirs(savedPath, exist_ok=True)

    store_detail_path = os.path.join(savedPath, "idx" + str(args.idx) + 'DBGCN_detail.txt')
    store_acc_path = os.path.join(savedPath, "idx" + str(args.idx) + 'DBGCN_acc.txt')
    mean_acc_path = os.path.join(savedPath, "idx" + str(args.idx) + 'DBGCN_Macc.txt')
    time_txt_path = os.path.join(savedPath, "idx" + str(args.idx) + 'time_DBGCN.txt')
    mtime_txt_path = os.path.join(savedPath, "idx" + str(args.idx) + 'Mtime_DBGCN.txt')
    os.makedirs(savedPath, exist_ok=True)

    adjpath = sorted(get_file_names(adjstore))
    # print(adjpath)
    print('The size of %s data is %d' % ("dataset", len(adjpath)))
    contentpath = sorted(get_file_names(contentstore))

    #batch train and test
    all_acc=0
    f_detail = open(store_detail_path, 'w')
    f_acc = open(store_acc_path, 'w')
    f_macc = open(mean_acc_path, 'w')
    f_time = open(time_txt_path, 'w')
    f_mtime = open(mtime_txt_path , 'w')
    time_all = 0
    for index in tqdm(range(len(adjpath)), total=len(adjpath)):
        # Load data
        adjn = adjpath[index]
        contentn = contentpath[index]
        adjname = adjn.split('/')[-1].split('.')[0]
        contentname = contentn.split('/')[-1].split('.')[0]
        if adjname != contentname:
            print("File reading order does not correspond！！！")
            break
        plantname = adjname + '.txt'
        saven = os.path.join(savedPath, plantname)
        #-----
        train_idx_txt = os.path.join(train_idxPath, plantname)
        print("saven:", saven)

        adj, features, labels = load_data(contentn,adjn)
        labels = labels - int(labels.min())
        print("classes:", int(labels.max()) + 1)
        # points = features[:, -3:]
        idx_train = np.loadtxt(train_idx_txt)

        print("the traing set number:", len(idx_train))
        idx_all = [i for i in range(features.shape[0])]
        idx_test = [x for x in idx_all if x not in idx_train]
        idx_train = torch.LongTensor(idx_train)
        idx_test = torch.LongTensor(idx_test)
        model = DBGCN(
                        nfeat=features.shape[1],
                        nlayers=args.layer,
                        nhidden=args.hidden,
                        nclass=int(labels.max()) + 1,
                        dropout=args.dropout,
                        knum=args.kd,
                        lamda = args.lamda,
                        alpha=args.alpha,
                        variant=args.variant)

        optimizer = optim.Adam([
            {'params': model.params2, 'weight_decay': args.wd2},  # fc params
            {'params': model.params1, 'weight_decay': args.wd1},  # SGCB params
            {'params': model.params_dynamic, "lr": args.lr * 0.1},  # DGCB params
            {'params': model.params_norm, "lr": args.lr * 0.1},  # align layerNorm params
            # {'params': model.params_last_fc, 'weight_decay': args.wd2,"lr": args.lr * 0.1},  # last fc params
                                ],lr=args.lr)
        if args.cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_test = idx_test.cuda()

        # Train model
        t_total = time.time()
        best_acc = 0
        best_epoch = 0
        max_epoch = args.epochs

        for epoch in range(args.epochs):
            _, _, best_epoch, best_acc = train(epoch, idx_train, saven, max_epoch=max_epoch, best_epoch=best_epoch, best_acc=best_acc)
        print("Optimization Finished!")
        time_o = time.time() - t_total
        time_all += time_o
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print("best_epoch:",best_epoch)

        #evaluation
        model.eval()
        all_acc, acc = test(saven,all_acc)
        print("all_acc:", all_acc)
        f_detail.write(str(index) + ' ' + str(plantname) + ' ' +
                str(int(labels.max()) + 1) + ' ' + str(acc.data.cpu().numpy()) + '\n' )
        f_acc.write(str(acc.data.cpu().numpy()) + '\n' )
        f_time.write(str(time_o) + '\n')

    print(f"mean_acc about {len(adjpath)} palnt is {all_acc/len(adjpath)}")
    mean_acc = (all_acc/len(adjpath)).data.cpu().numpy()
    mtime = time_all / len(adjpath)
    f_macc.write(str(mean_acc) + '\n' )
    f_mtime.write(str(mtime) + '\n')
    print("End of write")
    f_detail.close()
    f_acc.close()
    f_macc.close()
    f_time.close()
    f_mtime.close()