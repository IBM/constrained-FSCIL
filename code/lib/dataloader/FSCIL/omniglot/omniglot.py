#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import os
import os.path as osp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
class omniglot(Dataset):

    def __init__(self, root='./data', train=True,
                 transform=None,
                 index_path=None, index=None, base_sess=None):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.IMAGE_PATH = os.path.join(root, 'omniglot/mergeddata')
        self.SPLIT_PATH = os.path.join(root, 'omniglot/split')

        txt_path = osp.join(self.SPLIT_PATH, setname + '.txt')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1
        self.wnids = []
        for l in lines:
            l = l.replace("'","")
            name = l.split('/')
            wnid = ''.join([elem+'/' for elem in name[:-1]])
            path = osp.join(self.IMAGE_PATH, l)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb
        if train:
            image_size = 32
            self.transform = transforms.Compose([
                transforms.Resize([32,32]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=6),
                transforms.ToTensor(),
                ])
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([32,32]),
                transforms.ToTensor(),
                ])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)


    def SelectfromTxt(self, data2label, index_path):
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
#        for line in lines:
#            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for l in lines:
            l = l.replace("'","")
            img_path = os.path.join(self.IMAGE_PATH, l)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('1'))
        image = (1- image)
        return image, targets


if __name__ == '__main__':
    idx = np.arange(1200)
    txt_path = "/gpfs/u/home/IMEM/IMEMmchr/barn/06_mann/CEC-CVPR2021/data/index_list/omniglot/query_batch_2.txt"
    # class_index = open(txt_path).read().splitlines()
    base_class = 100
    class_index = np.arange(base_class)
    dataroot = '/gpfs/u/home/IMEM/IMEMmchr/scratch-shared/data/'
    batch_size_base = 1
    trainset = omniglot(root=dataroot, train=False, transform=None,base_sess=True, index_path=txt_path,index=idx)
    cls = np.unique(trainset.targets)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                              pin_memory=True)
    print(cls)


    for batch in trainloader: 
        print(batch)
        pdb.set_trace()
