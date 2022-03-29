#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import numpy as np
import torch
from lib.dataloader.FSCIL.sampler import CategoriesSampler
import pdb

def set_up_datasets(args):
    
    if args.dataset == 'mini_imagenet':
        import lib.dataloader.FSCIL.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    elif args.dataset == 'cifar100':
        import lib.dataloader.FSCIL.cifar100.cifar100 as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'omniglot':
        import lib.dataloader.FSCIL.omniglot.omniglot as Dataset
        args.base_class = 1200
        args.num_classes=1623
        args.way = 47
        args.shot = 5
        args.sessions = 10
    args.Dataset=Dataset
    return args

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader_meta(args, do_augment =False)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args,session,do_augment =False)
    return trainset, trainloader, testloader

def get_base_dataloader(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)

    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.data_folder, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.data_folder, train=False, download=False,
                                        index=class_index, base_sess=True)


    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.data_folder, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.MiniImageNet(root=args.data_folder, train=False, index=class_index)

    if args.dataset == 'omniglot':
        txt_path = "data/index_list/" + args.dataset + "/support_batch_1.txt"
        trainset = args.Dataset.omniglot(root=args.data_folder, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.omniglot(root=args.data_folder, train=False, index=class_index)


    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_training, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size_inference, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader



def get_base_dataloader_meta(args,do_augment=True):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.data_folder, train=True, download=True,
                                         index=class_index, base_sess=True) #, do_augment=do_augment)
        testset = args.Dataset.CIFAR100(root=args.data_folder, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.data_folder, train=True,
                                             index_path=txt_path, do_augment=do_augment)
        testset = args.Dataset.MiniImageNet(root=args.data_folder, train=False,
                                            index=class_index)
    
    if args.dataset == 'omniglot':
        txt_path = "data/index_list/" + args.dataset + "/support_batch_1.txt"
        trainset = args.Dataset.omniglot(root=args.data_folder, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.omniglot(root=args.data_folder, train=False,
                                            index=class_index)


    sampler = CategoriesSampler(trainset.targets, args.max_train_iter, args.num_ways_training,
                                 args.num_shots_training + args.num_query_training)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size_inference, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args,session, do_augment=True):
    
    # Load support set (don't do data augmentation here )
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.data_folder, train=True, download=False,
                                         index=class_index, base_sess=False) #, do_augment=do_augment)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.data_folder, train=True,
                                       index_path=txt_path, do_augment=do_augment)

    if args.dataset == 'omniglot':
        txt_path = "data/index_list/" + args.dataset + "/support_batch_" + str(session+1) + '.txt'
        trainset = args.Dataset.omniglot(root=args.data_folder, train=True,
                                       index_path=txt_path)

    # always load entire dataset in one batch    
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=trainset.__len__() , shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)


    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.data_folder, train=False, download=False,
                                        index=class_new, base_sess=False)

    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.data_folder, train=False,
                                      index=class_new)

    if args.dataset == 'omniglot':
        testset = args.Dataset.omniglot(root=args.data_folder, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size_inference, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list