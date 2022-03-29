#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import numpy as np
import sys, os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
from lib.nudging import nudge_prototypes
from .embeddings.ResNet12 import ResNet12
from .embeddings.resnet import resnet18
from .embeddings.ResNet20 import ResNet20
from lib.torch_blocks import fixCos, softstep, step, softabs, softrelu, cosine_similarity_multi, scaledexp
t.manual_seed(0) #for reproducability
import math
import pdb
# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------
class KeyValueNetwork(nn.Module):

    # ----------------------------------------------------------------------------------------------
    # Special Functions & Properties
    # ----------------------------------------------------------------------------------------------

    def __init__(self, args,mode="meta"):
        super().__init__()

        self.args = args
        self.mode = mode

        # Modules
        if args.block_architecture == "mini_resnet12":
            self.embedding = ResNet12(args)
        elif args.block_architecture == "mini_resnet18": 
            self.embedding = resnet18(num_classes=args.dim_features)
        elif args.block_architecture == "mini_resnet20": 
            self.embedding = ResNet20(num_classes=args.dim_features)
        
        # Load pretrain FC module 
        if args.pretrainFC == "spherical": # use cosine similarity
            self.fc_pretrain = fixCos(args.dim_features,args.base_class)
        else: 
            self.fc_pretrain = nn.Linear(args.dim_features,args.base_class,bias=False)

        # Activations
        activation_functions = {
            'softabs':  (lambda x: softabs(x, steepness=args.sharpening_strength)),
            'softrelu': (lambda x: softrelu(x, steepness=args.sharpening_strength)),
            'relu':     nn.ReLU, 
            'abs':      t.abs,
            'scaledexp': (lambda x: scaledexp(x, s = args.sharpening_strength)),
            'exp':      t.exp
        }
        approximations = {
            'softabs':  'abs',
            'softrelu': 'relu'
        }
        
        self.sharpening_activation = activation_functions[args.sharpening_activation]

        # Access to intermediate activations
        self.intermediate_results = dict()
        
        self.feat_replay = t.zeros((args.num_classes,self.embedding.n_interm_feat)).cuda(args.gpu)
        self.label_feat_replay = t.diag(t.ones(self.args.num_classes)).cuda(args.gpu)

    # ----------------------------------------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------------------------------------

    def forward(self, inputs):
        '''
        Forward pass of main model

        Parameters:
        -----------
        inputs:  Tensor (B,H,W)
            Input data
        Return: 
        -------
        output:  Tensor (B,ways)
        ''' 
        # Embed batch
        query_vectors = self.embedding(inputs)

        if self.mode =="pretrain":
            output =  self.fc_pretrain(query_vectors)
        else: 
            ##################### Cosine similarities #########################################################
            self.similarities = cosine_similarity_multi(query_vectors, self.key_mem, rep=self.args.representation)
          
            ################# Sharpen the similarities with a soft absolute activation ############################
            similarities_sharpened = self.sharpening_activation(self.similarities)
                
            # Normalize the similarities in order to turn them into weightings
            if self.args.normalize_weightings:
                denom = t.sum(similarities_sharpened, dim=1, keepdim=True)
                weightings = t.div(similarities_sharpened, denom)
            else:
                weightings = similarities_sharpened

            # Return weighted sum of labels
            if self.args.average_support_vector_inference:
                output = weightings
            else:
                output = t.matmul(weightings, self.val_mem)

        return output

    def write_mem(self,x,y):
        '''
        Rewrite key and value memory

        Parameters:
        -----------
        x:  Tensor (B,D)
            Input data
        y:  Tensor (B,w)
            One-hot encoded classes
        ''' 
        self.key_mem = self.embedding(x)
        self.val_mem = y

        if self.args.average_support_vector_inference:
            self.key_mem = t.matmul(t.transpose(self.val_mem,0,1), self.key_mem)
        return


    def reset_prototypes(self,args): 
        if hasattr(self,'key_mem'):
            self.key_mem.data.fill_(0.0)
        else: 
            self.key_mem = nn.parameter.Parameter(t.zeros(self.args.num_classes, self.args.dim_features),requires_grad=False).cuda(args.gpu)
            self.val_mem = nn.parameter.Parameter(t.diag(t.ones(self.args.num_classes)),requires_grad=False).cuda(args.gpu)

    def update_prototypes(self,x,y): 
        '''
        Update key memory  

        Parameters:
        -----------
        x:  Tensor (B,D)
            Input data
        y:  Tensor (B)
            lables 
        ''' 

        support_vec = self.embedding(x)
        y_onehot = F.one_hot(y, num_classes = self.args.num_classes).float()
        prototype_vec = t.matmul(t.transpose(y_onehot,0,1), support_vec)
        self.key_mem.data += prototype_vec

    def bipolarize_prototypes(self):
        '''
        Bipolarize key memory   
        '''
        self.key_mem.data = t.sign(self.key_mem.data)

    def get_sum_support(self,x,y):
        '''
        Compute prototypes
        
        Parameters:
        -----------
        x:  Tensor (B,D)
            Input data
        y:  Tensor (B)
            lables 
        '''
        support_vec = self.embedding(x)
        y_onehot = F.one_hot(y, num_classes = self.args.num_classes).float()
        sum_cnt = t.sum(y_onehot,dim=0).unsqueeze(1)
        sum_support = t.matmul(t.transpose(y_onehot,0,1), support_vec)
        return sum_support, sum_cnt


    def update_feat_replay(self,x,y): 
        '''
        Compute feature representatin of new data and update
        Parameters:
        -----------
        x   t.Tensor(B,in_shape)
            Input raw images
        y   t.Tensor (B)
            Input labels

        Return: 
        -------
        '''
        feat_vec = self.embedding.forward_conv(x)
        y_onehot = F.one_hot(y, num_classes = self.args.num_classes).float()
        sum_cnt = t.sum(y_onehot,dim=0).unsqueeze(1)
        sum_feat_vec = t.matmul(t.transpose(y_onehot,0,1), feat_vec)
        avg_feat_vec = t.div(sum_feat_vec,sum_cnt+1e-8)
        self.feat_replay += avg_feat_vec 

    def get_feat_replay(self): 
        return self.feat_replay, self.label_feat_replay

    def update_prototypes_feat(self,feat,y_onehot,nways=None): 
        '''
        Update key 

        Parameters:
        -----------
        feat:  Tensor (B,d_f)
            Input data
        y:  Tensor (B)
        nways: int
            If none: update all prototypes, if int, update only nwyas prototypes
        ''' 
        support_vec = self.get_support_feat(feat)
        prototype_vec = t.matmul(t.transpose(y_onehot,0,1), support_vec)

        if nways is not None: 
            self.key_mem.data[:nways] += prototype_vec[:nways]
        else:
            self.key_mem.data += prototype_vec

    def get_support_feat(self,feat): 
        '''
        Pass activations through final FC 

        Parameters:
        -----------
        feat:  Tensor (B,d_f)
            Input data
        Return:
        ------
        support_vec:  Tensor (B,d)
            Mapped support vectors
        ''' 
        support_vec = self.embedding.fc(feat)
        return support_vec

    def nudge_prototypes(self,num_ways,writer,session,gpu): 
        '''
        Prototype nudging
        Parameters:
        -----------
        num_ways:   int
        writer:     Tensorboard writer
        session:    int
        gpu:        int

        ''' 
        prototypes_orig = self.key_mem.data[:num_ways]
        self.key_mem.data[:num_ways]  = nudge_prototypes(prototypes_orig,writer,session,
                                                        gpu=self.args.gpu,num_epochs=self.args.nudging_iter,
                                                        bipolarize_prototypes=self.args.bipolarize_prototypes,
                                                        act=self.args.nudging_act,
                                                        act_exp = self.args.nudging_act_exp)
        return

    def hrr_superposition(self,num_ways,nsup=2): 
        '''
        Compression an retrieval of EM with HRR
        Parameters: 
        ----------
        num_ways: Number of active ways, if not specified entire memory will be bipolarized
        nsup: number of superimposed vectors 
        '''
        n_comp = math.ceil(num_ways/nsup)
        for m in range(n_comp):        
            # generate a new set of keys
            key =1/math.sqrt(self.args.dim_features)*t.randn((nsup,self.args.dim_features))
            superpos = t.zeros(self.args.dim_features,1).cuda(self.args.gpu) 
            for way in range(nsup): 
                rotMat = t.FloatTensor().set_(key[way].repeat(2).storage(), storage_offset=0, 
                                                size=t.Size((self.args.dim_features,self.args.dim_features)), 
                                                stride=(1, 1)).cuda(self.args.gpu) 
                superpos = superpos + t.mm(rotMat,self.key_mem.data[way+m*nsup].view(-1,1))

            # retrieval 
            for way in range(nsup):
                if way+m*nsup<num_ways: # only restore if needed 
                    rotMat = t.FloatTensor().set_(key[way].repeat(2).storage(), storage_offset=0, 
                                                    size=t.Size((self.args.dim_features,self.args.dim_features)),
                                                    stride=(1, 1)).cuda(self.args.gpu) 
                    self.key_mem.data[way+m*nsup] = t.mm(rotMat,superpos).squeeze()