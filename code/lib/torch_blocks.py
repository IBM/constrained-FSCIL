#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
import torch as t
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------------------------------
def softstep(x):
    return (t.tanh(5 * (x - 1)) + 1) / 2 + (t.tanh(5 * (x + 1)) - 1) / 2

def step(x):
    return (t.sign((x - 1)) + 1) / 2 + (t.sign((x + 1)) - 1) / 2

def softabs(x, steepness=10):
    return t.sigmoid(steepness * (x - 0.5)) + t.sigmoid(steepness * (-x - 0.5))

def scaledexp(x, s=1.0): 
    return t.exp(x*s)

def softrelu(x, steepness=10):
    return t.sigmoid(steepness * (x - 0.5))

class Tanh10x(t.nn.Module):
    def __init__(self): 
        super(Tanh10x,self).__init__()

    def forward(self, x): 
        y = t.tanh(10*x)
        return y

SIM_ACT = {"bipolar": t.sign, "tanh": nn.Tanh(),"tanh10x":Tanh10x(), "real": nn.Identity()}
# --------------------------------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------------------------------

def cosine_similarity_multi(a, b, rep = "real"):
    """
    Compute the cosine similarity between two vectors

    Parameters:
    ----------
    a:  Tensor(N_a,D)
    b:  Tensor(N_b,D)
    rep: str
        Representation to compute cosine similarity: real | bipolar | tanh
    Return 
    ------
    similarity: Tensor(N_a,N_b)
    """
    sim_act = SIM_ACT[rep]
    a_normalized = F.normalize(sim_act(a), dim=1) 
    b_normalized = F.normalize(sim_act(b), dim=1)
    similiarity = F.linear(a_normalized, b_normalized) 

    return similiarity


# --------------------------------------------------------------------------------------------------
# Layer modules
# --------------------------------------------------------------------------------------------------

class fixCos(nn.Module):
    def __init__(self, num_features, num_classes, s=1.0):
        '''
        Fixed scale alpha (given as s)
        '''
        super(fixCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = nn.Parameter(t.Tensor([s]))
        self.W = nn.Parameter(t.zeros((num_classes,num_features)))

    def forward(self, input):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # scaled dot product
        logits = self.s*F.linear(x, W)
        return logits

class myCosineLoss(nn.Module): 
    def __init__(self, rep="real"):
        super(myCosineLoss, self).__init__()
        self.sim_act = SIM_ACT[rep]
        self.cos = nn.CosineSimilarity()

    def forward(self,a,b):
        sim = self.cos(self.sim_act(a), self.sim_act(b))
        return -t.mean(sim)

        