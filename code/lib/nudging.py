#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
import torch as t
t.manual_seed(100)
import math
import torch.nn.functional as f

class exp_loss(t.nn.Module): 
    def __init__(self,scale): 
        super(exp_loss, self).__init__()
        self.scale = scale

    def forward(self,x): 
        return t.exp(self.scale * x)-1

class doubleexp_loss(t.nn.Module): 
    def __init__(self,scale): 
        super(doubleexp_loss, self).__init__()
        self.scale = scale

    def forward(self,x): 
        return t.exp(self.scale * x)+t.exp(-self.scale * x)-2

class nudging_model(t.nn.Module):
    def __init__(self, num_ways, act="tanhshrink",act_exp=5):
        super(nudging_model, self).__init__()
 
        self.act_exp = act_exp
        self.tnhscaleP = t.nn.Parameter(t.ones([1], dtype=t.float32)*1, requires_grad=False)  # init to 1.5
        self.mask = t.nn.Parameter(t.triu(t.ones([num_ways, num_ways], dtype=t.uint8), diagonal=1), requires_grad=False) # .to(self._device)
        self.mask_sum = t.sum(self.mask)
        self.cos = t.nn.CosineSimilarity()

        if act == "exp": 
            self.act = exp_loss(act_exp)
        elif act == "doubleexp":
            self.act = doubleexp_loss(act_exp)
        else: 
            raise ValueError("Non-valid nudging activation function. Got {:}".format(act))

    def init_params(self, initial_prototypes):

        self.prod_vecs = t.nn.Parameter(initial_prototypes)

    def forward(self, initial_prototypes):        
        # compute cross-correlation loss
        prod_vecs = t.tanh(self.tnhscaleP * self.prod_vecs)
        norm_prod_vecs = f.normalize(prod_vecs, p=2, dim=1)
        prod_sims = t.tensordot(norm_prod_vecs, t.transpose(norm_prod_vecs, 0, 1), dims=1)*self.mask
        prod_sim_loss = self.act(prod_sims) 
        prod_sim_loss = t.sum(prod_sim_loss)/self.mask_sum
      
        # compute deviation loss
        initial_prototypes = t.tanh(self.tnhscaleP *initial_prototypes)
        deviation = self.cos(prod_vecs, initial_prototypes)
        avg_deviation_loss = t.mean(1-deviation)

        total_loss = prod_sim_loss + avg_deviation_loss
        return total_loss, prod_sim_loss, avg_deviation_loss


def nudge_prototypes(avg_prototypes,writer,session=0,gpu=0, 
                    num_epochs=10,bipolarize_prototypes=False,
                    learning_rate=0.1,act="exp",
                    act_exp=4):
    '''
    Prototype nudging

    Parameters:
    -----------
    avg_prototypes:     Tensor (num_ways, D)
        Current prototypes
    writer:             Tensorboard writer
    session:            int
    gpu:                int 
        GPU index 
    num_epoch:          int
    bipolarize_prototypes: Boolean
        Bipolarize prototypes before nudging. Always set false (not effective for now)
    learning_rate:      float
    act:                string
        Nudging activation: "doubleexp", "exp"
    act_exp:            float
        Exponent in nudgin activation
    ''' 

    num_ways, dim_features = avg_prototypes.shape
    model = nudging_model(num_ways, act=act,act_exp=act_exp)
    model.init_params(avg_prototypes.detach().cpu())

    optimizer = t.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.9)

    model.cuda(gpu)
    avg_prototypes.cuda(gpu)

    if bipolarize_prototypes: 
        avg_prototypes = t.sign(avg_prototypes)

    for epoch in range(num_epochs): 

        optimizer.zero_grad()
        total_loss, prod_sim_loss, avg_deviation_loss = model(avg_prototypes)
        total_loss.backward()
        optimizer.step()

        writer.add_scalar('nudging/loss_sess{:}/total'.format(session), total_loss.item(), epoch)
        writer.add_scalar('nudging/loss_sess{:}/prod_sim'.format(session), prod_sim_loss.item(), epoch)
        writer.add_scalar('nudging/loss_sess{:}/avg_dev'.format(session), avg_deviation_loss.item(), epoch)
        
    return model.prod_vecs.data 

