#!/usr/bin/env python
# coding: utf-8

import torch
from utils import data_utils
import numpy as np 



def mpjpe_error(batch_pred,batch_gt): 

    batch_pred=batch_pred.contiguous().view(-1,3)
    batch_gt=batch_gt.contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))

def fde_error(batch_pred,batch_gt): 

    batch_pred=batch_pred[:,-1,:,:].contiguous().view(-1,3)
    batch_gt=batch_gt[:,-1,:,:].contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))

def weighted_mpjpe_error(batch_pred,batch_gt, joint_weights): 
    # 'BackTop', 'LShoulderBack', 'RShoulderBack',
    #                   'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut'
    batch_pred=batch_pred.contiguous()
    batch_gt=batch_gt.contiguous()
    diff = batch_gt - batch_pred
    diff *= joint_weights
    all_joints_error = (torch.norm(diff.view(-1,3),2,1)).view(batch_pred.shape[0], -1)
    return torch.mean(all_joints_error, dim=1)

def perjoint_error(batch_pred, batch_gt):
    batch_pred = batch_pred.contiguous()
    batch_gt=batch_gt.contiguous()
    diff = batch_gt - batch_pred
    batch_joint_errors = torch.mean(torch.norm(diff,2,3), 1)
    return torch.mean(batch_joint_errors,0)
    
def euler_error(ang_pred, ang_gt):

    # only for 32 joints
    
    dim_full_len=ang_gt.shape[2]

    # pred_expmap[:, 0:6] = 0
    # targ_expmap[:, 0:6] = 0
    pred_expmap = ang_pred.contiguous().view(-1,dim_full_len).view(-1, 3)
    targ_expmap = ang_gt.contiguous().view(-1,dim_full_len).view(-1, 3)

    pred_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(pred_expmap))
    pred_eul = pred_eul.view(-1, dim_full_len)

    targ_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(targ_expmap))
    targ_eul = targ_eul.view(-1, dim_full_len)
    mean_errors = torch.mean(torch.norm(pred_eul - targ_eul, 2, 1))

    return mean_errors

def disc_l2_loss(disc_value):
    
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def adv_disc_l2_loss(real_disc_value, fake_disc_value):
    
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb

