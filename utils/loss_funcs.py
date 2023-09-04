#!/usr/bin/env python
# coding: utf-8

import torch
from utils import data_utils
import numpy as np 



def mpjpe_loss(pred, output):
    diff = pred-output
    dist = torch.norm(diff.reshape(diff.shape[0], diff.shape[1], -1, 3), dim=-1)
    loss = torch.mean(dist.reshape(dist.shape[0], -1))
    return loss

def fde_error(pred, output): 
    pred_last_step = pred[:,-1:]
    output_last_step = output[:,-1:]
    return mpjpe_loss(pred_last_step, output_last_step)

def weighted_mpjpe_error(batch_pred,batch_gt, joint_weights): 
    # 'BackTop', 'LShoulderBack', 'RShoulderBack',
    #                   'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut'
    batch_pred=batch_pred.contiguous()
    batch_gt=batch_gt.contiguous()
    diff = batch_gt - batch_pred
    diff *= joint_weights
    all_joints_error = (torch.norm(diff.view(-1,3),2,1)).view(batch_pred.shape[0], -1)
    return torch.mean(all_joints_error, dim=1)

def perjoint_error(pred, output):
    diff = pred-output
    dist = torch.norm(diff.reshape(diff.shape[0], diff.shape[1], -1, 3), dim=-1)
    loss = torch.mean(dist, dim=1)
    return torch.mean(loss,0), loss

def perjoint_fde(pred, output):
    pred_last_step = pred[:,-1:]
    output_last_step = output[:,-1:]
    return perjoint_error(pred_last_step, output_last_step) 
    
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

