#!/usr/bin/env python
# coding: utf-8

import torch
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
  