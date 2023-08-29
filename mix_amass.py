import os
import numpy as np
import torch
import random
from utils.read_json_data import read_json, get_pose_history, missing_data
from utils.ang2joint import *

path_to_data = 'amass/'

skel = np.load('./body_models/smpl_skeleton.npz')
p3d0 = torch.from_numpy(skel['p3d0']).float()
parents = skel['parents']
parent = {i: parents[i] for i in range(len(parents))}
n = 0

amass_splits = {'train':['CMU']}

alice_filenames = []
alice_poses = []

lengths = []
for ds in amass_splits['train']:
    print("here")
    
    if not os.path.isdir(path_to_data + ds + '/'):
        continue
    print('>>> loading {}'.format(ds))

    for sub in os.listdir(path_to_data + ds):
        if not os.path.isdir(path_to_data + ds + '/' + sub):
            print("EVER")
            continue
        for act in os.listdir(path_to_data + ds + '/' + sub):
            if not act.endswith('.npz'):
                print("NEVER")
                continue

            alice_filenames.append(path_to_data + ds + '/' + sub + '/' + act)

            pose_all = np.load(path_to_data + ds + '/' + sub + '/' + act)
            try:
                poses = pose_all['poses']
            except:
                print('no poses at {}_{}_{}'.format(ds, sub, act))
                continue
            
            fn = poses.shape[0]
            poses = torch.from_numpy(poses).float()
            poses = poses.reshape([fn, -1, 3])
            poses[:, 0] = 0
            p3d0_tmp = p3d0.repeat([fn, 1, 1])
            p3d_human = ang2joint(p3d0_tmp, poses, parent)

            alice_poses.append(p3d_human.numpy())
            lengths.append(p3d_human.shape[0])

bob_poses = []

for alice_index, alice_pose in enumerate(alice_poses):
    bob_not_found = True
    while bob_not_found:
        bob_index = random.randint(0, len(alice_poses)-1)
        if bob_index == alice_index: continue
        bob_pose = alice_poses[bob_index]

        if alice_pose.shape[0] <= bob_pose.shape[0]:
            bob_poses.append(bob_pose[:alice_pose.shape[0]])
            bob_not_found = False
        elif alice_pose.shape[0] > 22900:
            if bob_pose.shape[0] > 22000:
                import pdb; pdb.set_trace()
                alice_poses[alice_index] = alice_poses[alice_index][:bob_pose.shape[0]]
                bob_poses.append(bob_pose)
                bob_not_found = False
        
for alice_pose, bob_pose in zip(alice_poses, bob_poses):
    ### Do some transformation and combine so there is no close interaction
    if alice_pose.shape[0] != bob_pose.shape[0]:
        print("KALIYUGA")

