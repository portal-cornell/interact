import os
import numpy as np
import torch
import random
import json
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
            trans = pose_all['trans']
            frame_rate = pose_all['mocap_framerate']
            fn = poses.shape[0]
            sample_rate = int(frame_rate // 15)
            fidxs = range(0, fn, sample_rate)
            fn = len(fidxs)

            poses = poses[fidxs]
            trans = trans[fidxs]
            poses = torch.from_numpy(poses).float()
            poses = poses.reshape([fn, -1, 3])
            # poses[:, 0:, 0] = np.expand_dims(trans, 1)
            # import pdb; pdb.set_trace()
            # poses[:, 0] = torch.from_numpy(trans).float()
            poses[:, 0] = 0
            p3d0_tmp = p3d0.repeat([fn, 1, 1])
            p3d_human = ang2joint(p3d0_tmp, poses, parent)
            
            alice_poses.append(p3d_human.numpy())
            lengths.append(p3d_human.shape[0])
        if len(alice_poses)>100:
            break

max_length = np.max(np.array(lengths))
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
        elif alice_pose.shape[0] == max_length:
            bob_poses.append(alice_pose)
            bob_not_found = False
        # elif alice_pose.shape[0] > 22900:
        #     if bob_pose.shape[0] > 22000:
        #         alice_poses[alice_index] = alice_poses[alice_index][:bob_pose.shape[0]]
        #         bob_poses.append(bob_pose)
        #         bob_not_found = False

import copy
for idx, (alice_pose, bob_pose) in enumerate(zip(alice_poses, bob_poses)):
    ### TODD: Do some transformation and combine so there is no close interaction
    # import pdb; pdb.set_trace()
    away = False
    while not away:
        translation = np.random.rand(2)*2-1
        rotation = np.random.rand(1)*3.14*0.5

        # import pdb; pdb.set_trace()
        cand_bob_pose = copy.deepcopy(bob_pose)
        cand_bob_pose[:, :, [0, 2]] += translation
        x, y = cand_bob_pose[:, :, 0], cand_bob_pose[:, :, 2]
        cand_bob_pose[:, :, 0] = x*np.cos(rotation)+y*np.sin(rotation)
        cand_bob_pose[:, :, 2] = -x*np.sin(rotation)+y*np.cos(rotation)

        dist = np.min(np.linalg.norm(cand_bob_pose-alice_pose,axis=-1))
        if dist > 0.5:
            away = True

    print(alice_filenames[idx])
    alice_bob_data = {
        'alice': alice_pose.tolist(),
        'bob': cand_bob_pose.tolist()
    }

    filename = alice_filenames[idx].replace('/', '_').replace('.npz', '')

    with open(f'synthetic_amass/{filename}.json', 'w') as f:
        json.dump(alice_bob_data, f, indent=4)
    # import pdb; pdb.set_trace()

