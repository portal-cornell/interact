import os
import pathlib
import numpy as np
import torch
import random
import json
from utils.ang2joint import *
import copy

path_to_data = 'amass/'

skel = np.load('./body_models/smpl_skeleton.npz')
p3d0 = torch.from_numpy(skel['p3d0']).float()
parents = skel['parents']
parent = {i: parents[i] for i in range(len(parents))}
n = 0

# original splits below
amass_splits = {
    'train':['CMU', 'MPI_Limits', 'Eyes_Japan_Dataset', 'KIT', 'EKUT', 'ACCAD'],
    'val': ['SFU'],
    'test': ['BioMotionLab_NTroje']
    }



for split in ['train', 'val', 'test']:
    alice_filenames = []
    alice_poses = []

    lengths = []
    print("IN SPLIT = ", split)
    for ds in amass_splits[split]:
        if not os.path.isdir(path_to_data + ds + '/'):
            continue
        print('>>> loading {}'.format(ds))

        for sub in os.listdir(path_to_data + ds):
            if not os.path.isdir(path_to_data + ds + '/' + sub):
                continue
            for act in os.listdir(path_to_data + ds + '/' + sub):
                if not act.endswith('.npz'):
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
                trans = torch.from_numpy(trans).float()
                
                global_orient = poses[:, :1]
                rot_matrix = rodrigues(global_orient)
                trans_rot = torch.matmul(trans.unsqueeze(1),rot_matrix)

                poses[:, 0] = 0
                p3d0_tmp = p3d0.repeat([fn, 1, 1])
                p3d_human = ang2joint(p3d0_tmp, poses, parent)
                
                p3d_human += trans_rot
                alice_poses.append(p3d_human.numpy())
                lengths.append(p3d_human.shape[0])

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

    pathlib.Path(f'./synthetic_amass_rotated/{split}').mkdir(parents=True, exist_ok=True)
    for idx, (alice_pose, bob_pose) in enumerate(zip(alice_poses, bob_poses)):
        ### TODD: Do some transformation and combine so there is no close interaction
        # import pdb; pdb.set_trace()
        away = False
        while not away:
            translation = np.random.rand(2)*4
            bob_rotation = np.random.rand(1)*3.14
            alice_rotation = np.random.rand(1)*3.14

            cand_alice_pose = copy.deepcopy(alice_pose)
            x, y = cand_alice_pose[:, :, 0], cand_alice_pose[:, :, 2]
            cand_alice_pose[:, :, 0] = x*np.cos(alice_rotation)+y*np.sin(alice_rotation)
            cand_alice_pose[:, :, 2] = -x*np.sin(alice_rotation)+y*np.cos(alice_rotation)

            cand_bob_pose = copy.deepcopy(bob_pose)
            cand_bob_pose[:, :, [0, 2]] += translation
            x, y = cand_bob_pose[:, :, 0], cand_bob_pose[:, :, 2]
            cand_bob_pose[:, :, 0] = x*np.cos(bob_rotation)+y*np.sin(bob_rotation)
            cand_bob_pose[:, :, 2] = -x*np.sin(bob_rotation)+y*np.cos(bob_rotation)

            dist = np.min(np.linalg.norm(cand_bob_pose-cand_alice_pose,axis=-1))
            if dist > 0.5:
                away = True

        alice_bob_data = {
            'alice': cand_alice_pose.tolist(),
            'bob': cand_bob_pose.tolist()
        }
        print(alice_filenames[idx])
        filename = alice_filenames[idx].replace('/', '_').replace('.npz', '')

        with open(f'./synthetic_amass_rotated/{split}/{filename}.json', 'w') as f:
            json.dump(alice_bob_data, f, indent=4)

