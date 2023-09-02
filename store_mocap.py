from amc_parser import *
import json
import numpy as np
import os
import pathlib

def get_motion_list(joints, motions, scale=0.056444):
    motion_list = []
    for frame_idx in range(len(motions)):
        joints['root'].set_motion(motions[frame_idx])
        joints_list=[]
        for joint in joints.values():
            xyz=np.array([joint.coordinate[0],\
                joint.coordinate[1],joint.coordinate[2]]).squeeze(1)
            joints_list.append(xyz*scale)
        motion_list.append(np.array(joints_list).tolist())
    return motion_list

alice_bob_pairs = [(18,19), (20,21), (22,23), (33, 34)]

pathlib.Path(f'./cmu_mocap').mkdir(parents=True, exist_ok=True)

for A, B in alice_bob_pairs:
    episodes = os.listdir(f'./mocap/all_asfamc/subjects/{A}/')
    for ep_num in range(1, len(episodes)):
        if ep_num<10:
            ep_num = f"0{ep_num}"
        asf_path_A = f'./mocap/all_asfamc/subjects/{A}/{A}.asf'
        amc_path_A = f'./mocap/all_asfamc/subjects/{A}/{A}_{ep_num}.amc'

        asf_path_B = f'./mocap/all_asfamc/subjects/{B}/{B}.asf'
        amc_path_B = f'./mocap/all_asfamc/subjects/{B}/{B}_{ep_num}.amc'

        joints_A = parse_asf(asf_path_A)
        motions_A = parse_amc(amc_path_A)

        joints_B = parse_asf(asf_path_B)
        motions_B = parse_amc(amc_path_B)

        joint_data_A = get_motion_list(joints_A, motions_A)
        joint_data_B = get_motion_list(joints_B, motions_B)

        alice_bob_data = {
            'alice': joint_data_A,
            'bob': joint_data_B
        }

        with open(f'./cmu_mocap/{A}_{B}_{ep_num}.json', 'w') as f:
            json.dump(alice_bob_data, f, indent=4)