from amc_parser import *
import numpy as np
import os

#two subjects data

data=[]
test_data=[]

## 18-19, 20-21, 22-23, 33-34
A='18'
B='19'
ep_num = '01'

asf_path_A = './mocap/all_asfamc/subjects/'+A+'/'+A+'.asf'
amc_path_A = './mocap/all_asfamc/subjects/'+A+'/'+A+'_'+ep_num+'.amc'

asf_path_B = './mocap/all_asfamc/subjects/'+B+'/'+B+'.asf'
amc_path_B = './mocap/all_asfamc/subjects/'+B+'/'+B+'_'+ep_num+'.amc'


joints_A = parse_asf(asf_path_A)
motions_A = parse_amc(amc_path_A)

joints_B = parse_asf(asf_path_B)
motions_B = parse_amc(amc_path_B)

motion_list_A=[]

def get_motion_list(joints, motions):
    motion_list = []
    for frame_idx in range(len(motions)):
        joints['root'].set_motion(motions[frame_idx])
        joints_list=[]
        for joint in joints.values():
            xyz=np.array([joint.coordinate[0],\
                joint.coordinate[1],joint.coordinate[2]]).squeeze(1)
            joints_list.append(xyz)
        motion_list.append(np.array(joints_list))
    return motion_list

motion_list_A = get_motion_list(joints_A, motions_A)
motion_list_B = get_motion_list(joints_B, motions_B)

joint_names = list(joints_A.keys())
joints_to_indices = {}

for idx, joint in enumerate(joint_names):
    joints_to_indices[joint] = idx

import json

with open('mocap_mapping.json', 'w') as f:
    json.dump(joints_to_indices, f, indent=4)
# import pdb; pdb.set_trace()

