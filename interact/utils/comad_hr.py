from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
from interact.utils.read_json_data import read_json, get_pose_history, missing_data
from hydra import compose, initialize
# from read_json_data import read_json, get_pose_history

class CoMaD_HR(Dataset):

    def __init__(
            self, 
            input_n = 15,
            output_n = 15,
            sample_rate = 120,
            output_rate = 15,
            split='train',
            subtask=None
            ):
        cfg = compose(config_name="datasets", overrides=[])
        self.data_dir = cfg.comad_hr
        self.input_frames = input_n 
        self.output_frames = output_n 
        self.sample_rate = sample_rate
        self.output_rate = output_rate
        self.split = split
        self.mapping_json = cfg.comad_mapping

        self.alice_input, self.alice_output = [], []
        self.bob_input, self.bob_output = [], []
        self.robot_input, self.robot_output = [], []
        self.sequence_len = input_n + output_n
        self.input_n = input_n
        self.output_n = output_n
        self.subtask = subtask

        ### harcoded joints order
        self.alice_joint_used = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10])
        self.bob_joints_used = np.array([0, 1])
        self.robot_joints_used = np.array([10, 11])
        self.add_comad_dataset()

    def add_comad_dataset(self):
        for task in os.listdir(f'{self.data_dir}/{self.split}'):
            for episode in os.listdir(f'{self.data_dir}/{self.split}/{task}/HR'):
                print(f'Episode: {self.data_dir}/{self.split}/{task}/HR/{episode}')
                try:
                    json_data = read_json(f'{self.data_dir}/{self.split}/{task}/HR/{episode}/data.json')
                except:
                    continue
                metadata = read_json(f'{self.data_dir}/{self.split}/{task}/HR/{episode}/metadata.json')
                # breakpoint()
                alice_name, bob_name, robot_name = json_data.keys()    
                downsample_rate = self.sample_rate // self.output_rate
                # TODO: Change this to use metadata to figure out who Alice and Bob are
                alice_tensor = get_pose_history(json_data, 
                                alice_name)[::downsample_rate,self.alice_joint_used]
                bob_tensor = get_pose_history(json_data, 
                                bob_name)[::downsample_rate, self.bob_joints_used]
                try: 
                    robot_tensor = get_pose_history(json_data, 
                                robot_name)[::downsample_rate, self.robot_joints_used]
                except:
                    print("Robot not found")
                    continue
                # robot_tensor = get_pose_history(json_data, 
                #                 robot_name)[::downsample_rate, self.robot_joints_used]
                def fix_orientation(tensor):
                    tensor[:,:,[0,1,2]] = tensor[:,:,[0,2,1]]
                    tensor[:,:,0] *= -1
                    return tensor

                alice_tensor = fix_orientation(alice_tensor)
                bob_tensor = fix_orientation(bob_tensor)
                robot_tensor = fix_orientation(robot_tensor)

                # chop the tensor into a bunch of slices of size sequence_len
                for start_frame in range(alice_tensor.shape[0]-self.sequence_len):
                    end_frame = start_frame + self.sequence_len
                    if missing_data(alice_tensor[start_frame:end_frame]) or \
                        missing_data(bob_tensor[start_frame:end_frame]) or \
                        missing_data(robot_tensor[start_frame:end_frame]):
                        # print("MISSING DATA")
                        continue
                    self.alice_input.append(alice_tensor[start_frame:start_frame+self.input_n])
                    self.alice_output.append(alice_tensor[start_frame+self.input_n:end_frame])

                    self.bob_input.append(bob_tensor[start_frame:start_frame+self.input_n])
                    self.bob_output.append(bob_tensor[start_frame+self.input_n:end_frame])

                    self.robot_input.append(robot_tensor[start_frame:start_frame+self.input_n])
                    self.robot_output.append(robot_tensor[start_frame+self.input_n:end_frame])
        print(len(self.alice_input))

    def __len__(self):
        return len(self.alice_input)

    def __getitem__(self, idx):
        # each element of the data list is of shape (sequence length, 25 joints, 3d)
        return self.alice_input[idx], self.alice_output[idx], self.bob_input[idx], self.bob_output[idx], self.robot_input[idx], self.robot_output[idx]

if __name__ == "__main__":
    dataset = CoMaD_HR()