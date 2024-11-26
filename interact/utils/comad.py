from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
from interact.utils.read_json_data import read_json, get_pose_history, missing_data
from hydra import compose 
# from read_json_data import read_json, get_pose_history

def convert_time_to_frame(time, hz, offset):
    mins = int(time[:time.find(':')])
    secs = int(time[time.find(':')+1:])
    return ((mins*60 + secs) * hz) + offset

class CoMaD(Dataset):

    def __init__(
            self, 
            input_n = 15,
            output_n = 15,
            sample_rate = 120,
            output_rate = 15,
            split='train',
            ):
        cfg = compose(config_name="datasets", overrides=[])
        self.data_dir = cfg.comad
        self.input_frames = input_n 
        self.output_frames = output_n 
        self.sample_rate = sample_rate
        self.output_rate = output_rate
        self.split = split
        self.mapping_json = cfg.comad_mapping

        self.alice_input, self.alice_output = [], []
        self.bob_input, self.bob_output = [], []
        self.sequence_len = input_n + output_n
        self.input_n = input_n
        self.output_n = output_n

        joint_names = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut',
                        'LHandOut', 'RHandOut']

        mapping = read_json(self.mapping_json)
        self.joint_used = np.array([mapping[joint_name] for joint_name in joint_names])
        self.add_comad_dataset()

    def add_comad_dataset(self):
        for task in os.listdir(f'{self.data_dir}/{self.split}'):
            for episode in os.listdir(f'{self.data_dir}/{self.split}/{task}/HH'):
                print(f'Episode: {self.data_dir}/{self.split}/{task}/HH/{episode}')
                try:
                    json_data = read_json(f'{self.data_dir}/{self.split}/{task}/HH/{episode}/data.json')
                except:
                    continue
                metadata = read_json(f'{self.data_dir}/{self.split}/{task}/HH/{episode}/metadata.json')
                alice_name, bob_name = metadata.keys()
                downsample_rate = self.sample_rate // self.output_rate
                alice_tensor = get_pose_history(json_data, 
                                alice_name)[::downsample_rate,self.joint_used]
                bob_tensor = get_pose_history(json_data, 
                                bob_name)[::downsample_rate,self.joint_used]
                # chop the tensor into a bunch of slices of size sequence_len
                for start_frame in range(alice_tensor.shape[0]-self.sequence_len):
                    end_frame = start_frame + self.sequence_len
                    if missing_data(alice_tensor[start_frame:end_frame]) or \
                        missing_data(bob_tensor[start_frame:end_frame]):
                        # print("MISSING DATA")
                        continue
                    self.alice_input.append(alice_tensor[start_frame:start_frame+self.input_n])
                    self.alice_output.append(alice_tensor[start_frame+self.input_n:end_frame])

                    self.bob_input.append(bob_tensor[start_frame:start_frame+self.input_n])
                    self.bob_output.append(bob_tensor[start_frame+self.input_n:end_frame])

                    ### Flip Alice and Bob
                    self.alice_input.append(bob_tensor[start_frame:start_frame+self.input_n])
                    self.alice_output.append(bob_tensor[start_frame+self.input_n:end_frame])

                    self.bob_input.append(alice_tensor[start_frame:start_frame+self.input_n])
                    self.bob_output.append(alice_tensor[start_frame+self.input_n:end_frame])
                # break
        print(len(self.alice_input))

    def __len__(self):
        return len(self.alice_input)

    def __getitem__(self, idx):
        # each element of the data list is of shape (sequence length, 25 joints, 3d)
        return self.alice_input[idx], self.alice_output[idx], self.bob_input[idx], self.bob_output[idx]

if __name__ == "__main__":
    dataset = CoMaD()