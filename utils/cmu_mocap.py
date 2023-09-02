from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
from utils.read_json_data import read_json, get_pose_history
# from read_json_data import read_json, get_pose_history
class CMU_Mocap(Dataset):

    def __init__(
            self, 
            data_dir = 'cmu_mocap', 
            input_n = 15,
            output_n = 15,
            sample_rate = 120,
            output_rate = 15,
            mapping_json = "mocap_mapping.json", 
            split='train'
            ):
        """
        data_dir := './mocap_data'
        mapping_file := './mapping.json'
        """
        self.data_dir = data_dir
        self.input_frames = input_n 
        self.output_frames = output_n 
        self.sample_rate = sample_rate
        self.output_rate = output_rate
        self.split = split
        self.mapping_json = mapping_json

        self.alice_input, self.alice_output = [], []
        self.bob_input, self.bob_output = [], []
        self.sequence_len = input_n + output_n
        self.input_n = input_n
        self.output_n = output_n

        joint_names = ['lowerneck', 
            'lclavicle', 'rclavicle', 
            'lradius', 'rradius', 
            'lwrist', 'rwrist',
            'lhand', 'rhand']

        mapping = read_json(self.mapping_json)
        self.joint_used = np.array([mapping[joint_name] for joint_name in joint_names])
        self.add_cmu_mocap_dataset()

    def add_cmu_mocap_dataset(self):
        for episode in os.listdir(f'{self.data_dir}/{self.split}'):
            print(f'Episode: {self.data_dir}/{self.split}/{episode}')
            json_data = read_json(f'{self.data_dir}/{self.split}/{episode}')

            downsample_rate = self.sample_rate // self.output_rate
            alice_tensor = get_pose_history(json_data, 
                            "alice")[::downsample_rate]
            bob_tensor = get_pose_history(json_data, 
                            "bob")[::downsample_rate]
            # chop the tensor into a bunch of slices of size sequence_len
            for start_frame in range(alice_tensor.shape[0]-self.sequence_len):
                end_frame = start_frame + self.sequence_len
                self.alice_input.append(alice_tensor[start_frame:start_frame+self.input_n])
                self.alice_output.append(alice_tensor[start_frame+self.input_n:end_frame])

                self.bob_input.append(bob_tensor[start_frame:start_frame+self.input_n])
                self.bob_output.append(bob_tensor[start_frame+self.input_n:end_frame])

                ### Flip Alice and Bob
                self.alice_input.append(bob_tensor[start_frame:start_frame+self.input_n])
                self.alice_output.append(bob_tensor[start_frame+self.input_n:end_frame])

                self.bob_input.append(alice_tensor[start_frame:start_frame+self.input_n])
                self.bob_output.append(alice_tensor[start_frame+self.input_n:end_frame])
            
        print(len(self.alice_input))

    def __len__(self):
        return len(self.alice_input)

    def __getitem__(self, idx):
        # each element of the data list is of shape (sequence length, 25 joints, 3d)
        return self.alice_input[idx], self.alice_output[idx], self.bob_input[idx], self.bob_output[idx]

if __name__ == "__main__":
    dataset = CMU_Mocap()