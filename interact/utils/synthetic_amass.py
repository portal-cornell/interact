from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
from interact.utils.read_json_data import read_json, get_pose_history
from hydra import compose
# from read_json_data import read_json, get_pose_history

class Synthetic_AMASS(Dataset):

    def __init__(
            self, 
            # data_dir = './interact/data/synthetic_amass', 
            input_n = 15,
            output_n = 15,
            sample_rate = 15,
            output_rate = 15,
            # mapping_json = "./interact/data/mapping/amass_mapping.json", 
            split='train'
            ):
        cfg = compose(config_name="datasets", overrides=[])
        self.data_dir = cfg.synthetic_amass
        self.input_frames = input_n 
        self.output_frames = output_n 
        self.sample_rate = sample_rate
        self.output_rate = output_rate
        self.split = split
        self.mapping_json = cfg.amass_mapping

        self.alice_input, self.alice_output = [], []
        self.bob_input, self.bob_output = [], []
        self.sequence_len = input_n + output_n
        self.input_n = input_n
        self.output_n = output_n

        joint_names = ['Neck', 
            'L_Shoulder', 'R_Shoulder',
            'L_Elbow', 'R_Elbow',
            'L_Wrist', 'R_Wrist', 
            'L_Hand', 'R_Hand']

        mapping = read_json(self.mapping_json)
        self.joint_used = np.array([mapping[joint_name] for joint_name in joint_names])
        self.add_amass_dataset()

    def add_amass_dataset(self):
        for episode in os.listdir(f'{self.data_dir}/{self.split}'):
            print(f'Episode: {self.data_dir}/{self.split}/{episode}')
            json_data = read_json(f'{self.data_dir}/{self.split}/{episode}')

            downsample_rate = self.sample_rate // self.output_rate
            alice_tensor = get_pose_history(json_data, 
                            "alice")[::downsample_rate,self.joint_used]
            bob_tensor = get_pose_history(json_data, 
                            "bob")[::downsample_rate,self.joint_used]
            # chop the tensor into a bunch of slices of size sequence_len
            for start_frame in range(alice_tensor.shape[0]-self.sequence_len):
                end_frame = start_frame + self.sequence_len
                self.alice_input.append(alice_tensor[start_frame:start_frame+self.input_n])
                self.alice_output.append(alice_tensor[start_frame+self.input_n:end_frame])

                self.bob_input.append(bob_tensor[start_frame:start_frame+self.input_n])
                self.bob_output.append(bob_tensor[start_frame+self.input_n:end_frame])
            # break
        print(len(self.alice_input))

    def __len__(self):
        return len(self.alice_input)

    def __getitem__(self, idx):
        # each element of the data list is of shape (sequence length, 25 joints, 3d)
        return self.alice_input[idx], self.alice_output[idx], self.bob_input[idx], self.bob_output[idx]

if __name__ == "__main__":
    dataset = Synthetic_AMASS()