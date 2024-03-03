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
            # data_dir = './interact/data/comad_data', 
            input_n = 15,
            output_n = 15,
            sample_rate = 120,
            output_rate = 15,
            # mapping_json = "./interact/data/mapping/comad_mapping.json", 
            # transition_file = "./interact/data/comad_data/test_transition.json",
            split='train',
            subtask=None,
            transitions=False
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
        self.subtask = subtask

        joint_names = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut',
                        'LHandOut', 'RHandOut']

        mapping = read_json(self.mapping_json)
        self.joint_used = np.array([mapping[joint_name] for joint_name in joint_names])

        transition_json = read_json(cfg.comad_transitions)
        # print(transition_json)
        self.transition_map = {k[:-2]: {} for k in transition_json.keys()}
        for k, v in transition_json.items():
            ep_name = k[:-2]
            self.transition_map[ep_name]['Alice' if v['reaching'] == 'true' else 'Bob'] = v
        if not transitions:
            self.add_comad_dataset()
        else:
            self.add_comad_transitions()

    def add_comad_dataset(self):
        for episode in os.listdir(f'{self.data_dir}/{self.split}'):
            if self.subtask and self.subtask not in episode:
                continue
            print(f'Episode: {self.data_dir}/{self.split}/{episode}')
            try:
                json_data = read_json(f'{self.data_dir}/{self.split}/{episode}')
            except:
                continue

            downsample_rate = self.sample_rate // self.output_rate
            # TODO: Change this to use metadata to figure out who Alice and Bob are
            alice_tensor = get_pose_history(json_data, 
                            "Atiksh")[::downsample_rate,self.joint_used]
            bob_tensor = get_pose_history(json_data, 
                            "Kushal")[::downsample_rate,self.joint_used]
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

    def add_comad_transitions(self):
        for episode in os.listdir(f'{self.data_dir}/{self.split}'):
            if self.subtask and self.subtask not in episode:
                continue
            print(f'Episode: {self.data_dir}/{self.split}/{episode}')
            try:
                json_data = read_json(f'{self.data_dir}/{self.split}/{episode}')
            except:
                continue

            slices = [(convert_time_to_frame(s_time, self.sample_rate, 0), convert_time_to_frame(e_time, self.sample_rate, 0))
                for (s_time, e_time) in self.transition_map[episode]['Alice']['timestamps']]
                
            downsample_rate = self.sample_rate // self.output_rate
            # TODO: Change this to use metadata to figure out who Alice and Bob are
            for s, e in slices:
                alice_tensor = get_pose_history(json_data, 
                                self.transition_map[episode]['Alice']['name'])[s:e:downsample_rate,self.joint_used]
                bob_tensor = get_pose_history(json_data, 
                                self.transition_map[episode]['Bob']['name'])[s:e:downsample_rate,self.joint_used]
                # chop the tensor into a bunch of slices of size sequence_len
                for start_frame in range(alice_tensor.shape[0]-self.sequence_len):
                    end_frame = start_frame + self.sequence_len
                    if missing_data(alice_tensor[start_frame:end_frame]) or \
                        missing_data(bob_tensor[start_frame:end_frame]):
                        print("MISSING DATA")
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