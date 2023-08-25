import json
import torch

def missing_data(tensor):
    return torch.any(torch.all(tensor == 0, dim=-1))


"""
Reads in the Motive json data into python object
"""
def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data 

"""
Motive coordinate system (x, y, z) => Real coordinate system (-x, z, y)
"""
def transform_coords(tensor):
    tmp = tensor
    tmp[:, :, 0] = tensor[:, :, 0] * -1
    tmp[:, :, [1, 2]] = tensor[:, :, [2, 1]]
    return tmp

"""
data : python object where keys are names of skeletons mapped to (frames, joints, 3) matrices
skeleton_name : name of the skeleton (Kushal or Prithwish)
fps : frame rate at which we want to downsample to
"""
def get_pose_history(data, skeleton_name):
    return torch.tensor(data[skeleton_name])

if __name__ == "__main__":
    mapping = read_json('./mapping.json')
    print(mapping)
    json_data = read_json('./chopping_stirring_data/chopping_stirring_0.json')
    kushal_tensor = get_pose_history(json_data, "Kushal")
    print(kushal_tensor.shape)
    prithwish_tensor = get_pose_history(json_data, "Prithwish")
    print(prithwish_tensor.shape)