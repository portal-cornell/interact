import sys
import numpy as np
import json

def convert_time_to_frame(time, hz=15, offset=0):
    mins = int(time[:time.find(':')])
    secs = float(time[time.find(':')+1:])
    return int(((mins*60 + secs) * hz) + offset * hz)
# import pdb; pdb.set_trace()
bag_name = sys.argv[1]

with open(f'comad/human_robot_data/timestamps/{bag_name}.txt') as f: data = f.readlines()
start_end_times = [list(map(convert_time_to_frame, se.split('-'))) for se in data]
clips = []

episode_file = f'comad/human_robot_data/jsons/{bag_name}.json'
with open(episode_file, 'r') as f:
    data = json.load(f)
atiksh_frames = data['Atiksh']
kushal_frames = data['Kushal']
robot_frames = data['Robot']
for start_frame, end_frame in start_end_times:
    atiksh_arr = np.nan_to_num(atiksh_frames[start_frame:end_frame]).tolist()
    kushal_arr = np.nan_to_num(kushal_frames[start_frame:end_frame]).tolist()
    robot_arr = np.nan_to_num(robot_frames[start_frame:end_frame]).tolist()
    current_clip = {"Atiksh":atiksh_arr, "Kushal":kushal_arr, "Robot":robot_arr}
    clips.append(current_clip)



for idx in range(len(clips)):
    with open(f'./comad/human_robot_data/jsons/{bag_name}_{idx}.json', 'w') as f:
        json.dump(clips[idx], f, indent=4)