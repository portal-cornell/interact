import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

activity_name = 'handover'
activity_number = '4'

csv_name = f'./comad/csv/{activity_name}/{activity_name}_{activity_number}.csv'
df = pd.read_csv(csv_name, header=None,skiprows=2, low_memory=False)
joints_to_idx = {}
joints_to_idx = {}
marker_names = list(df.iloc[1])

# first_marker = 26*26+26*3+3
# last_marker = 26*26+26*8+20

first_marker_condition = (df.loc[0] == 'Marker') & (df.loc[1] == "Atiksh:BackLeft")
first_marker = first_marker_condition[first_marker_condition].index[0]

last_marker_condition = (df.loc[0] == 'Marker') & (df.loc[1] == "Kushal:WaistRFront")
last_marker = last_marker_condition[last_marker_condition].index[0] + 2 # this index might need to be -1 to get the last of the 3d coords

print(first_marker)
print(last_marker)
print(marker_names[first_marker:last_marker+1])
for idx, marker_name in enumerate(marker_names[first_marker:last_marker+1]):
    joints_to_idx[marker_name] = len(joints_to_idx)-1
print(joints_to_idx)

frames = np.array(df.iloc[5:], dtype=float)
frames = frames[:, first_marker:last_marker+1]
print(frames.shape)
input()
frames = frames.reshape(-1, 50, 3)/1000
print(frames.shape)

table_set1_ep = [
    ("1:01","1:22"),
    ("1:53","2:08"),
    ("2:41","3:01"),
    ("3:35","4:00"),
    ("4:25","4:48"),
    ("5:18","5:37"),
    ("5:58","6:19"),
    ("6:56","7:16"),
    ("8:44","9:02"),
    ("9:25","9:47"),
]
table_set1_offset = 1

table_set2_ep = [
    ("0:32","0:54"),
    ("1:18","1:42"),
    ("2:30","2:56"),
    ("3:19","3:36"),
    ("3:56","4:20"),
    ("5:38","6:07"),
    ("6:58","7:21"),
    ("7:48","8:12"),
    ("8:41","9:02"),
    ("9:29","9:49"),
    ("10:18","10:38"),
    ("11:01","11:22"),
    ("11:50","12:12"),
    ("12:41","13:07"),
    ("14:02","14:26"),
    ("14:49","15:08"),
    ("15:29","15:49"),
    ("16:33","16:53"),
    ("17:31","17:49"),
    ("18:10","18:28"),
    ("18:49","19:05"),
]
table_set2_offset = .5

table_set3_ep = [
    ("0:28", "0:49"),
    ("1:15", "1:32"),
    ("1:59", "2:21"),
    ("2:41", "3:02"),
    ("3:20", "3:37"),
    ("4:06", "4:23"),
    ("4:42", "5:03"),
    ("5:27", "5:43"),
    ("6:09", "6:30"),
    ("6:54", "7:18"),
    ("7:48", "8:07"),
    ("8:33", "8:54"),
    ("9:12", "9:29"),
]

table_set3_offset = .5

table_set4_ep = [
    ("0:27", "0:43"),
    ("1:03", "1:21"),
    ("1:44", "1:59"),
    ("2:18", "2:36"),
    ("2:59", "3:18"),
    ("3:35", "3:54"),
    ("4:24", "4:44"),
    ("5:03", "5:17"),
    ("6:28", "6:51"),
    ("7:08", "7:29"),
    ("7:48", "8:03"),
    ("8:28", "8:48"),
    ("9:19", "9:36")
]

table_set4_offset = 1

react_stir1_ep = [
    ("0:30", "1:06"),
    ("2:21", "3:01"),
    ("3:30", "4:01"),
    ("4:29", "4:57"),
    ("5:28", "6:05"),
    ("7:27", "7:56"),
    ("8:22", "8:52"),
    ("9:37", "10:03"),
]

react_stir1_offset = 1

react_stir2_ep = [
    ("0:12", "0:37"),
    ("1:02", "1:28"),
    ("1:48", "2:15"),
    ("2:48", "3:14"),
    ("3:58", "4:20"),
    ("4:36", "4:59"),
    ("5:23", "5:49"),
    ("6:13", "6:39"),
    ("7:07", "7:30"),
    ("7:54", "8:14"),
    ("8:31", "9:00"),
]

react_stir2_offset = 1

start_end_file = f'./comad/json/{activity_name}/{activity_name}_{activity_number}/start_end.txt'

current_ep = []
curr_offset = 0
with open(start_end_file, 'r') as f:
    times = f.readlines()
    curr_offset = int(times[0])
    times = times[1::1]
    for line in times:
        start_end = line.split(" - ")
        current_ep.append((start_end[0], start_end[1]))


def convert_time_to_frame(time, hz, offset):
    mins = int(time[:time.find(':')])
    secs = int(time[time.find(':')+1:])
    return int(((mins*60 + secs) * hz) + offset * hz)

clips = []
atiksh_arr_fin = []
kushal_arr_fin = []
for start, end in current_ep:
    start_frame = convert_time_to_frame(start, 120, curr_offset)
    end_frame = convert_time_to_frame(end, 120, curr_offset) + 240  #The +240 frames here is essentially adding 2 extra seconds to the episode

    atiksh_arr = np.nan_to_num(frames[start_frame:end_frame, :25, :]).tolist()
    atiksh_arr_fin.append(atiksh_arr)
    kushal_arr = np.nan_to_num(frames[start_frame:end_frame, 25:, :]).tolist()
    kushal_arr_fin.append(kushal_arr)
    current_clip = {"Atiksh":atiksh_arr, "Kushal":kushal_arr}
    clips.append(current_clip)

# data = []
# for i in range(len(atiksh_arr_fin)):
#     for j in range(len(atiksh_arr_fin[i])):
#         interim = [atiksh_arr_fin[i][j], kushal_arr_fin[i][j]]
#         data.append(interim)

# np.save(f"./comad/{activity_name}/{activity_name}_train.npy", np.array(data))
import json
for idx in range(len(clips)):
    with open(f'./comad/json/{activity_name}/{activity_name}_{activity_number}/{activity_name}_{activity_number}_{idx}.json', 'w') as f:
        json.dump(clips[idx], f, indent=4)