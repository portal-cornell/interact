import bagpy
from bagpy import bagreader
import pandas as pd
import os
import numpy as np
import re
import math


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1) * 180 / math.pi
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2) * 180 / math.pi
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4) * 180 / math.pi
    
    return [roll_x, pitch_y, yaw_z] # in degrees

bag_name = "take_2"

digit_parser = r"-?\d*\.\d*"

bag_path = f"./comad/human_robot_data/bag_files/{bag_name}.bag"
topic_names = ["/ee_curr_pose", "/human_forecast"]

br = bagreader(bag_path)


pose = None
human = None

print("preload")
if (os.path.exists(f"./comad/human_robot_data/bag_files/{bag_name}/{topic_names[0]}.csv") and
         os.path.exists(f"./comad/human_robot_data/bag_files/{bag_name}/{topic_names[1]}.csv")):
    ee_pose = br.message_by_topic(topic_names[0])
    human_forecast = br.message_by_topic(topic_names[1])
    pose = pd.read_csv(ee_pose)
    human = pd.read_csv(human_forecast)
else:
    pose = pd.read_csv(f"./comad/human_robot_data/bag_files/{bag_name}/{topic_names[0]}.csv")
    human = pd.read_csv(f"./comad/human_robot_data/bag_files/{bag_name}/{topic_names[1]}.csv")
print("postload")

robot_frames = []
print(len(pose['header.seq']))
for i in range(len(pose['header.seq'])):
    eul = euler_from_quaternion(pose['pose.orientation.y'][i], pose['pose.orientation.z'][i], pose['pose.orientation.w'][i], pose['pose.orientation.x'][i])
    robot_frames.append([pose['pose.position.x'][i], pose['pose.position.y'][i], pose['pose.position.z'][i], 
                         eul[0], eul[1], eul[2]])
    #The pose input is wrong on charm_follow the xyzw is rolled to wxyz, so x=w, y=x, and so forth

atiksh_frames = []
kushal_frames = []
print(len(human['markers']))
for i in range(len(human['markers'])):
    human_str_arr = (human['markers'][i]).split(', ')
    atiksh_intermediary = []
    kushal_intermediary = []
    for j in range(len(human_str_arr)):
        if "Atiksh" not in human_str_arr[j] and "Kushal" not in human_str_arr[j]:
            continue
        start = human_str_arr[j].find("points:")
        end = human_str_arr[j].find("colors:")
        sub_str = human_str_arr[j][start:end]
        sub_arr = re.findall(pattern=digit_parser, string=sub_str)
        sub_arr = list(map(float, sub_arr))
        sub_arr_1 = [sub_arr[0], sub_arr[1], sub_arr[2]]
        sub_arr_2 = [sub_arr[3], sub_arr[4], sub_arr[5]]
        if "Atiksh" in human_str_arr[j]:
            atiksh_intermediary.append(sub_arr_1)
            atiksh_intermediary.append(sub_arr_2)
        elif "Kushal" in human_str_arr[j]:
            kushal_intermediary.append(sub_arr_1)
            kushal_intermediary.append(sub_arr_2)
    atiksh_frames.append(atiksh_intermediary)
    kushal_frames.append(kushal_intermediary)
print(len(atiksh_frames))
print(len(kushal_frames))
print(len(robot_frames))
input()
take_1_ep = [
    ("0:42", "0:52"),
    ("0:59", "1:10"),
    ("1:49", "2:03"),
    ("2:29", "2:52"),
    ("2:58", "3:14"),
    ("3:18", "3:32"),
    ("3:58", "4:13"),
    ("4:20", "4:33"),
    ("4:52", "5:04"),
    ("5:22", "5:36"),
    ("5:41", "5:54")
]

take_2_ep = [
    ("0:14", "0:27"),
    ("0:31", "0:45"),
    ("0:48", "1:01"),
    ("1:05", "1:17"),
    ("1:20", "1:32"),
    ("1:37", "1:51"),
    ("1:54", "2:07"),
    ("2:21", "2:32"),
    ("2:35", "2:47"),
    ("2:51", "3:04")
]

def convert_time_to_frame(time, hz, offset):
    mins = int(time[:time.find(':')])
    secs = int(time[time.find(':')+1:])
    return int(((mins*60 + secs) * hz) + offset * hz)

clips = []
for start, end in take_2_ep:
    start_frame = convert_time_to_frame(start, 19, 0)
    end_frame = convert_time_to_frame(end, 19, 0) #+ 240  #The +240 frames here is essentially adding 2 extra seconds to the episode

    atiksh_arr = np.nan_to_num(atiksh_frames[start_frame:end_frame]).tolist()
    kushal_arr = np.nan_to_num(kushal_frames[start_frame:end_frame]).tolist()
    robot_arr = np.nan_to_num(robot_frames[start_frame:end_frame]).tolist()
    current_clip = {"Atiksh":atiksh_arr, "Kushal":kushal_arr, "Robot":robot_arr}
    clips.append(current_clip)

import json
for idx in range(len(clips)):
    with open(f'./comad/human_robot_data/bag_files/{bag_name}/{bag_name}_{idx}.json', 'w') as f:
        json.dump(clips[idx], f, indent=4)
