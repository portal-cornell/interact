import bagpy
from bagpy import bagreader
import pandas as pd
import os

bag_name = "./comad/human_robot_data/bag_files/take_2.bag"
topic_names = ["/ee_curr_pose", "/human_forecast"]

csv_destination = f"./comad/{bag_name}/{bag_name}"

br = bagreader(bag_name)

ee_pose = br.message_by_topic(topic_names[0])
human_forecast = br.message_by_topic(topic_names[1])

pose = pd.read_csv(ee_pose)
human = pd.read_csv(human_forecast)

hr_data_ep1 = []

