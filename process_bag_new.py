import rospy
# from quaternion import from_euler_angles, as_float_array
from visualization_msgs.msg import Marker, MarkerArray

from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys 
import json

# MOCAP_OFFSETS = [1.22+0.25, -1.3-1.1, -0.85+0.1]
def convert_time_to_frame(time, hz=15, offset=0):
    mins = int(time[:time.find(':')])
    secs = float(time[time.find(':')+1:])
    return int(((mins*60 + secs) * hz) + offset * hz)
# import pdb; pdb.set_trace()
bag_name = sys.argv[1]

with open(f'comad/human_robot_data/timestamps/{bag_name}.txt') as f: data = f.readlines()

start_end_times = [list(map(convert_time_to_frame, se.split('-'))) for se in data]

ATIKSH_OFFSETS = {
    'take_1': [1.45, -2.4, -0.75],
    'take_2': [1.45, -2.4, -0.75],
    'cabinet_1': [1.35, -0.4, -0.75],
    'cabinet_2': [1.35, -0.4, -0.75],
    'cart_1': [1.47, -2.4, -0.75],
    'cart_2': [1.47, -2.4, -0.75],
    'take_3': [1.47, -2.4, -0.75]
}

KUSHAL_OFFSETS = {
    'take_1': [1.22, -1.3, -0.85],
    'take_2': [1.22, -1.3, -0.85],
    'cabinet_1': [0.9, -1.3, -0.85],
    'cabinet_2': [0.9, -1.3, -0.85],
    'cart_1': [0.9, -1.3, -0.85],
    'cart_2': [0.9, -1.3, -0.85],
    'take_3': [0.9, -1.3, -0.85],
}

class BagReaderHR():
    def __init__(self):
        rospy.init_node('realworld', anonymous=True)
        ee_sub = rospy.Subscriber('/ee_curr_pose', MarkerArray, self.ee_callback, queue_size=1)
        human_forecast_subscriber = rospy.Subscriber('/human_forecast', MarkerArray, self.human_forecast_callback, queue_size=1)
        self.atiksh_pose = None
        self.kushal_pose = None
        self.ee_pose = None

    def human_forecast_callback(self, marker_array):
        self.atiksh_pose = {}
        self.kushal_pose = {}
        for marker in marker_array.markers:
            # print(marker.header.frame_id)
            p1, p2 = marker.points
            
            if 'Kushal' in marker.ns:
                x1 = p1.x + KUSHAL_OFFSETS[bag_name][0]
                y1 = p1.y + KUSHAL_OFFSETS[bag_name][1]
                z1 = p1.z + KUSHAL_OFFSETS[bag_name][2]
                x2 = p2.x + KUSHAL_OFFSETS[bag_name][0]
                y2 = p2.y + KUSHAL_OFFSETS[bag_name][1]
                z2 = p2.z + KUSHAL_OFFSETS[bag_name][2]
                left_edge, right_edge = marker.ns.split('-')[2].split('_')
                self.kushal_pose[left_edge] = np.array([x1, y1, z1])
                self.kushal_pose[right_edge] = np.array([x2, y2, z2])
            if 'Atiksh' in marker.ns:
                x1 = p1.x + ATIKSH_OFFSETS[bag_name][0]
                y1 = p1.y + ATIKSH_OFFSETS[bag_name][1]
                z1 = p1.z + ATIKSH_OFFSETS[bag_name][2]
                x2 = p2.x + ATIKSH_OFFSETS[bag_name][0]
                y2 = p2.y + ATIKSH_OFFSETS[bag_name][1]
                z2 = p2.z + ATIKSH_OFFSETS[bag_name][2]
                left_edge, right_edge = marker.ns.split('-')[2].split('_')
                p1, p2 = marker.points
                self.atiksh_pose[left_edge] = np.array([x1, y1, z1])
                self.atiksh_pose[right_edge] = np.array([x2, y2, z2])

    def get_marker_array(self, alpha=1, red=1, green=0, blue=0):
        pose = self.ee_pose
        SCALE = 0.015
        marker = Marker()
        marker.header.frame_id = "panda_link0"
        marker.header.stamp = rospy.Time.now()
        marker.type = marker.LINE_LIST
        marker.id = 0
        marker.scale.x = SCALE
        marker.action = marker.ADD 
        marker.ns = f'robot-ee'
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha
        p1m = Marker()
        p1m.header.frame_id = "panda_link0"
        p1m.header.stamp = rospy.Time.now()
        p1m.type = marker.SPHERE_LIST
        p1m.id = 1
        p1m.scale.x = SCALE*2
        p1m.scale.y = SCALE*2
        p1m.scale.z = SCALE*2
        p1m.action = p1m.ADD
        p1m.color.r = red
        p1m.color.g = green
        p1m.color.b = blue
        p1m.color.a = alpha/2
        pos1, pos2 = pose[0], pose[1]
        p1, p2 = Point(), Point()
        x, y, z = pos1
        p1.x, p1.y, p1.z = x, y, z
        x, y, z = pos2
        p2.x, p2.y, p2.z = x, y, z

        p1m.points = [p1]
        marker.points = [p1, p2]

        marker_array = MarkerArray()
        marker_array.markers = [marker, p1m]
        return marker_array
    
    def get_atiksh_rpy(self):
        wrist_pos = self.atiksh_pose['RWristOut']
        hand_pos = self.atiksh_pose['RHandOut']
        diff = hand_pos-wrist_pos
        dist = np.linalg.norm(diff)
        return np.arccos(diff[0]/dist), np.arccos(diff[1]/dist), np.arccos(diff[2]/dist)

    def ee_callback(self, markerArray, ee_length = 0.1):
        marker, p1m = markerArray.markers
        p1, p2 = marker.points
        self.ee_pose = [[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]]

ee_pub = rospy.Publisher('/ee_pub', MarkerArray)
br = BagReaderHR()
rate = rospy.Rate(1500)

robot_frames = []
kushal_frames = []
atiksh_frames = []

start = False
end = False

def get_relevant_joints(joint_dic):
    relevant_joints=['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 
                        'WaistRBack', 'LHandOut', 'RHandOut']
    joints = []
    for js in relevant_joints:
        joints.append(joint_dic[js])

    # import pdb; pdb.set_trace()
    return joints

for timestep in range(10*1500):
    print(f'Time = {timestep/15}')
    if br.ee_pose is not None:
        robot_ee = br.get_marker_array()
        ee_pub.publish(robot_ee)
    if br.atiksh_pose is not None:
        start = True
    if start:
        atiksh_frames.append(get_relevant_joints(br.atiksh_pose))
        kushal_frames.append(get_relevant_joints(br.kushal_pose))
        if br.ee_pose is None:
            robot_frames.append([])
        else:
            robot_frames.append(br.ee_pose)

    rate.sleep()

clips = []
task_name = bag_name.split('_')[0]
for start_frame, end_frame in start_end_times:
    atiksh_arr = np.nan_to_num(atiksh_frames).tolist()
    kushal_arr = np.nan_to_num(kushal_frames).tolist()
    robot_arr = np.nan_to_num(robot_frames).tolist()
    current_clip = {"Atiksh":atiksh_arr, "Kushal":kushal_arr, "Robot":robot_arr}
    clips.append(current_clip)

with open(f'./comad/human_robot_data/{task_name}_jsons/{bag_name}.json', 'w') as f:
    json.dump(clips[0], f, indent=4)
clips = []
for start_frame, end_frame in start_end_times:
    atiksh_arr = np.nan_to_num(atiksh_frames[start_frame:end_frame]).tolist()
    kushal_arr = np.nan_to_num(kushal_frames[start_frame:end_frame]).tolist()
    robot_arr = np.nan_to_num(robot_frames[start_frame:end_frame]).tolist()
    current_clip = {"Atiksh":atiksh_arr, "Kushal":kushal_arr, "Robot":robot_arr}
    clips.append(current_clip)



for idx in range(len(clips)):
    with open(f'./comad/human_robot_data/jsons/{bag_name}_{idx}.json', 'w') as f:
        json.dump(clips[idx], f, indent=4)