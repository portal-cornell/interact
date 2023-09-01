import rospy
import json
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np
from pynput import keyboard
import argparse

with open('amass_mapping.json', 'r') as f:
        mapping = json.load(f)

relevant_joints=['Neck', 'L_Shoulder', 'R_Shoulder',
                        'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 
                        'L_Hip', 'R_Hip', 'L_Hand', 'R_Hand']

def get_relevant_joints(all_joints):                       
    relevant_joint_pos = []
    for joint in relevant_joints:
        pos = all_joints[mapping[joint]]
        relevant_joint_pos.append(pos)
    return relevant_joint_pos


def get_marker(id, pose, edge, ns = 'current', alpha=1, red=1, green=1, blue=1):
    # relevant_joints=['BackTop', 'LShoulderBack', 'RShoulderBack',
    #                     'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 
    #                     'WaistRBack', 'LHandOut', 'RHandOut']

    SCALE = 0.015
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.type = marker.LINE_LIST
    marker.id = id
    marker.scale.x = SCALE
    marker.action = marker.ADD 
    marker.ns = f'{ns}-{relevant_joints[edge[0]]}_{relevant_joints[edge[1]]}'
    marker.color.r = red
    marker.color.g = green
    marker.color.b = blue
    marker.color.a = alpha
    p1m = Marker()
    p1m.header.frame_id = "map"
    p1m.header.stamp = rospy.Time.now()
    p1m.type = marker.SPHERE_LIST
    p1m.id = id + 101
    p1m.scale.x = SCALE*2
    p1m.scale.y = SCALE*2
    p1m.scale.z = SCALE*2
    p1m.action = p1m.ADD
    p1m.color.r = red
    p1m.color.g = green
    p1m.color.b = blue
    p1m.color.a = alpha/2
    pos1, pos2 = pose[edge[0]], pose[edge[1]]
    p1, p2 = Point(), Point()
    x, y, z = pos1.tolist()
    #for forward positioned mocap
    p1.x, p1.y, p1.z = -x, z, y
    x, y, z = pos2.tolist()
    p2.x, p2.y, p2.z = -x, z, y

    # if edge[0] == 1 and edge[1] == 2:
    #     print(((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[1]-pos2[2])**2)**.5) 
    #for sideways positioned mocap
    # p1.x, p1.y, p1.z = -x-0.2, z-0.6, y+0.1
    # x, y, z = pos2.tolist()
    # p2.x, p2.y, p2.z = -x-0.2, z-0.6, y+0.1

    p1m.points = [p1, p2]
    marker.points = [p1, p2]
    return marker,p1m


def get_marker_array(current_joints, future_joints, forecast_joints, person = "Kushal"):
    id_offset = 100000 if person == 'Kushal' else 0
    color = 1 if person == "Kushal" else 0
    marker_array = MarkerArray()
    edges = [
            (0, 1), (0, 2),
            (1, 3), (3, 5),
            (2, 4), (4, 6),
            (5,9), (6,10)
        ]
    # extra edges to connect the pose back to the hips
    extra_edges = [(1, 7), (7, 8), (8, 2)]

    for idx, edge in enumerate(edges + extra_edges):
        tup = get_marker(idx, current_joints, edge,ns=f'current', alpha=1, 
                         red=0.1, 
                         green=0.1, 
                         blue=0.0)
        marker_array.markers.append(tup[0])
        marker_array.markers.append(tup[1])

    return marker_array



if __name__ == '__main__':
    rospy.init_node('forecaster')
    human_A_forecast = rospy.Publisher("/alice_forecast", MarkerArray, queue_size=1)
    human_B_forecast = rospy.Publisher("/bob_forecast", MarkerArray, queue_size=1)

    parser = argparse.ArgumentParser(description='Arguments for running the scripts')
    parser.add_argument('--ep_num',type=str,default="0",help="episode number of their interaction")
    args = parser.parse_args()

    episode_file = f"./synthetic_amass/{args.ep_num}.json"
    with open(episode_file, 'r') as f:
        data = json.load(f)

    pause = False
    def on_press(key):
        if key == keyboard.Key.space:
            pause = True
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    rate = rospy.Rate(15)

    joint_data_A = np.array(data["alice"])
    joint_data_B = np.array(data["bob"])

    # for i in range(100):
    for timestep in range(len(data[list(data.keys())[0]])):
        # print(round(timestep/120, 1))
        if not pause and listener.running:
            
            current_joints_A = get_relevant_joints(joint_data_A[timestep])
            current_joints_B = get_relevant_joints(joint_data_B[timestep])
            marker_array_A = get_marker_array(current_joints=current_joints_A, 
                                            future_joints=None,
                                            forecast_joints=None,
                                            person="Atiksh")
            marker_array_B = get_marker_array(current_joints=current_joints_B, 
                            future_joints=None,
                            forecast_joints=None,
                            person="Kushal")
                            
            human_A_forecast.publish(marker_array_A)
            human_B_forecast.publish(marker_array_B)
            rate.sleep()
        else:
            input("Press enter to continue")
            pause = False
            listener = keyboard.Listener(on_press=on_press)
            listener.start()

