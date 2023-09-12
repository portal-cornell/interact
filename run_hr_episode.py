import rospy
import json
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np
from pynput import keyboard
import argparse
import os
import torch
import torch_dct
from MRT.Models import IntentInformedHRForecaster

device = 'cuda'
def get_model(ONE_HIST=False, CONDITIONAL=True, bob_hand = True, align_rep = True):
    bob_joints_list = list(range(9)) if not bob_hand else list(range(5,9))  
    robot_joints_list = [6,8]
    model = IntentInformedHRForecaster(d_word_vec=128, d_model=128, d_inner=1024,
                n_layers=3, n_head=8, d_k=64, d_v=64,
                device=device,
                conditional_forecaster=CONDITIONAL,
                bob_joints_list=bob_joints_list,
                bob_joints_num=len(bob_joints_list),
                one_hist=ONE_HIST,
                robot_joints_list=robot_joints_list,
                robot_joints_num=2,
                align_rep=align_rep).to(device)
    model_id = f'{"1hist" if ONE_HIST else "2hist"}_{"marginal" if not CONDITIONAL else "conditional"}'
    model_id += f'_{"withAMASS"}_{"handwrist" if bob_hand else "alljoints"}'
    model_id += '_ft'
    model_id += '_hr'
    model_id += '_noalign'
    directory = f'./checkpoints_new_arch_finetuned_hr_oriented/saved_model_{model_id}'
    model.load_state_dict(torch.load(f'{directory}/50.model', map_location=torch.device('cuda')))
    model.eval()
    return model

marginal_model = get_model(ONE_HIST=True, CONDITIONAL=False, bob_hand=False)
conditional_model = get_model(ONE_HIST=False, CONDITIONAL=True, bob_hand=False)
model_joints_idx = [0,1,2,3,4,5,6,9,10]

def get_history(joint_data, current_idx, history_length, skip_rate = int(15/15), relevant_joints=['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 
                        'WaistRBack', 'LHandOut', 'RHandOut']):
    history_joints = []
    for i in range(current_idx-(history_length-1)*skip_rate, current_idx+1, skip_rate):
        idx = max(0, i)
        history_joints.append(get_relevant_joints(joint_data[idx], relevant_joints=relevant_joints))
    return history_joints

def get_relevant_joints(all_joints, relevant_joints=['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 
                        'WaistRBack', 'LHandOut', 'RHandOut']):                   
    relevant_joint_pos = []
    for i, joint in enumerate(relevant_joints):
        pos = all_joints[i]
        relevant_joint_pos.append(pos)
    return relevant_joint_pos

def get_future(joint_data, current_idx, future_length=15, skip_rate = int(15/15), relevant_joints=['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 
                        'WaistRBack', 'LHandOut', 'RHandOut']):
    future_joints = []
    for i in range(current_idx+skip_rate, current_idx + future_length*skip_rate + 1, skip_rate):
        idx = min(i, len(joint_data)-1)
        future_joints.append(get_relevant_joints(joint_data[idx], relevant_joints=relevant_joints))
    return future_joints

def get_forecast(model, alice_hist_raw, bob_hist_raw, alice_future_raw, bob_future_raw, robot_hist_raw, robot_future_raw):
    alice_hist = torch.Tensor(np.array(alice_hist_raw)[:,model_joints_idx]).reshape(len(alice_hist_raw),-1).unsqueeze(0).to(device)
    bob_hist = torch.Tensor(np.array(bob_hist_raw)[:,model_joints_idx]).reshape(len(bob_hist_raw),-1).unsqueeze(0).to(device)

    # alice_future = torch.Tensor(np.array(alice_future_raw)[:,model_joints_idx]).reshape(len(alice_future_raw),-1).unsqueeze(0)
    bob_future = torch.Tensor(np.array(bob_future_raw)[:,model_joints_idx]).reshape(len(bob_future_raw),-1).unsqueeze(0).to(device)

    ### TODO: Turn robot data into tensors and subtract offset, then pass into forward pass
    robot_hist = torch.Tensor(np.array(robot_hist_raw)[:,:]).reshape(len(robot_hist_raw),-1).unsqueeze(0).to(device)
    robot_future = torch.Tensor(np.array(robot_future_raw)[:,:]).reshape(len(robot_future_raw),-1).unsqueeze(0).to(device)

    offset = alice_hist[:, -1, :].unsqueeze(1)
    alice_hist = alice_hist-offset
    bob_hist = bob_hist-offset
    bob_future = bob_future-offset
    robot_hist = robot_hist-(offset[:,:,-6:])
    robot_future = robot_future-(offset[:,:,-6:])
    with torch.no_grad():
        results, _ = model(alice_hist, bob_hist, bob_future, robot_hist, robot_future)
    # import pdb; pdb.set_trace()
    # print(results.shape)
    results = results + offset
    alice_future_raw = np.array(alice_future_raw)
    alice_future_raw = fix_orientation(alice_future_raw)
    alice_hist_raw = np.array(alice_hist_raw)
    alice_hist_raw = fix_orientation(alice_hist_raw)
    alice_future_raw[:,model_joints_idx,:] = results.cpu().numpy().reshape(15, 9, 3)
    # import pdb; pdb.set_trace()
    alice_future_raw[:, 7:9, :] = alice_hist_raw[-1:, 7:9, :]
    return alice_future_raw[:,:,:]

def get_marker(id, pose, edge, ns = 'current', alpha=1, red=1, green=1, blue=1):
    relevant_joints=['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 
                        'WaistRBack', 'LHandOut', 'RHandOut']

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

    p1.x, p1.y, p1.z = -x, z, y
    # p1.x, p1.y, p1.z = x, y, z
    x, y, z = pos2.tolist()
    p2.x, p2.y, p2.z = -x, z, y
    # p2.x, p2.y, p2.z = x, y, z

    p1m.points = [p1, p2]
    marker.points = [p1, p2]
    return marker,p1m


def get_marker_array(current_joints, future_joints, marginal_forecast_joints, conditional_forecast_joints, person = "Kushal"):
    id_offset = 100000 if person == 'Kushal' else 0
    color = 1 if person == "Kushal" else 0
    marker_array = MarkerArray()
    edges = [
            (0, 1), (0, 2),
            (1, 3), (3, 5),
            (2, 4), (4, 6),
            (5, 9), (6, 10)
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

    if marginal_forecast_joints is not None:
        for idx, edge in enumerate(edges):
            for timestep in [15]:
                tup = get_marker(idx+100*timestep, marginal_forecast_joints[timestep-1], edge,ns=f'marginal_forecast', alpha=1, 
                                red=1.0, 
                                green=0.1, 
                                blue=0.1)
                marker_array.markers.append(tup[0])
                marker_array.markers.append(tup[1])

    if conditional_forecast_joints is not None:
        for idx, edge in enumerate(edges):
            for timestep in [15]:
                tup = get_marker(idx+10000*timestep, conditional_forecast_joints[timestep-1], edge,ns=f'conditional_forecast', alpha=1, 
                                red=0.1, 
                                green=0.1, 
                                blue=1.0)
                marker_array.markers.append(tup[0])
                marker_array.markers.append(tup[1])
    r_hand_edges = [(6, 10)]
    if future_joints is not None:
        for idx, edge in enumerate(r_hand_edges):
            for timestep in [15]:
                tup = get_marker(idx+1000000*timestep, future_joints[timestep-1], edge,ns=f'conditional_forecast', alpha=2, 
                                red=0.1, 
                                green=0.8, 
                                blue=0.1)
                marker_array.markers.append(tup[0])
                marker_array.markers.append(tup[1])

    return marker_array

def forecast_jumped(forecast, prev_forecast):
    # print(forecast.shape)
    return False

def fix_orientation(tensor):
    tensor[:,:,[0,1,2]] = tensor[:,:,[0,2,1]]
    tensor[:,:,0] *= -1
    return tensor

if __name__ == '__main__':
    rospy.init_node('forecaster', anonymous=True)
    human_A_forecast = rospy.Publisher("/alice_forecast", MarkerArray, queue_size=1)
    human_B_forecast = rospy.Publisher("/bob_forecast", MarkerArray, queue_size=1)

    parser = argparse.ArgumentParser(description='Arguments for running the scripts')
    parser.add_argument('--dataset',type=str,default="handover",help="Dataset Type")
    parser.add_argument('--set_num',type=str,default="0",help="Number of Dataset")
    parser.add_argument('--ep_num', type=str,default="-1",help="Episode to watch/leave blank if wanting to watch whole set")
    parser.add_argument('--ros_rate', type=int,default=600,help="Playback Speed")

    args = parser.parse_args()


    dataset_folder = f"./comad_json/test/"
    mapping_file = "./mapping/comad_mapping.json"

    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    pause = False
    def on_press(key):
        global pause
        if key == keyboard.Key.space:
            pause = True
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    rate = rospy.Rate(args.ros_rate)

    person_data = {}
    prev_forecast_joints_A = None
    if args.ep_num != "-1":
        while True:
            # episode_file = f"{dataset_folder}/{args.dataset}_{args.set_num}_{args.ep_num}.json"
            episode_file = f"hr/test_hr/take_1_2.json"
            with open(episode_file, 'r') as f:
                data = json.load(f)
            for stream_person in data:
                person_data[stream_person] = fix_orientation(np.array(data[stream_person]))
                print((person_data[stream_person]).shape)
            for timestep in range(len(data[list(data.keys())[0]])):
                print(round(timestep/120, 1))
                if not pause and listener.running:
                    joint_data_A = person_data["Atiksh"]
                    joint_data_B = person_data["Kushal"]
                    joint_data_R = person_data["Robot"]
                    current_joints_A = get_relevant_joints(joint_data_A[timestep])
                    current_joints_B = get_relevant_joints(joint_data_B[timestep])
                    current_joints_R = get_relevant_joints(joint_data_R[timestep], relevant_joints=['a','b'])

                    T_in = 15
                    T_out = 15

                    history_joints_A = get_history(joint_data_A, timestep, T_in)
                    history_joints_B = get_history(joint_data_B, timestep, T_in)
                    history_joints_R = get_history(joint_data_R, timestep, T_in, relevant_joints=['a','b'])

                    future_joints_A = get_future(joint_data_A, timestep, T_out)
                    future_joints_B = get_future(joint_data_B, timestep, T_out)
                    future_joints_R = get_future(joint_data_R, timestep, T_out, relevant_joints=['a','b'])

                    # marginal_forecast_joints_A = get_forecast(marginal_model, history_joints_A, history_joints_B, future_joints_A, future_joints_B, history_joints_R, future_joints_R)
                    marginal_forecast_joints_B = get_forecast(marginal_model, history_joints_B, history_joints_A, future_joints_B, future_joints_A, history_joints_R, future_joints_R)
                    
                    # conditional_forecast_joints_A = get_forecast(conditional_model, history_joints_A, history_joints_B, future_joints_A, future_joints_B, history_joints_R, future_joints_R)
                    conditional_forecast_joints_B = get_forecast(conditional_model, history_joints_B, history_joints_A, future_joints_B, future_joints_A, history_joints_R, future_joints_R)

                    # marker_array_A = get_marker_array(current_joints=current_joints_A, 
                    #                                 future_joints=future_joints_A,
                    #                                 marginal_forecast_joints=marginal_forecast_joints_A,
                    #                                 conditional_forecast_joints=conditional_forecast_joints_A,
                    #                                 person="Atiksh")
                    # marker_array_B = get_marker_array(current_joints=current_joints_B, 
                    #                                 future_joints=future_joints_B,
                    #                                 marginal_forecast_joints=marginal_forecast_joints_B,
                    #                                 conditional_forecast_joints=conditional_forecast_joints_B,
                    #                                 person="Kushal")
                    marker_array_B = get_marker_array(current_joints=current_joints_B, 
                                                    future_joints=None,
                                                    marginal_forecast_joints=None,
                                                    conditional_forecast_joints=None,
                                                    person="Kushal")
                                    
                    # human_A_forecast.publish(marker_array_A)
                    human_B_forecast.publish(marker_array_B)

                    # if forecast_jumped(forecast_joints_A, prev_forecast_joints_A):
                    #     pause = True
                    # prev_forecast_joints_A = forecast_joints_A
                    rate.sleep()
                else:
                    input("Press enter to continue")
                    pause = False
                    listener = keyboard.Listener(on_press=on_press)
                    listener.start()

