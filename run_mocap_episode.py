from amc_parser import *
import argparse
import rospy
import json
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np
from pynput import keyboard
import torch
import torch_dct as dct
from MRT.Models import Transformer, ConditionalForecaster

ONE_HIST = False
CONDITIONAL = False
device = 'cpu'
model = ConditionalForecaster(d_word_vec=128, d_model=128, d_inner=1024,
            n_layers=3, n_head=8, d_k=64, d_v=64,device=device,conditional_forecaster=CONDITIONAL)
model_id = f'{"1hist" if ONE_HIST else "2hist"}_{"marginal" if not CONDITIONAL else "conditional"}'
directory = f'./saved_model_{model_id}'
model.load_state_dict(torch.load(f'{directory}/2.model', map_location=torch.device('cpu')))
model.eval()
with open('mapping/mocap_mapping.json', 'r') as f:
        mapping = json.load(f)

relevant_joints=['lowerneck', 'lclavicle', 'rclavicle',
                        'lradius', 'rradius', 'lwrist', 'rwrist', 'lhipjoint', 
                        'rhipjoint', 'lhand', 'rhand']
model_joints_idx = [0,1,2,3,4,5,6,9,10]

def get_motion_list(joints, motions):
    motion_list = []
    for frame_idx in range(len(motions)):
        joints['root'].set_motion(motions[frame_idx])
        joints_list=[]
        for joint in joints.values():
            xyz=np.array([joint.coordinate[0],\
                joint.coordinate[1],joint.coordinate[2]]).squeeze(1)
            joints_list.append(xyz)
        motion_list.append(np.array(joints_list))
    return motion_list

def get_relevant_joints(all_joints, scale = 0.056444):                       
    relevant_joint_pos = []
    for joint in relevant_joints:
        pos = all_joints[mapping[joint]]
        relevant_joint_pos.append(pos*scale)
    return relevant_joint_pos

def get_history(joint_data, current_idx, history_length, skip_rate = int(120/15)):
    history_joints = []
    for i in range(current_idx-(history_length-1)*skip_rate, current_idx+1, skip_rate):
        idx = max(0, i)
        history_joints.append(get_relevant_joints(joint_data[idx]))
    return history_joints

def get_future(joint_data, current_idx, future_length=15, skip_rate = int(120/15)):
    future_joints = []
    for i in range(current_idx+skip_rate, current_idx + future_length*skip_rate + 1, skip_rate):
        idx = min(i, len(joint_data)-1)
        future_joints.append(get_relevant_joints(joint_data[idx]))
    return future_joints

def get_forecast(alice_hist_raw, bob_hist_raw, alice_future_raw, bob_future_raw):
    alice_hist = torch.Tensor(np.array(alice_hist_raw)[:,model_joints_idx]).reshape(len(alice_hist_raw),-1).unsqueeze(0)
    bob_hist = torch.Tensor(np.array(bob_hist_raw)[:,model_joints_idx]).reshape(len(bob_hist_raw),-1).unsqueeze(0)

    alice_future = torch.Tensor(np.array(alice_future_raw)[:,model_joints_idx]).reshape(len(alice_future_raw),-1).unsqueeze(0)
    bob_future = torch.Tensor(np.array(bob_future_raw)[:,model_joints_idx]).reshape(len(bob_future_raw),-1).unsqueeze(0)

    offset = alice_hist[:, -1, :].unsqueeze(1)
    alice_hist = alice_hist-offset
    bob_hist = bob_hist-offset
    bob_future = bob_future-offset
    with torch.no_grad():
        results = model(alice_hist, bob_hist, bob_future, one_hist=ONE_HIST, cond_future=CONDITIONAL)
    # import pdb; pdb.set_trace()
    results = results + offset
    alice_future_raw = np.array(alice_future_raw)
    alice_hist_raw = np.array(alice_hist_raw)
    alice_future_raw[:,model_joints_idx,:] = results.reshape(15, 9, 3)
    # import pdb; pdb.set_trace()
    alice_future_raw[:, 7:9, :] = alice_hist_raw[-1:, 7:9, :]
    return alice_future_raw[:,:,:]



def get_marker(id, pose, edge, ns = 'current', alpha=1, red=1, green=1, blue=1):
    

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
    p1.x, p1.y, p1.z = -z+0.4, -x+1.3, y+0.1
    x, y, z = pos2.tolist()
    p2.x, p2.y, p2.z = -z+0.4, -x+1.3, y+0.1

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
    # import pdb; pdb.set_trace()
    for idx, edge in enumerate(edges):
        tup = get_marker(idx+100000, forecast_joints[-1], edge,ns=f'forecast', alpha=1, 
                         red=0.1, 
                         green=0.1, 
                         blue=1.0)
        marker_array.markers.append(tup[0])
        marker_array.markers.append(tup[1])

    return marker_array

if __name__ == "__main__":
    rospy.init_node('forecaster')
    human_A_forecast = rospy.Publisher("/alice_forecast", MarkerArray, queue_size=1)
    human_B_forecast = rospy.Publisher("/bob_forecast", MarkerArray, queue_size=1)

    parser = argparse.ArgumentParser(description='Arguments for running the scripts')
    
    ## 18-19, 20-21, 22-23, 33-34
    parser.add_argument('--alice',type=str,default="18",help="Alice subject number")
    parser.add_argument('--bob',type=str,default="19",help="Bob subject number")
    
    parser.add_argument('--ep_num',type=str,default="01",help="episode number of their interaction")

    args = parser.parse_args()
    A, B = args.alice, args.bob
    ep_num = args.ep_num

    asf_path_A = './mocap/all_asfamc/subjects/'+A+'/'+A+'.asf'
    amc_path_A = './mocap/all_asfamc/subjects/'+A+'/'+A+'_'+ep_num+'.amc'

    asf_path_B = './mocap/all_asfamc/subjects/'+B+'/'+B+'.asf'
    amc_path_B = './mocap/all_asfamc/subjects/'+B+'/'+B+'_'+ep_num+'.amc'

    joints_A = parse_asf(asf_path_A)
    motions_A = parse_amc(amc_path_A)

    joints_B = parse_asf(asf_path_B)
    motions_B = parse_amc(amc_path_B)

    joint_data_A = get_motion_list(joints_A, motions_A)
    joint_data_B = get_motion_list(joints_B, motions_B)

    pause = False
    def on_press(key):
        if key == keyboard.Key.space:
            pause = True
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    rate = rospy.Rate(120)
    import time
    prev_time = time.time()
    for timestep in range(len(joint_data_B)):
        # print(round(timestep/120, 1))
        # print(time.time()-prev_time)
        prev_time = time.time()
        if not pause and listener.running:
            current_joints_A = get_relevant_joints(joint_data_A[timestep])
            current_joints_B = get_relevant_joints(joint_data_B[timestep])
            
            T_in = 15
            T_out = 15
            history_joints_A = get_history(joint_data_A, timestep, T_in)
            history_joints_B = get_history(joint_data_B, timestep, T_in)
            future_joints_A = get_future(joint_data_A, timestep, T_out)
            future_joints_B = get_future(joint_data_B, timestep, T_out)
            forecast_joints_A = None
            forecast_joints_B = None
            forecast_joints_A = get_forecast(history_joints_A, history_joints_B, future_joints_A, future_joints_B)
            forecast_joints_B = get_forecast(history_joints_B, history_joints_A, future_joints_B, future_joints_A)

            marker_array_A = get_marker_array(current_joints=current_joints_A, 
                                            future_joints=future_joints_A,
                                            forecast_joints=forecast_joints_A,
                                            person="Atiksh")
            marker_array_B = get_marker_array(current_joints=current_joints_B, 
                            future_joints=future_joints_B,
                            forecast_joints=forecast_joints_B,
                            person="Kushal")
            
            human_A_forecast.publish(marker_array_A)
            human_B_forecast.publish(marker_array_B)
            rate.sleep()
        else:
            input("Press enter to continue")
            pause = False
            listener = keyboard.Listener(on_press=on_press)
            listener.start()

# import pdb; pdb.set_trace()

