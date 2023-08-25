import torch.utils.data as data
import torch
import numpy as np

import sys



from utils.read_json_data import read_json, get_pose_history, missing_data
from utils.ang2joint import *
import networkx as nx
import os


class DATA(data.Dataset):
    def __init__(self):
        
        self.data=np.load('./mocap/train_3_120_mocap.npy',allow_pickle=True)
        
        self.len=len(self.data)
        
        # if joints==15:
        #     use=[0,1,2,3,6,7,8,14,16,17,18,20,24,25,27]
        #     self.data=self.data.reshape(self.data.shape[0],3,-1,31,3)
        #     self.data=self.data[:,:,:,use,:]
        #     self.data=self.data.reshape(self.data.shape[0],3,-1,45)

            
    def __getitem__(self, index):
        
        input_seq=self.data[index][:,:30,:][:,::2,:]#input, 30 fps to 15 fps
        output_seq=self.data[index][:,30:,:][:,::2,:]#output, 30 fps to 15 fps
        last_input=input_seq[:,-1:,:]
        output_seq=np.concatenate([last_input,output_seq],axis=1)

        human_input = input_seq[0,:,:]
        human_output = output_seq[0,:,:]
        
        robot_input = input_seq[1,:,:]
        robot_ouput = output_seq[1,:,:]



        return input_seq,output_seq
        
        
        
    def __len__(self):
        return self.len



class TESTDATA(data.Dataset):
    def __init__(self,dataset='mocap'):
        
        if dataset=='mocap':
            self.data=np.load('./mocap/test_3_120_mocap.npy',allow_pickle=True)
            
        
            # use=[0,1,2,3,6,7,8,14,16,17,18,20,24,25,27]
            # self.data=self.data
            # self.data=self.data.reshape(self.data.shape[0],self.data.shape[1],-1,31,3)
            # self.data=self.data[:,:,:,use,:]
            # self.data=self.data.reshape(self.data.shape[0],self.data.shape[1],-1,45)
        
        if dataset=='mupots':
            self.data=np.load('./mupots3d/mupots_120_3persons.npy',allow_pickle=True)

        self.len=len(self.data)

    def __getitem__(self, index):
        #split into robot/human data
        input_seq=self.data[index][:,:30,:][:,::2,:]#input, 30 fps to 15 fps
        output_seq=self.data[index][:,30:,:][:,::2,:]#output, 30 fps to 15 fps
        last_input=input_seq[:,-1:,:]
        output_seq=np.concatenate([last_input,output_seq],axis=1)

        return input_seq,output_seq

    def __len__(self):
        return self.len


hh_splits = [['./comad/table_set_1/', './comad/table_set_2'],
                  ['./comad/table_set_3/', './comad/table_set_4']]
hr_splits = [[]]
default_names = ["Atiksh", "Kushal"]
hr_names = ["Robot", "Kushal"]
data_types = ["Human-Human", "Human-Robot"]

class COMADDATA(data.Dataset):
    def __init__(self,data_dir,input_n,output_n,sample_rate, split=0, names=default_names, data_type=0):
            """
            data_dir := './comad'
            mapping_file := './mapping.json'
            """
            self.data_dir = data_dir
            self.input_frames = input_n 
            self.output_frames = output_n 
            self.sample_rate = sample_rate
            self.split = split
            self.robot_lst = []
            self.human_lst = []
            sequence_len = input_n + output_n
            joint_names = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 
                        'WaistRBack', 'LHandOut', 'RHandOut']
            mapping = read_json('./mapping.json')
            joint_used = np.array([mapping[joint_name] for joint_name in joint_names])
            mocap_splits = []
            if data_type == 0:
                mocap_splits = hh_splits
            else:
                mocap_splits = hr_splits

            #ignore atiksh's jsons when loading hr-data
            ignore_data = {
                "Atiksh":[],
                "Kushal":[],
                "Robot":[]
            }
            if data_types[data_type] == "Human-Human":
                missing_cnt = 0
                for ds in mocap_splits[split]:
                    print(f'>>> loading {ds}')
                    for episode in os.listdir(self.data_dir + '/' + ds):
                        print(f'Episode: {self.data_dir}/{ds}/{episode}')
                        json_data = read_json(f'{self.data_dir}/{ds}/{episode}')
                        for robot_name in names:
                            human_name = [name for name in names if name != robot_name][0]
                            if episode in ignore_data[robot_name]:
                                print('Ignoring for ' + robot_name)
                                continue
                            robot = get_pose_history(json_data, robot_name)
                            human = get_pose_history(json_data, human_name)
                            robot_frames = self.get_downsampled_frames(robot, 120)
                            human_frames = self.get_downsampled_frames(human, 120)
                            for start_frame in range(robot_frames.shape[0]-sequence_len):
                                end_frame = start_frame + sequence_len
                                if missing_data(robot_frames[start_frame:end_frame, joint_used, :]) or\
                                missing_data(human_frames[start_frame+input_n:end_frame, joint_used, :]):
                                    missing_cnt += 1
                                    continue
                                self.robot_lst.append(robot_frames[start_frame:end_frame, :, :])
                                self.human_lst.append(human_frames[start_frame+input_n:end_frame, :, :])
            else:
                missing_cnt = 0
                names = hr_names
                for ds in mocap_splits[split]:
                    print(f'>>> loading {ds}')
                    for episode in os.listdir(self.data_dir + '/' + ds):
                        print(f'Episode: {self.data_dir}/{ds}/{episode}')
                        json_data = read_json(f'{self.data_dir}/{ds}/{episode}')
                        robot_name = names[0]
                        human_name = names[1]
                        if episode in ignore_data[robot_name]:
                            print('Ignoring for ' + robot_name)
                            continue
                        robot = get_pose_history(json_data, robot_name)
                        human = get_pose_history(json_data, human_name)
                        robot_frames = self.get_downsampled_frames(robot, 19)
                        human_frames = self.get_downsampled_frames(human, 19)
                        for start_frame in range(robot_frames.shape[0]-sequence_len):
                            end_frame = start_frame + sequence_len
                            if missing_data(robot_frames[start_frame:end_frame, joint_used, :]) or\
                            missing_data(human_frames[start_frame+input_n:end_frame, joint_used, :]):
                                missing_cnt += 1
                                continue
                            self.robot_lst.append(robot_frames[start_frame:end_frame, :, :])
                            self.human_lst.append(human_frames[start_frame+input_n:end_frame, :, :])
            for idx, seq in enumerate(self.robot_lst):
                self.robot_lst[idx] = seq[:, :, :] - seq[input_n-1:input_n, 21:22, :]
            print(len(self.robot_lst))
            print(f'Missing: {missing_cnt}')

    def get_downsampled_frames(self, tensor, orig_framerate):
            orig_frames = tensor.shape[0]
            downsampled_frames = int(round((orig_frames/orig_framerate)*self.sample_rate))
            sample_idxs = np.linspace(0, orig_frames-1, downsampled_frames)
            select_frames = np.round(sample_idxs).astype(int)
            skipped_frames = tensor[select_frames]
            return skipped_frames

    def __len__(self):
        return len(self.robot_lst)

    def __getitem__(self, idx):
        # each element of the data list is of shape (sequence length, 25 joints, 3d)
        return self.robot_lst[idx], self.human_lst[idx] #what format needed for data loading
    
    
class AMASSDATA(data.Dataset):
    def __init__(self,data_dir,input_n,output_n,skip_rate, actions=None, split=0):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir,'amass/')         #  "D:\data\AMASS\\"
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        # self.sample_rate = opt.sample_rate
        self.p3d_human = []
        self.p3d_robot = []
        self.keys_human = []
        self.keys_robot = []
        self.data_idx_human = []
        self.data_idx_robot = []
        self.joint_used = np.arange(4, 22) # start from 4 for 17 joints, removing the non moving ones
        seq_len = self.in_n + self.out_n
        # mine below
        amass_splits = [
            ['CMU', 'MPI_Limits',],
            ['SFU'],
            ['BioMotionLab_NTroje'],
        ]


        # original splits below
        # amass_splits = [
        #     ['CMU', 'MPI_Limits', 'Eyes_Japan_Dataset', 'KIT', 'EKUT', 'ACCAD'],
        #     ['SFU',],
        #     ['BioMotionLab_NTroje'],
        # ]
        amass_splits = [['CMU'],['ACCAD']]
        print(amass_splits[split])
        # amass_splits = [
        #     ['SFU'],
        #     ['SFU',],
        #     ['SFU'],
        # ]


        # # original splits below
        # amass_splits = [
        #     ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'EKUT', 'TCD_handMocap', 'ACCAD'],
        #     ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        #     ['BioMotionLab_NTroje'],
        # ]
        # amass_splits = [['BioMotionLab_NTroje'], ['HumanEva'], ['SSM_synced']]
        # amass_splits = [['HumanEva'], ['HumanEva'], ['HumanEva']]
        # amass_splits[0] = list(
        #     set(amass_splits[0]).difference(set(amass_splits[1] + amass_splits[2])))


        # from human_body_prior.body_model.body_model import BodyModel
        # from smplx import lbs
        # root_path = os.path.dirname(__file__)
        # bm_path = root_path[:-6] + '/body_models/smplh/neutral/model.npz'
        # bm = BodyModel(bm_path=bm_path, num_betas=16, batch_size=1, model_type='smplh')
        # beta_mean = np.array([0.41771687, 0.25984767, 0.20500051, 0.13503872, 0.25965645, -2.10198147, -0.11915666,
        #                       -0.5498772, 0.30885323, 1.4813145, -0.60987528, 1.42565269, 2.45862726, 0.23001716,
        #                       -0.64180912, 0.30231911])
        # beta_mean = torch.from_numpy(beta_mean).unsqueeze(0).float()
        # # Add shape contribution
        # v_shaped = bm.v_template + lbs.blend_shapes(beta_mean, bm.shapedirs)
        # # Get the joints
        # # NxJx3 array
        # p3d0 = lbs.vertices2joints(bm.J_regressor, v_shaped)  # [1,52,3]
        # p3d0 = (p3d0 - p3d0[:, 0:1, :]).float().cuda().cpu().data.numpy()
        # parents = bm.kintree_table.data.numpy()[0, :]
        # np.savez_compressed('smpl_skeleton.npz', p3d0=p3d0, parents=parents)

        # load mean skeleton
        skel = np.load('./body_models/smpl_skeleton.npz')
        p3d0 = torch.from_numpy(skel['p3d0']).float()
        parents = skel['parents']
        parent = {}
        for i in range(len(parents)):
            parent[i] = parents[i]
        n = 0
        # path = self.path_to_data
        # dir_list = os.listdir(path)
        
        # print("Files and directories in '", path, "' :") 
        
        # # print the list
        # print(dir_list)
        for ds in amass_splits[split]:
            print("here")
            if not os.path.isdir(self.path_to_data + ds + '/'):
                print(self.path_to_data + ds + '/')
                print(ds)
                continue
            print('>>> loading {}'.format(ds))
            for sub in os.listdir(self.path_to_data + ds):
                if not os.path.isdir(self.path_to_data + ds + '/' + sub):
                    continue
                acts = []
                for act in os.listdir(self.path_to_data + ds + '/' + sub):
                    acts.append(act)
                    if not act.endswith('.npz') or len(acts) != 2:
                        continue
                    # if not ('walk' in act or 'jog' in act or 'run' in act or 'treadmill' in act):
                    #     continue
                    for i in range(2):
                        human = ""
                        robot = ""
                        if i == 0:
                            human = acts[0]
                            robot = acts[1]
                        else:
                            human = acts[1]
                            robot = acts[0]
                        human_pose_all = np.load(self.path_to_data + ds + '/' + sub + '/' + human)
                        robot_pose_all = np.load(self.path_to_data + ds + '/' + sub + '/' + robot)
                        try:
                            human_poses = human_pose_all['poses']
                            robot_poses =robot_pose_all['poses']
                        except:
                            print('no poses at {}_{}_{}'.format(ds, sub, act))
                            continue
                        human_frame_rate = human_pose_all['mocap_framerate']
                        robot_frame_rate = robot_pose_all['mocap_framerate']
                        # gender = pose_all['gender']
                        # dmpls = pose_all['dmpls']
                        # betas = pose_all['betas']
                        # trans = pose_all['trans']
                        human_fn = human_poses.shape[0]
                        robot_fn = robot_poses.shape[0]
                        human_sample_rate = int(human_frame_rate // 25)
                        robot_sample_rate = int(robot_frame_rate // 25)
                        human_fidxs = range(0, human_fn, human_sample_rate)
                        robot_fidxs = range(0, robot_fn, robot_sample_rate)


                        human_fn = len(human_fidxs)
                        human_poses = human_poses[human_fidxs]
                        human_poses = torch.from_numpy(human_poses).float()
                        human_poses = human_poses.reshape([human_fn, -1, 3])


                        robot_fn = len(robot_fidxs)
                        robot_poses = robot_poses[robot_fidxs]
                        robot_poses = torch.from_numpy(robot_poses).float()
                        robot_poses = robot_poses.reshape([robot_fn, -1, 3])
                        # remove global rotation
                        human_poses[:, 0] = 0
                        p3d0_tmp = p3d0.repeat([human_fn, 1, 1])
                        p3d_human = ang2joint(p3d0_tmp, human_poses, parent)
                        p3d0_tmp = p3d0.repeat([robot_fn, 1, 1])
                        p3d_robot = ang2joint(p3d0_tmp, robot_poses, parent)
                        # self.p3d[(ds, sub, act)] https://amass.is.tue.mpg.de/download.php= p3d.cpu().data.numpy()
                        self.p3d_human.append(p3d_human.cpu().data.numpy())
                        self.p3d_robot.append(p3d_robot.cpu().data.numpy())
                        if split == 2:
                            human_valid_frames = np.arange(0, human_fn - seq_len + 1, skip_rate)
                            robot_valid_frames = np.arange(0, robot_fn - seq_len + 1, skip_rate)
                        else:
                            human_valid_frames = np.arange(0, human_fn - seq_len + 1, skip_rate)
                            robot_valid_frames = np.arange(0, robot_fn - seq_len + 1, skip_rate)
                        # tmp_data_idx_1 = [(ds, sub, act)] * len(valid_frames)
                        self.keys_human.append((ds, sub, human))
                        self.keys_robot.append((ds, sub, robot))
                        tmp_data_idx_1 = [n] * len(human_valid_frames)
                        tmp_data_idx_2 = list(human_valid_frames)
                        self.data_idx_human.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        tmp_data_idx_1 = [n] * len(robot_valid_frames)
                        tmp_data_idx_2 = list(robot_valid_frames)
                        self.data_idx_robot.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    n += 1
    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        hkey, hstart_frame = self.data_idx_human[item]
        hfs = np.arange(hstart_frame, hstart_frame + self.in_n + self.out_n)
        rkey, rstart_frame = self.data_idx_robot[item]
        rfs = np.arange(rstart_frame, rstart_frame + self.in_n + self.out_n)
        return self.p3d_human[hkey][hfs], self.p3d_robot[rkey][rfs]  # , key


# In[12]:


def normalize_A(A): # given an adj.matrix, normalize it by multiplying left and right with the degree matrix, in the -1/2 power
        
        A=A+np.eye(A.shape[0])
        
        D=np.sum(A,axis=0)
        
        
        D=np.diag(D.A1)

        
        D_inv = D**-0.5
        D_inv[D_inv==np.infty]=0
        
        return D_inv*A*D_inv


# In[ ]:


def spatio_temporal_graph(joints_to_consider,temporal_kernel_size,spatial_adjacency_matrix): # given a normalized spatial adj.matrix,creates a spatio-temporal adj.matrix

    
    number_of_joints=joints_to_consider

    spatio_temporal_adj=np.zeros((temporal_kernel_size,number_of_joints,number_of_joints))
    for t in range(temporal_kernel_size):
        for i in range(number_of_joints):
            spatio_temporal_adj[t,i,i]=1 # create edge between same body joint,for t consecutive frames
            for j in range(number_of_joints):
                if spatial_adjacency_matrix[i,j]!=0: # if the body joints are connected
                    spatio_temporal_adj[t,i,j]=spatial_adjacency_matrix[i,j]
    return spatio_temporal_adj


# In[20]:


def get_adj_AMASS(joints_to_consider,temporal_kernel_size): # returns adj.matrix to be fed to the network
    if joints_to_consider==22:
        edgelist = [
                    (0, 1), (0, 2), #(0, 3),
                    (1, 4), (5, 2), #(3, 6),
                    (7, 4), (8, 5), #(6, 9),
                    (7, 10), (8, 11), #(9, 12),
                    #(12, 13), (12, 14),
                    (12, 15),
                    #(13, 16), (12, 16), (14, 17), (12, 17),
                    (12, 16), (12, 17),
                    (16, 18), (19, 17), (20, 18), (21, 19),
                    #(22, 20), #(23, 21),#wrists
                    (1, 16), (2, 17)]

    # create a graph
    G=nx.Graph()
    G.add_edges_from(edgelist)
    # create adjacency matrix
    A = nx.adjacency_matrix(G,nodelist=list(range(0,joints_to_consider))).todense()
    #normalize adjacency matrix
    A=normalize_A(A)
    return torch.Tensor(spatio_temporal_graph(joints_to_consider,temporal_kernel_size,A))


# In[23]:


def mpjpe_error(batch_pred,batch_gt):
    #assert batch_pred.requires_grad==True
    #assert batch_gt.requires_grad==False

    
    batch_pred=batch_pred.contiguous().view(-1,3)
    batch_gt=batch_gt.contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))