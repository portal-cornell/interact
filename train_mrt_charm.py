import torch
import torch.optim as optim
import numpy as np
import torch_dct as dct #https://github.com/zh217/torch-dct
import time

from MRT.Models import Transformer,Discriminator,JOINT_AMT
from utils.loss_funcs import disc_l2_loss,adv_disc_l2_loss
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import pathlib
from arg_parser import get_parser
from torch.utils.tensorboard import SummaryWriter

from data import DATA, TESTDATA
dataset = DATA()
test_dataset = TESTDATA()

from discriminator_data import D_DATA
real_=D_DATA()


def extract_histories(input_seq, human_idxs):
    return input_seq[:, human_idxs]

def forward_pass(model,alice_hist,bob_hist,bob_future,alice_idx,bob_idx,one_hist,conditional,bob_is_robot=False):
    use=None
    ONE_HIST = one_hist
    CONDITIONAL = conditional
    # TODO: Change the common joints depending on if bob is not a robot and if bob is a robot
    common_joints = [i for i in range(JOINT_AMT*3)] if not bob_is_robot else [i for i in range(JOINT_AMT*3)]
    
    input_ = dct.dct(alice_hist) # batch, T_in (15), 45 (15joints * 3xyz)

    source_seq = input_[:,1:15,:]-input_[:,:14,:] # 14 displacements, as done in original code
    target_seq = dct.idct(input_[:,-1:,:]) # Alice's current position
    # relevant_human_idxs = [alice_idx]
    # if not ONE_HIST:
    #     relevant_human_idxs.append(bob_idx)
    # # TODO: Add MLP to project bob's joints into same embedding
    # histories = extract_histories(input_seq, relevant_human_idxs) # batch, N_person, T_in (15), 45

    # CONDITIONAL: Relative positions of Bob compared to a joint position at Alice's current timestep
    # TODO: Use relevant joints for bob (that will be wrist and hand)
    cond_future = (bob_hist[:,:15,:]-target_seq[:,-1:,:])
    if not CONDITIONAL:
        cond_future.fill_(0)

    rec_=model.forward(source_seq,target_seq,alice_hist,bob_hist,common_joints,use,cond_future=cond_future,bob_is_robot=bob_is_robot,one_hist=ONE_HIST)

    rec=dct.idct(rec_) # predicts displacements for each of Alice's joints - batch, T_out (15), 45
    
    results = alice_hist[:,-1:,:] # initial position
    for i in range(1,16):
        # iteratively concatenate more output prediction frames to get actual positions from the displacements
        # TODO: change to cumsum
        results = torch.cat([results,alice_hist[:,-1:,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1) # adding up displacements to get true positions
    # ignore first timestep of this since it overlaps with last timestep of input 
    results =results[:,1:,:] # batch, T_out (15), 45

    return rec, results

def compute_loss(model, input_seq_tmp, output_seq, alice_idx, bob_idx, ONE_HIST, CONDITIONAL, device):
    alice_hist = input_seq_tmp[:, alice_idx]
    bob_hist = input_seq_tmp[:, bob_idx]

    rec, results = forward_pass(model, alice_hist, bob_hist, output_seq[:, bob_idx],
                                alice_idx, bob_idx, ONE_HIST, CONDITIONAL, bob_is_robot=True)

    gt_disp = output_seq[:, alice_idx, 1:16, :] - output_seq[:, alice_idx, :15, :]

    loss = torch.mean((rec - gt_disp) ** 2)
    return loss, results


def train_model(args):
    JOINT_AMT = 9
    ONE_HIST = args.one_hist
    CONDITIONAL = args.conditional
    use=None
    
    batch_size=args.batch_size
    torch.autograd.set_detect_anomaly(True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    real_motion_dataloader = torch.utils.data.DataLoader(real_, batch_size=batch_size, shuffle=True)
    real_motion_all=list(enumerate(real_motion_dataloader))

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)

    device='cuda'
    model_id = f'{"1hist" if ONE_HIST else "2hist"}_{"marginal" if not CONDITIONAL else "conditional"}'
    writer = SummaryWriter(log_dir=args.log_dir+'/'+model_id)

    model = Transformer(d_word_vec=128, d_model=128, d_inner=1024,
                n_layers=3, n_head=8, d_k=64, d_v=64,device=device,conditional_forecaster=CONDITIONAL).to(device)
    discriminator = Discriminator(d_word_vec=JOINT_AMT*3, d_model=JOINT_AMT*3, d_inner=256,
                n_layers=3, n_head=8, d_k=32, d_v=32,device=device).to(device)

    lrate=args.lr_pred
    lrate2=args.lr_disc

    params = [
        {"params": model.parameters(), "lr": lrate}
    ]
    optimizer = optim.Adam(params)
    params_d = [
        {"params": discriminator.parameters(), "lr": lrate2}
    ]
    optimizer_d = optim.Adam(params_d)

    for epoch in range(50):
        total_loss=0
        
        for j,data in enumerate(dataloader,0):
            input_seq_tmp,output_seq=data
            # TODO: Once we have the dataloaders ready with human vs. robot data, this will change since dims are different
            input_seq_tmp=torch.tensor(input_seq_tmp,dtype=torch.float32).to(device) # batch, N_person, 15 (15 fps 1 second), 45 (15joints xyz) 
            output_seq=torch.tensor(output_seq,dtype=torch.float32).to(device) # batch, N_persons, 46 (last frame of input + future 3 seconds), 45 (15joints xyz)
            alice_idx = 0 # human at 0th index is who we want the forecast for
            bob_idx = 1 # human at 1st index is whose future we are conditioning on

            loss, results=compute_loss(model, input_seq_tmp, output_seq, alice_idx, bob_idx,
                                        ONE_HIST, CONDITIONAL, device)
            
            
            
            # if (j+1)%2==0:
                
            #     fake_motion=results

            #     disc_loss=disc_l2_loss(discriminator(fake_motion))
            #     loss=loss+0.0005*disc_loss
                
            #     fake_motion=fake_motion.detach()

            #     real_motion=real_motion_all[int(j/2)][1][1][:,alice_idx]
            #     real_motion=real_motion.view(-1,46,JOINT_AMT*3)[:,1:16,:].float().to(device)

            #     fake_disc_value = discriminator(fake_motion)
            #     real_disc_value = discriminator(real_motion)

            #     d_motion_disc_real, d_motion_disc_fake, d_motion_disc_loss = adv_disc_l2_loss(real_disc_value, fake_disc_value)                
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (j + 1) %2 == 0:
            #     optimizer_d.zero_grad()
            #     d_motion_disc_loss.backward()
            #     optimizer_d.step()
    
            total_loss=total_loss+loss
        
        """
        Tensorboard logging (train/val losses)
        """
        if (epoch + 1) % 2 == 0: 
            model.eval()
            val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for val_data in test_dataloader:
                    val_input_seq_tmp, val_output_seq = val_data
                    val_input_seq_tmp = torch.tensor(val_input_seq_tmp, dtype=torch.float32).to(device)
                    val_output_seq = torch.tensor(val_output_seq, dtype=torch.float32).to(device)
                    
                    val_alice_idx = 0
                    val_bob_idx = 1

                    cur_val_loss, val_results = compute_loss(model, val_input_seq_tmp, val_output_seq, val_alice_idx, val_bob_idx,
                                                     ONE_HIST, CONDITIONAL, device)
                    val_loss += cur_val_loss
                    num_val_batches += 1
            
            val_loss /= num_val_batches

            def mpjpe_loss(pred, output, dataset_scaling=1.0):
                n_joints = int(pred.shape[-1]/3)
                prediction = pred.view(pred.shape[0],-1,n_joints,3)
                gt = output.view(pred.shape[0],-1,n_joints,3)
                return 1000*torch.sqrt(((prediction/dataset_scaling - gt/dataset_scaling) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().detach().numpy()

            # NOTE: the 1.8 scaling is specific to CMU Mocap dataset, for others default to 1
            # TODO: Create a map from dataset name to scaling factor and reference that
            train_mpjpe = mpjpe_loss(results, output_seq[:,alice_idx,1:16], dataset_scaling=1.8)
            val_mpjpe = mpjpe_loss(val_results, val_output_seq[:,alice_idx,1:16], dataset_scaling=1.8)
            writer.add_scalar('loss/train', loss, epoch + 1)
            writer.add_scalar('loss/val', val_loss, epoch + 1)
            writer.add_scalar('mpjpe/train', np.mean(train_mpjpe), epoch + 1)
            writer.add_scalar('mpjpe/val', np.mean(val_mpjpe), epoch + 1)

            model.train()

        print('epoch:',epoch,'loss:',total_loss/(j+1))
        if (epoch+1)%5==0:
            directory = f'./saved_model_{model_id}'
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            save_path=f'{directory}/{epoch}.model'
            torch.save(model.state_dict(),save_path)

    writer.close()


if __name__ == "__main__":
    train_model(get_parser().parse_args())


        
        
