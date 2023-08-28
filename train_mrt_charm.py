import torch
import torch.optim as optim
import numpy as np
import torch_dct as dct #https://github.com/zh217/torch-dct
import time

from MRT.Models import Transformer,Discriminator
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


def train_model(args):
    ONE_HIST = args.one_hist
    CONDITIONAL = args.conditional
    
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
    discriminator = Discriminator(d_word_vec=45, d_model=45, d_inner=256,
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
                    
            use=None
            input_seq,output_seq=data
            input_seq=torch.tensor(input_seq,dtype=torch.float32).to(device) # batch, N_person, 15 (15 fps 1 second), 45 (15joints xyz) 
            output_seq=torch.tensor(output_seq,dtype=torch.float32).to(device) # batch, N_persons, 46 (last frame of input + future 3 seconds), 45 (15joints xyz)
            alice_idx = 0 # human at 0th index is who we want the forecast for
            bob_idx = 1 # human at 1st index is whose future we are conditioning on

            input_ = dct.dct(input_seq)

            source_seq = input_[:,alice_idx,1:15,:]-input_[:,alice_idx,:14,:]
            target_seq = dct.idct(input_[:,alice_idx,-1:,:])
            relevant_human_idxs = [alice_idx]
            if not ONE_HIST:
                relevant_human_idxs.append(bob_idx)
            histories = extract_histories(input_seq, relevant_human_idxs)
            cond_future = (output_seq[:,bob_idx,:15,:]-input_[:,alice_idx,-1:,:1]) if CONDITIONAL else None

            rec_=model.forward(source_seq,target_seq,histories,use,cond_future=cond_future)

            rec=dct.idct(rec_)
            
            results = output_seq[:,alice_idx,:1,:] # initial position
            for i in range(1,16):
                # iteratively concatenate more output prediction frames
                results = torch.cat([results,output_seq[:,alice_idx,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1) # adding up displacements to get true positions
            results =results[:,1:,:]
            gt_vel = output_seq[:,alice_idx,1:16,:]-output_seq[:,alice_idx,:15,:]
            loss=torch.mean((rec-gt_vel)**2)
            
            
            if (j+1)%2==0:
                
                fake_motion=results

                disc_loss=disc_l2_loss(discriminator(fake_motion))
                loss=loss+0.0005*disc_loss
                
                fake_motion=fake_motion.detach()

                real_motion=real_motion_all[int(j/2)][1][1][:,alice_idx]
                real_motion=real_motion.view(-1,46,45)[:,1:16,:].float().to(device)

                fake_disc_value = discriminator(fake_motion)
                real_disc_value = discriminator(real_motion)

                d_motion_disc_real, d_motion_disc_fake, d_motion_disc_loss = adv_disc_l2_loss(real_disc_value, fake_disc_value)                
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (j + 1) %2 == 0:
                optimizer_d.zero_grad()
                d_motion_disc_loss.backward()
                optimizer_d.step()
    
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
                    val_input_seq, val_output_seq = val_data
                    val_input_seq = torch.tensor(val_input_seq, dtype=torch.float32).to(device)
                    val_output_seq = torch.tensor(val_output_seq, dtype=torch.float32).to(device)

                    val_input_ = dct.dct(val_input_seq)

                    val_source_seq = val_input_[:,alice_idx,1:15,:]-val_input_[:,alice_idx,:14,:]
                    val_target_seq = dct.idct(val_input_[:,alice_idx,-1:,:])
                    val_relevant_human_idxs = [alice_idx]
                    if not ONE_HIST:
                        val_relevant_human_idxs.append(bob_idx)
                    val_histories = extract_histories(val_input_seq, val_relevant_human_idxs)
                    val_cond_future = (val_output_seq[:,bob_idx,:15,:]-val_input_[:,alice_idx,-1:,:1]) if CONDITIONAL else None

                    val_rec_ = model.forward(val_source_seq,val_target_seq,
                        val_histories,use,
                        cond_future=val_cond_future)
                    val_rec = dct.idct(val_rec_)
                    val_gt_vel = val_output_seq[:,alice_idx,1:16,:]-val_output_seq[:,alice_idx,:15,:]
                    val_loss += torch.mean((val_rec-val_gt_vel)**2)
                    
                    num_val_batches += 1
            
            val_loss /= num_val_batches

            writer.add_scalar('loss/train', loss, epoch + 1)
            writer.add_scalar('loss/val', val_loss, epoch + 1)

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


        
        
