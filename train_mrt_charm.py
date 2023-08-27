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

from data import DATA, TESTDATA
dataset = DATA()
test_dataset = TESTDATA()

from discriminator_data import D_DATA
real_=D_DATA()

def train_model(args):
    ONE_HIST = args.one_hist
    CONDITIONAL = args.conditional
    
    batch_size=64
    torch.autograd.set_detect_anomaly(True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    real_motion_dataloader = torch.utils.data.DataLoader(real_, batch_size=batch_size, shuffle=True)
    real_motion_all=list(enumerate(real_motion_dataloader))

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)

    device='cuda'


    model = Transformer(d_word_vec=128, d_model=128, d_inner=1024,
                n_layers=3, n_head=8, d_k=64, d_v=64,device=device,conditional_forecaster=CONDITIONAL).to(device)
    discriminator = Discriminator(d_word_vec=45, d_model=45, d_inner=256,
                n_layers=3, n_head=8, d_k=32, d_v=32,device=device).to(device)

    lrate=0.0003
    lrate2=0.0005

    params = [
        {"params": model.parameters(), "lr": lrate}
    ]
    optimizer = optim.Adam(params)
    params_d = [
        {"params": discriminator.parameters(), "lr": lrate}
    ]
    optimizer_d = optim.Adam(params_d)

    for epoch in range(100):
        total_loss=0
        
        for j,data in enumerate(dataloader,0):
                    
            use=None
            input_seq,output_seq=data
            input_seq=torch.tensor(input_seq,dtype=torch.float32).to(device) # batch, N_person, 15 (15 fps 1 second), 45 (15joints xyz) 
            output_seq=torch.tensor(output_seq,dtype=torch.float32).to(device) # batch, N_persons, 46 (last frame of input + future 3 seconds), 45 (15joints xyz)
            alice_idx = 0 # human at 0th index is who we want the forecast for
            bob_idx = 1 # human at 1st index is whose future we are conditioning on

            if ONE_HIST:
                input_seq = input_seq[:,:1] 
                output_seq = output_seq[:,:1]

            # first 1 second predict future 1 second
            # input_=input_seq.view(-1,15,input_seq.shape[-1]) # batch x n_person ,15: 15 fps, 1 second, 45: 15joints x 3
            
            # output_=output_seq.view(output_seq.shape[0]*output_seq.shape[1],-1,input_seq.shape[-1])

            input_ = dct.dct(input_seq)

            rec_=model.forward(input_[:,alice_idx,1:15,:]-input_[:,alice_idx,:14,:],dct.idct(input_[:,alice_idx,-1:,:]),input_seq if not ONE_HIST else input_seq[:, alice_idx].unsqueeze(1),use,
                cond_future=(output_seq[:,bob_idx,:15,:]-input_[:,alice_idx,-1:,:1]) if CONDITIONAL else None)

            rec=dct.idct(rec_)
            
            results = output_seq[:,alice_idx,:1,:] # initial position
            for i in range(1,16):
                # iteratively concatenate more output prediction frames
                results = torch.cat([results,output_seq[:,alice_idx,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1) # adding up displacements to get true positions
            results =results[:,1:,:]
            # cumulative_sum = torch.cumsum(rec, dim=1)
            # results = output_[:, :1, :] + cumulative_sum[:, :45, :]
            loss=torch.mean(1000*(rec[:,:,:]-(output_seq[:,alice_idx,1:16,:]-output_seq[:,alice_idx,:15,:]))**2)
            
            
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
                d_motion_disc_loss *= 1000 #convert to mm
                
            # TODO: Add tensorboard logging for train and validation losses every 5 epochs
            # if (epoch%5) == 0:
            #     with torch.no_grad():
            #     model.eval()
            #     for jjj,data in enumerate(test_dataloader,0):
                
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (j + 1) %2 == 0:
                optimizer_d.zero_grad()
                d_motion_disc_loss.backward()
                optimizer_d.step()
    
            total_loss=total_loss+loss

        print('epoch:',epoch,'loss:',total_loss/(j+1))
        if (epoch+1)%5==0:
            directory = f'./saved_model_{"1hist" if ONE_HIST else "2hist"}_{"marginal" if not CONDITIONAL else "conditional"}'
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            save_path=f'{directory}/{epoch}.model'
            torch.save(model.state_dict(),save_path)


if __name__ == "__main__":
    train_model(get_parser().parse_args())


        
        
