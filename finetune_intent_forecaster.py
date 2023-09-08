import torch
import torch.optim as optim
import numpy as np
import torch_dct as dct #https://github.com/zh217/torch-dct
import time

from MRT.Models import IntentInformedForecaster
from utils.loss_funcs import mpjpe_loss,disc_l2_loss,adv_disc_l2_loss
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import pathlib
from arg_parser import get_parser
from torch.utils.tensorboard import SummaryWriter

from utils.cmu_mocap import CMU_Mocap
from utils.synthetic_amass import Synthetic_AMASS
from utils.comad import CoMaD
from torch.utils.data import ConcatDataset, DataLoader

dataset_map = {
    'CMU-Mocap' : {},
    'AMASS' : {},
    'COMAD' : {}
}

def get_dataloader(split='train', batch_size=256, include_amass=True, include_CMU_mocap=True, include_COMAD=False):
    datalst = []
    if include_CMU_mocap:
        if split not in dataset_map['CMU-Mocap']:
            dataset_map['CMU-Mocap'][split] = CMU_Mocap(split=split)
        datalst.append(dataset_map['CMU-Mocap'][split])
    if include_amass:
        if split not in dataset_map['AMASS']:
            dataset_map['AMASS'][split] = Synthetic_AMASS(split=split)
        datalst.append(dataset_map['AMASS'][split])
    if include_COMAD:
        if split not in dataset_map['COMAD']:
            dataset_map['COMAD'][split] = CoMaD(split=split)
        datalst.append(dataset_map['COMAD'][split])
    dataset = ConcatDataset(datalst)
    dataloader = DataLoader(dataset, 
                batch_size=batch_size, 
                shuffle=True if split == 'train' else False)
    return dataloader

def log_metrics(dataloader, split, writer, epoch):
    total_loss, n=0, 0
    model.eval()
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            # [b.to(device) for b in batch]
            offset = batch[0].reshape(batch[0].shape[0], 
                                        batch[0].shape[1], -1)[:, -1].unsqueeze(1)
            alice_hist, alice_fut, bob_hist, bob_fut = [(b.reshape(b.shape[0], 
                                        b.shape[1], -1) - offset).to(device) for b in batch]
            alice_forecasts = model(alice_hist, bob_hist, bob_fut)

            loss = mpjpe_loss(alice_forecasts, alice_fut)
            batch_dim = alice_hist.shape[0]
            total_loss+=loss*batch_dim
            n+=batch_dim
    
    print(f"{split} loss after epoch {epoch+1} = ", total_loss.item()/n)
    writer.add_scalar(f'{split}/mpjpe', total_loss.item()/n, epoch+1)

if __name__ == "__main__":
    args = get_parser().parse_args()

    ONE_HIST = args.one_hist
    CONDITIONAL = args.conditional

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = f'{"1hist" if ONE_HIST else "2hist"}_{"marginal" if not CONDITIONAL else "conditional"}'
    model_id += f'_{"noAMASS" if args.no_amass else "withAMASS"}_{"handwrist" if args.bob_hand else "alljoints"}'
    load_model_id = model_id[:]
    model_id += '_ft'
    writer = SummaryWriter(log_dir=args.log_dir+'/'+model_id)
    
    train_dataloader = get_dataloader(split='train', batch_size=args.batch_size, include_amass=False, include_CMU_mocap=False, include_COMAD=True)
    # val_dataloader = get_dataloader(split='val', batch_size=args.batch_size, include_amass=(not args.no_amass), include_CMU_mocap=True)
    # test_dataloader = get_dataloader(split='test', batch_size=args.batch_size, include_amass=(not args.no_amass), include_CMU_mocap=True)
    # amass_dataloader = get_dataloader(split='test', batch_size=args.batch_size, include_amass=True, include_CMU_mocap=False)
    cmu_mocap_dataloader = get_dataloader(split='test', batch_size=args.batch_size, include_amass=False, include_CMU_mocap=True)
    comad_val_dataloader = get_dataloader(split='val', batch_size=args.batch_size, include_amass=False, include_CMU_mocap=False, include_COMAD=True)
    # comad_test_dataloader = get_dataloader(split='test', batch_size=args.batch_size, include_amass=False, include_CMU_mocap=False, include_COMAD=True)
    bob_joints_list = list(range(9)) if not args.bob_hand else list(range(5,9))

    model = IntentInformedForecaster(d_word_vec=128, d_model=128, d_inner=1024,
                n_layers=3, n_head=8, d_k=64, d_v=64,
                device=device,
                conditional_forecaster=CONDITIONAL,
                bob_joints_list=bob_joints_list,
                bob_joints_num=len(bob_joints_list),
                one_hist=ONE_HIST).to(device)

    model.load_state_dict(torch.load(f'./checkpoints_new_arch/saved_model_{load_model_id}/{50}.model'))

    params = [
        {"params": model.parameters(), "lr": args.lr_ft}
    ]

    optimizer = optim.Adam(params,weight_decay=1e-05)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                milestones=[15,25,35,40], 
                gamma=0.1)

    directory = f'./checkpoints_new_arch_finetuned/saved_model_{model_id}'
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        total_loss, total_mpjpe, n = 0, 0, 0
        model.train()
        for j, batch in enumerate(train_dataloader):
            offset = batch[0].reshape(batch[0].shape[0], 
                                        batch[0].shape[1], -1)[:, -1].unsqueeze(1)
            alice_hist, alice_fut, bob_hist, bob_fut = [(b.reshape(b.shape[0], 
                                        b.shape[1], -1) - offset).to(device) for b in batch]
            alice_forecasts = model(alice_hist, bob_hist, bob_fut)
            loss = mpjpe_loss(alice_forecasts, alice_fut)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_dim = alice_hist.shape[0]
            total_loss+=loss*batch_dim
            n+=batch_dim

        print(f"train loss after epoch {epoch+1} = ", total_loss.item()/n)
        writer.add_scalar('train/mpjpe', total_loss.item()/n, epoch+1)
        if (epoch+1)%3 == 0:
            log_metrics(comad_val_dataloader, 'comad_val', writer, epoch)
            # log_metrics(test_dataloader, 'test', writer, epoch)
            # log_metrics(amass_dataloader, 'amass_test', writer, epoch)
            log_metrics(cmu_mocap_dataloader, 'cmu_mocap_test', writer, epoch)
        
        save_path=f'{directory}/{epoch+1}.model'
        torch.save(model.state_dict(),save_path)
        scheduler.step()
    