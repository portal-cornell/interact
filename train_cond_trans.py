import torch
import torch.optim as optim
import numpy as np
import torch_dct as dct #https://github.com/zh217/torch-dct
import time

from MRT.Models import ConditionalForecaster
from utils.loss_funcs import mpjpe_loss,disc_l2_loss,adv_disc_l2_loss
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import pathlib
from arg_parser import get_parser
from torch.utils.tensorboard import SummaryWriter

from utils.cmu_mocap import CMU_Mocap
from utils.synthetic_amass import Synthetic_AMASS
from torch.utils.data import ConcatDataset, DataLoader

dataset_map = {
    'CMU-Mocap' : {},
    'AMASS' : {}
}

def get_dataloader(split='train', batch_size=256, include_amass=True, include_CMU_mocap=True):
    datalst = []
    if include_CMU_mocap:
        if split not in dataset_map['CMU-Mocap']:
            dataset_map['CMU-Mocap'][split] = CMU_Mocap(split=split)
        datalst.append(dataset_map['CMU-Mocap'][split])
    if include_amass:
        if split not in dataset_map['AMASS']:
            dataset_map['AMASS'][split] = Synthetic_AMASS(split=split)
        datalst.append(dataset_map['AMASS'][split])
    dataset = ConcatDataset(datalst)
    dataloader = DataLoader(dataset, 
                batch_size=batch_size, 
                shuffle=True if split == 'train' else False)
    return dataloader

def log_metrics(dataloader, split, writer, epoch):
    total_loss, total_mpjpe, n = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            offset = batch[0].reshape(batch[0].shape[0], 
                                        batch[0].shape[1], -1, 3)[:, -1].unsqueeze(1)
            alice_hist, alice_fut, bob_hist, bob_fut = [(b.reshape(b.shape[0], 
                                        b.shape[1], -1, 3)-offset).to(device) for b in batch]
            alice_hist, alice_fut, bob_hist, bob_fut = add_translation_joint(alice_hist, 
                                alice_fut, bob_hist, bob_fut)

            alice_forecasts = model(alice_hist, bob_hist, bob_fut)

            loss = mpjpe_loss(alice_forecasts, alice_fut)
            mpjpe = get_real_mpjpe_loss(alice_forecasts, alice_fut)

            batch_dim = alice_hist.shape[0]
            total_loss+=loss*batch_dim
            total_mpjpe+=mpjpe*batch_dim
            n+=batch_dim
    
    print(f"{split} loss after epoch {epoch+1} = ", total_loss.item()/n)
    print(f"{split} mpjpe after epoch {epoch+1} = ", total_mpjpe.item()/n)

    writer.add_scalar(f'{split}/loss', total_loss.item()/n, epoch+1)
    writer.add_scalar(f'{split}/mpjpe', total_mpjpe.item()/n, epoch+1)

def get_translation_joint(joint_data):
    translation_joint = joint_data.mean(dim=2).unsqueeze(2)
    joint_data = torch.cat([translation_joint, joint_data-translation_joint], dim=2)
    return joint_data.reshape(joint_data.shape[0], joint_data.shape[1], -1)

def add_translation_joint(alice_hist, alice_fut, bob_hist, bob_fut):
    alice_hist = get_translation_joint(alice_hist)
    alice_fut = get_translation_joint(alice_fut)
    bob_hist = get_translation_joint(bob_hist)
    bob_fut = get_translation_joint(bob_fut)
    return alice_hist, alice_fut, bob_hist, bob_fut

def remove_translation_joint(joint_data):
    joint_data = joint_data.reshape(joint_data.shape[0], joint_data.shape[1], -1, 3)
    joint_data = joint_data[:, :, 1:]+joint_data[:, :, :1]
    return joint_data.reshape(joint_data.shape[0], joint_data.shape[1], -1)

def get_real_mpjpe_loss(forecast, future):
    forecast = remove_translation_joint(forecast)
    future = remove_translation_joint(future)
    return mpjpe_loss(forecast, future)

if __name__ == "__main__":
    args = get_parser().parse_args()

    ONE_HIST = args.one_hist
    CONDITIONAL = args.conditional

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = f'{"1hist" if ONE_HIST else "2hist"}_{"marginal" if not CONDITIONAL else "conditional"}'
    model_id += f'_{"noAMASS" if args.no_amass else "withAMASS"}_{"handwrist" if args.bob_hand else "alljoints"}'
    model_id += '_trans'
    if args.condition_last:
        model_id += f'_condlast'
    writer = SummaryWriter(log_dir=args.log_dir+'/'+model_id)
    
    train_dataloader = get_dataloader(split='train', batch_size=args.batch_size, include_amass=(not args.no_amass))
    val_dataloader = get_dataloader(split='val', batch_size=args.batch_size, include_amass=(not args.no_amass), include_CMU_mocap=True)
    test_dataloader = get_dataloader(split='test', batch_size=args.batch_size, include_amass=(not args.no_amass), include_CMU_mocap=True)
    amass_dataloader = get_dataloader(split='test', batch_size=args.batch_size, include_amass=True, include_CMU_mocap=False)
    cmu_mocap_dataloader = get_dataloader(split='test', batch_size=args.batch_size, include_amass=False, include_CMU_mocap=True)
    bob_joints_list = list(range(10)) if not args.bob_hand else [0,6,7,8,9]

    model = ConditionalForecaster(d_word_vec=128, d_model=128, d_inner=1024,
                n_layers=3, n_head=8, d_k=64, d_v=64,
                device=device,
                conditional_forecaster=CONDITIONAL,
                bob_joints_list=bob_joints_list,
                alice_joints_num=10,
                bob_joints_num=len(bob_joints_list),
                one_hist=ONE_HIST).to(device)

    params = [
        {"params": model.parameters(), "lr": args.lr_pred}
    ]

    optimizer = optim.Adam(params,weight_decay=1e-05)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                milestones=[15,25,35,40], 
                gamma=0.1)

    directory = f'./checkpoints_trans/saved_model_{model_id}'
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        total_loss, total_mpjpe, n = 0, 0, 0
        model.train()
        for j, batch in enumerate(train_dataloader):
            offset = batch[0].reshape(batch[0].shape[0], 
                                        batch[0].shape[1], -1, 3)[:, -1].unsqueeze(1)
            alice_hist, alice_fut, bob_hist, bob_fut = [(b.reshape(b.shape[0], 
                                        b.shape[1], -1, 3)-offset).to(device) for b in batch]
            alice_hist, alice_fut, bob_hist, bob_fut = add_translation_joint(alice_hist, 
                                alice_fut, bob_hist, bob_fut)
            if args.condition_last:
                bob_fut_zeros = torch.zeros_like(bob_fut)
                bob_fut[:,:-1] = bob_fut_zeros[:,:-1]
            alice_forecasts = model(alice_hist, bob_hist, bob_fut)

            loss = mpjpe_loss(alice_forecasts, alice_fut)
            mpjpe = get_real_mpjpe_loss(alice_forecasts, alice_fut)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_dim = alice_hist.shape[0]
            total_loss+=loss*batch_dim
            total_mpjpe+=mpjpe*batch_dim
            n+=batch_dim

        print(f"train loss after epoch {epoch+1} = ", total_loss.item()/n)
        print(f"train mpjpe after epoch {epoch+1} = ", total_mpjpe.item()/n)
        writer.add_scalar('train/loss', total_loss.item()/n, epoch+1)
        writer.add_scalar('train/mpjpe', total_mpjpe.item()/n, epoch+1)

        log_metrics(val_dataloader, 'val', writer, epoch)
        log_metrics(test_dataloader, 'test', writer, epoch)
        log_metrics(amass_dataloader, 'amass_test', writer, epoch)
        log_metrics(cmu_mocap_dataloader, 'cmu_mocap_test', writer, epoch)
        
        save_path=f'{directory}/{epoch+1}.model'
        torch.save(model.state_dict(),save_path)
        scheduler.step()
    