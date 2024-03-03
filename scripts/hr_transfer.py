# python finetune_hr.py  --log-dir=./logs_ft_intent_hr_align --align_rep -c

import torch
import torch.optim as optim
import numpy as np
import torch_dct as dct #https://github.com/zh217/torch-dct
import time
import hydra
from omegaconf import DictConfig

from interact.model.Models import IntentInformedForecaster
from interact.model.Models import IntentInformedHRForecaster
from interact.utils.loss_funcs import mpjpe_loss
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import pathlib
from interact.utils.arg_parser import get_parser
from torch.utils.tensorboard import SummaryWriter

from interact.utils.cmu_mocap import CMU_Mocap
from interact.utils.synthetic_amass import Synthetic_AMASS
from interact.utils.comad import CoMaD
from interact.utils.comad_hr import CoMaD_HR
from torch.utils.data import ConcatDataset, DataLoader

dataset_map = {
    'CMU-Mocap' : {},
    'AMASS' : {},
    'COMAD' : {}
}

def get_dataloader(split='train', batch_size=256, include_amass=True, include_CMU_mocap=True, include_COMAD=False):
    # datalst = []
    # if include_CMU_mocap:
    #     if split not in dataset_map['CMU-Mocap']:
    #         dataset_map['CMU-Mocap'][split] = CMU_Mocap(split=split)
    #     datalst.append(dataset_map['CMU-Mocap'][split])
    # if include_amass:
    #     if split not in dataset_map['AMASS']:
    #         dataset_map['AMASS'][split] = Synthetic_AMASS(split=split)
    #     datalst.append(dataset_map['AMASS'][split])
    # if include_COMAD:
    #     if split not in dataset_map['COMAD']:
    #         dataset_map['COMAD'][split] = CoMaD(split=split)
    #     datalst.append(dataset_map['COMAD'][split])
    dataset = CoMaD_HR(split=split)
    dataloader = DataLoader(dataset, 
                batch_size=batch_size, 
                shuffle=True if split == 'train' else False)
    return dataloader

def log_metrics(dataloader, split, writer, epoch):
    total_loss, total_mpjpe, total_alignment_loss, n=0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            offset = batch[0].reshape(batch[0].shape[0], 
                                        batch[0].shape[1], -1)[:, -1].unsqueeze(1)
            alice_hist, alice_fut, bob_hist, bob_fut = [(batch[i].reshape(batch[i].shape[0], 
                                        batch[i].shape[1], -1) - offset).to(device) for i in range(4)]
            robot_hist, robot_fut = [(batch[i].reshape(batch[i].shape[0], batch[i].shape[1], -1) - offset[:, :, -6:]).to(device) for i in range(4,6)]
            alice_forecasts, alignment_loss = model(alice_hist, bob_hist, bob_fut, robot_hist, robot_fut)
            loss = mpjpe_loss(alice_forecasts, alice_fut) 

            batch_dim = alice_hist.shape[0]
            total_mpjpe += loss.item()*batch_dim

            if cfg.align_rep:
                total_alignment_loss += alignment_loss.item()*batch_dim
                loss += cfg.align_weight*alignment_loss

            total_loss+=loss*batch_dim
            n+=batch_dim
    
    print(f"{split} mpjpe after epoch {epoch+1} = ", total_mpjpe/n)
    writer.add_scalar(f'{split}/loss', total_loss.item()/n, epoch+1)
    writer.add_scalar(f'{split}/mpjpe', total_mpjpe/n, epoch+1)
    writer.add_scalar(f'{split}/align_loss', alignment_loss/n, epoch+1)

@hydra.main(config_path="../config", config_name="training")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    selected_model = cfg.selected_hr_model
    model_config = cfg.hr_models[selected_model] 

    model_id = f'{"1hist" if model_config.one_hist else "2hist"}_{"marginal" if not model_config.conditional_forecaster else "conditional"}'
    model_id += '_ft'
    load_model_id = model_id[:]
    model_id += '_hr'
    model_id += '_noalign' if not model_config.align_rep else ''
    writer = SummaryWriter(log_dir=f"{cfg.Training.log_dir}/{model_id}")
    
    train_dataloader = get_dataloader(split='train', batch_size=cfg.Training.batch_size, include_amass=False, include_CMU_mocap=False, include_COMAD=True)
    # val_dataloader = get_dataloader(split='val', batch_size=cfg.batch_size, include_amass=False, include_CMU_mocap=False, include_COMAD=True)
    # test_dataloader = get_dataloader(split='test', batch_size=cfg.batch_size, include_amass=False, include_CMU_mocap=False, include_COMAD=True)
    # amass_dataloader = get_dataloader(split='test', batch_size=cfg.batch_size, include_amass=True, include_CMU_mocap=False)
    # cmu_mocap_dataloader = get_dataloader(split='test', batch_size=cfg.batch_size, include_amass=False, include_CMU_mocap=True)
    # comad_val_dataloader = get_dataloader(split='val', batch_size=cfg.batch_size, include_amass=False, include_CMU_mocap=False, include_COMAD=True)
    # comad_test_dataloader = get_dataloader(split='test', batch_size=cfg.batch_size, include_amass=False, include_CMU_mocap=False, include_COMAD=True)

    model = hydra.utils.instantiate(
        model_config,
        device=device
    ).to(device)

    model.hh.load_state_dict(torch.load(f'{cfg.Training.finetuning.output_dir}/{load_model_id}/{30}.model'))

    params = [
        {"params": model.parameters(), "lr": cfg.Training.lr_ft}
    ]

    optimizer = optim.Adam(params,cfg.Training.weight_decay)

    directory = f'{cfg.Training.hr_training.output_dir}/{model_id}'
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.Training.epochs):
        total_loss, total_mpjpe, total_alignment_loss, n = 0, 0, 0, 0
        model.train()
        for j, batch in enumerate(train_dataloader):
            offset = batch[0].reshape(batch[0].shape[0], 
                                        batch[0].shape[1], -1)[:, -1].unsqueeze(1)
            alice_hist, alice_fut, bob_hist, bob_fut = [(batch[i].reshape(batch[i].shape[0], 
                                        batch[i].shape[1], -1) - offset).to(device) for i in range(4)]
            robot_hist, robot_fut = [(batch[i].reshape(batch[i].shape[0], batch[i].shape[1], -1) - offset[:, :, -6:]).to(device) for i in range(4,6)]
            alice_forecasts, alignment_loss = model(alice_hist, bob_hist, bob_fut, robot_hist, robot_fut)
            loss = mpjpe_loss(alice_forecasts, alice_fut) 

            batch_dim = alice_hist.shape[0]
            total_mpjpe += loss.item()*batch_dim
            
            if model_config.align_rep:
                total_alignment_loss += alignment_loss.item()*batch_dim
                loss += cfg.Training.align_weight*alignment_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss+=loss*batch_dim
            n+=batch_dim

        print(f"train mpjpe after epoch {epoch+1} = ", total_mpjpe/n)
        writer.add_scalar('train/loss', total_loss.item()/n, epoch+1)
        writer.add_scalar('train/mpjpe', total_mpjpe/n, epoch+1)
        writer.add_scalar('train/align_loss', alignment_loss/n, epoch+1)
        # if (epoch+1)%3 == 0:
        #     log_metrics(val_dataloader, 'comad_hr_val', writer, epoch)
        #     log_metrics(test_dataloader, 'comad_hr_test', writer, epoch)

        
        save_path=f'{directory}/{epoch+1}.model'
        torch.save(model.state_dict(),save_path)

if __name__ == "__main__":
    main()