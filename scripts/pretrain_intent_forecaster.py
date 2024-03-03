# python train_intent_forecaster.py --log-dir=./logs_intent_pretrain -c

import torch
import torch.optim as optim
import numpy as np
import torch_dct as dct #https://github.com/zh217/torch-dct
import time
import hydra
from omegaconf import DictConfig

from interact.model.Models import IntentInformedForecaster
from interact.utils.loss_funcs import mpjpe_loss
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import pathlib
from interact.utils.arg_parser import get_parser
from torch.utils.tensorboard import SummaryWriter

from interact.utils.cmu_mocap import CMU_Mocap
from interact.utils.synthetic_amass import Synthetic_AMASS
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

@hydra.main(config_path="../config", config_name="training")
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    selected_model = cfg.selected_model
    model_config = cfg.models[selected_model]  

    ONE_HIST = model_config.one_hist
    CONDITIONAL = model_config.conditional_forecaster

    model_id = f'{"1hist" if ONE_HIST else "2hist"}_{"marginal" if not CONDITIONAL else "conditional"}'
    # model_id += f'_{"handwrist" if bob_hand else "alljoints"}'

    writer = SummaryWriter(log_dir=f"{cfg.Training.log_dir}/{model_id}")

    train_dataloader = get_dataloader(split='train', batch_size=cfg.Training.batch_size, include_amass=True, include_CMU_mocap=False)
    # val_dataloader = get_dataloader(split='val', batch_size=args.batch_size, include_amass=True, include_CMU_mocap=True)
    # test_dataloader = get_dataloader(split='test', batch_size=args.batch_size, include_amass=True, include_CMU_mocap=True)
    # amass_dataloader = get_dataloader(split='test', batch_size=args.batch_size, include_amass=True, include_CMU_mocap=False)
    # cmu_mocap_dataloader = get_dataloader(split='test', batch_size=args.batch_size, include_amass=False, include_CMU_mocap=True)

    model = hydra.utils.instantiate(
        model_config,
        device=device
    ).to(device)

    params = [
        {"params": model.parameters(), "lr": cfg.Training.lr_pred}
    ]

    optimizer = optim.Adam(params, weight_decay=cfg.Training.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.Training.scheduler.milestones, gamma=cfg.Training.scheduler.gamma)

    directory = f"{cfg.Training.pretraining.output_dir}/{model_id}"
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.Training.epochs):
        total_loss, n = 0, 0
        model.train()
        for j, batch in enumerate(train_dataloader):
            offset = batch[0].reshape(batch[0].shape[0], batch[0].shape[1], -1)[:, -1].unsqueeze(1)
            alice_hist, alice_fut, bob_hist, bob_fut = [(b.reshape(b.shape[0], b.shape[1], -1) - offset).to(device) for b in batch]
            alice_forecasts = model(alice_hist, bob_hist, bob_fut)
            loss = mpjpe_loss(alice_forecasts, alice_fut)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_dim = alice_hist.shape[0]
            total_loss += loss * batch_dim
            n += batch_dim

        print(f"train loss after epoch {epoch + 1} = ", total_loss.item() / n)
        writer.add_scalar('train/mpjpe', total_loss.item() / n, epoch + 1)
        # if (epoch+1)%5 == 0:
        #     log_metrics(val_dataloader, 'val', writer, epoch)
            # log_metrics(test_dataloader, 'test', writer, epoch)
            # log_metrics(amass_dataloader, 'amass_test', writer, epoch)
            # log_metrics(cmu_mocap_dataloader, 'cmu_mocap_test', writer, epoch)

        save_path = f'{directory}/{epoch + 1}.model'
        torch.save(model.state_dict(), save_path)
        scheduler.step()

if __name__ == "__main__":
    main()