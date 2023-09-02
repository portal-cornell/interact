import torch
import torch.optim as optim
import numpy as np
import torch_dct as dct #https://github.com/zh217/torch-dct
import time

from MRT.Models import Transformer
from utils.loss_funcs import disc_l2_loss,adv_disc_l2_loss
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import pathlib
from arg_parser import get_parser
from torch.utils.tensorboard import SummaryWriter

from utils.cmu_mocap import CMU_Mocap
from utils.synthetic_amass import Synthetic_AMASS
from torch.utils.data import ConcatDataset, DataLoader

if __name__ == "__main__":
    args = get_parser().parse_args()

    ONE_HIST = args.one_hist
    CONDITIONAL = args.conditional

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model_id = f'{"1hist" if ONE_HIST else "2hist"}_{"marginal" if not CONDITIONAL else "conditional"}'
    writer = SummaryWriter(log_dir=args.log_dir+'/'+model_id)

    cmu_mocap_train = CMU_Mocap(split='train')
    synthetic_amass_train = Synthetic_AMASS(split='train')
    train_dataset = ConcatDataset([cmu_mocap_train, 
                    synthetic_amass_train])
    train_dataloader = DataLoader(train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True)

    model = Transformer(d_word_vec=128, d_model=128, d_inner=1024,
                n_layers=3, n_head=8, d_k=64, d_v=64,
                device=device,
                conditional_forecaster=CONDITIONAL).to(device)

    params = [
        {"params": model.parameters(), "lr": args.lr_pred}
    ]
    optimizer = optim.Adam(params)
    