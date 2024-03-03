import json
import numpy as np
from interact.model.Models import IntentInformedForecaster
import torch
import numpy as np
import torch
import torch_dct as dct #https://github.com/zh217/torch-dct
from torch.utils.data import DataLoader, ConcatDataset
from interact.utils.cmu_mocap import CMU_Mocap
from interact.utils.comad import CoMaD
from interact.utils.synthetic_amass import Synthetic_AMASS
from interact.utils.loss_funcs import mpjpe_loss, fde_error, perjoint_error, perjoint_fde
from interact.utils.arg_parser import get_parser
import hydra

@hydra.main(config_path="../config", config_name="eval")
def main(cfg):
    dataset_map = {
        'cmu': lambda: CMU_Mocap(split='test'),
        'handover': lambda: CoMaD(split='test',subtask='handover',transitions=True),
        'react_stir': lambda: CoMaD(split='test',subtask='react_stir',transitions=True),
        'table_set': lambda: CoMaD(split='test',subtask='table_set')
    }

    models = cfg.hh_eval.models
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_results_dict = {}

    Dataset = ConcatDataset([dataset_map[cfg.hh_eval.eval_data]()])
    loader_test = DataLoader(
        Dataset,
        batch_size = cfg.hh_eval.batch_size,
        shuffle = False,
        num_workers = 0)

    model_info_cfg = hydra.compose(config_name="training", overrides=[])
    
    for model_path in models:
        model_config = model_info_cfg.hh_models[model_path]
        model = hydra.utils.instantiate(
            model_config,
            device=device
        ).to(device)

        model.load_state_dict(torch.load(f'{cfg.hh_eval.checkpoint_dir}/{model_path}/{30}.model'))
        model.eval()

        running_loss=0
        running_per_joint_error=0
        running_fde=0
        running_per_joint_fde=0
        running_per_joint_errors = []
        running_per_joint_fdes = []
        n=0
        with torch.no_grad():
            for cnt,batch in enumerate(loader_test): 
                offset = batch[0].reshape(batch[0].shape[0], 
                                        batch[0].shape[1], -1)[:, -1].unsqueeze(1)
                alice_hist, alice_fut, bob_hist, bob_fut = [(b.reshape(b.shape[0], 
                                            b.shape[1], -1) - offset).to(device) for b in batch]
                batch_dim = alice_hist.shape[0]
                n += batch_dim
                alice_forecasts = model(alice_hist, bob_hist, bob_fut)
                loss = mpjpe_loss(alice_forecasts, alice_fut)
                per_joint_error, per_joint_error_list = perjoint_error(alice_forecasts, alice_fut)
                fde = fde_error(alice_forecasts, alice_fut)   
                per_joint_fde, per_joint_fde_list = perjoint_fde(alice_forecasts, alice_fut)

                running_per_joint_errors += list(per_joint_error_list.cpu().numpy())
                running_per_joint_fdes += list(per_joint_fde_list.cpu().numpy())
                running_per_joint_fde += per_joint_fde*batch_dim
                running_loss += loss*batch_dim
                running_per_joint_error += per_joint_error*batch_dim
                running_fde += fde*batch_dim
        
        all_joints_ade_mean = np.array(running_per_joint_errors).mean(axis=1).mean()*1000
        all_joints_ade_std = np.array(running_per_joint_errors).mean(axis=1).std()*1000/np.sqrt((n))

        all_joints_fde_mean = np.array(running_per_joint_fdes).mean(axis=1).mean()*1000
        all_joints_fde_std = np.array(running_per_joint_fdes).mean(axis=1).std()*1000/np.sqrt((n))

        wrist_ade_mean = np.array(running_per_joint_errors)[:, 5:9].mean(axis=1).mean()*1000
        wrist_ade_std = np.array(running_per_joint_errors)[:, 5:9].mean(axis=1).std()*1000/np.sqrt((n))

        wrist_fde_mean = np.array(running_per_joint_fdes)[:, 5:9].mean(axis=1).mean()*1000
        wrist_fde_std = np.array(running_per_joint_fdes)[:, 5:9].mean(axis=1).std()*1000/np.sqrt((n))

        model_results_dict[model_path] = {
            'all_joints_ade': [all_joints_ade_mean, all_joints_ade_std],
            'all_joints_fde': [all_joints_fde_mean, all_joints_fde_std],
            'wrist_ade': [wrist_ade_mean, wrist_ade_std],
            'wrist_fde': [wrist_fde_mean, wrist_fde_std],
        }

    print('DATASET: ' + cfg.hh_eval.eval_data)
    metrics = ['all_joints','wrist']
    for metric in metrics:
        for model_path in models:
            fde_metric = metric + '_fde'
            mean = round(model_results_dict[model_path][fde_metric][0], 1)
            std = round(model_results_dict[model_path][fde_metric][1], 1)
            print(f'{mean} (\pm {std})', end = ' &')
            if model_path == "cvm":
                print('&', end='')
        print()
        print('='*20)

if __name__ == "__main__":
    main()
