import torch

def calc_reactive_stirring_task_cost(forecast_human_future, other_human_future, is_reaching, threshold = 0.4, 
                                     resting_pos = torch.Tensor([0.96849823, 1.35112756, 0.82395892]).to('cuda')):
    cost = torch.norm(other_human_future[:, :, 6]-resting_pos[None, None, :], dim=-1)*(forecast_human_future[:, :, :, 0].max(dim=2)[0]>threshold)*is_reaching[:, None]
    # return torch.sum(cost, dim=1).mean()
    return cost