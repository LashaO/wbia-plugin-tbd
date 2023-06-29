import torch
import sys
# sys.path.append('..'); from wbia_pie_v2.models import MiewIdNet
# from datasets import MiewIdDataset
from .model import MiewIdNet

def get_model(cfg, checkpoint_path=None, use_gpu=True):

    model = MiewIdNet(**dict(cfg.model_params))


    if use_gpu:
        device = torch.device("cuda")
        model.to(device)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print('loaded checkpoint from', checkpoint_path)

    return model