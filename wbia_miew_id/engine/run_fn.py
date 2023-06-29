import torch
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

from .train_fn import train_fn
from .eval_fn import eval_fn


def run_fn(config, model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=True):

    best_score = 0
    for epoch in range(config.engine.epochs):

        train_loss = train_fn(train_loader, model,criterion, optimizer, device,scheduler=scheduler,epoch=epoch, use_wandb=use_wandb)

        torch.save(model.state_dict(), f'{checkpoint_dir}/model_{epoch}.bin')
        
        valid_score = eval_fn(valid_loader, model, device, use_wandb=use_wandb)
        
        if valid_score > best_score:
            best_score = valid_score
            torch.save(model.state_dict(), f'{checkpoint_dir}/model_best.bin')
            print('best model found for epoch {}'.format(epoch))

    return best_score