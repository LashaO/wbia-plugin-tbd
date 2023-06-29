import torch
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import wandb

from metrics import AverageMeter, compute_distance_matrix, eval_onevsall


def eval_fn(data_loader,model,device, use_wandb=True):

    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in tk0:
            batch_embeddings = model.extract_feat(batch["image"].to(device))
            
            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            
            image_idx = batch["image_idx"].tolist()
            batch_embeddings_df = pd.DataFrame(batch_embeddings, index=image_idx)
            embeddings.append(batch_embeddings_df)

            batch_labels = batch['label'].tolist()
            labels.extend(batch_labels)
            
    embeddings = pd.concat(embeddings)

    dist_metric = 'cosine'
    qf = torch.Tensor(embeddings.values)
    q_pids = np.array(labels)

    distmat = compute_distance_matrix(qf, qf, dist_metric)
    distmat = distmat.numpy()

    print("Computing CMC and mAP ...")
    cmc, mAP = eval_onevsall(distmat, q_pids)

    ranks=[1, 5, 10, 20]
    print("** Results **")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        if use_wandb: wandb.log({"Rank-{:<3}".format(r): cmc[r - 1]})
        
    if use_wandb: wandb.log({"mAP": mAP})

    return mAP



