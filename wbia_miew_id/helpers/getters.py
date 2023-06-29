# import torch
# from models import MiewIdNet
# from datasets import MiewIdDataset

# def get_model(cfg, checkpoint_path=None, use_gpu=True):

#     model = MiewIdNet(**dict(cfg.model_params))


#     if use_gpu:
#         device = torch.device("cuda")
#         model.to(device)

#     if checkpoint_path:
#         model.load_state_dict(torch.load(checkpoint_path))
#         print('loaded checkpoint from', checkpoint_path)

#     return model

# def get_dataloader(df_data, images_dir, cfg, transforms, shuffle=True):
#     dataset = MiewIdDataset(
#         csv=df_data,
#         images_dir = images_dir,
#         transforms=transforms,
#     )


#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=cfg.engine.valid_batch_size,
#         num_workers=cfg.engine.num_workers,
#         shuffle=shuffle,
#         pin_memory=True,
#         drop_last=False,
#     )

#     return data_loader

# def get_embeddings_labels(data_loader,model,device, use_wandb=True):
#     model.eval()
#     tk0 = tqdm(data_loader, total=len(data_loader))
#     embeddings = []
#     labels = []
    
#     with torch.no_grad():
#         for batch in tk0:
#             batch_embeddings = model.extract_feat(batch["image"].to(CFG.device))
            
#             batch_embeddings = batch_embeddings.detach().cpu().numpy()
            
#             image_idx = batch["image_idx"].tolist()
#             batch_embeddings_df = pd.DataFrame(batch_embeddings, index=image_idx)
#             embeddings.append(batch_embeddings_df)

#             batch_labels = batch['label'].tolist()
#             labels.extend(batch_labels)
            
#     embeddings = pd.concat(embeddings)

#     return embeddings, labels