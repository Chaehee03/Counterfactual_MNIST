# import argparse
# import glob
# import os
# import pickle
#
# import torch
# import torchvision
# import yaml
# from torch.utils.data.dataloader import DataLoader
# from torchvision.utils import make_grid
# from tqdm import tqdm
#
# from dataset.mnist_dataset import MnistDataset
# # from dataset.adni_dataset import AdniDataset
# from models.vae import VAE
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # This script is for checking latent representations of VAE.
# def infer(args):
#     # Read the config file
#     with open(args.config_path, 'r') as file:
#         try:
#             config = yaml.safe_load(file)
#         except yaml.YAMLError as exc:
#             print(exc)
#     print(config)
#
#     dataset_config = config['dataset_params']
#     vae_config = config['vae_params']
#     train_config = config['train_params']
#
#     # Create the dataset
#     im_dataset_cls = {
#         'mnist': MnistDataset,
#         # 'adni': AdniDataset
#     }.get(dataset_config['name'])
#
#     im_dataset = im_dataset_cls(split='train',
#                                 im_path=dataset_config['im_path'],
#                                 im_size=dataset_config['im_size'],
#                                 im_channels=dataset_config['im_channels'])
#
#     data_loader = DataLoader(im_dataset,
#                              batch_size=1,
#                              shuffle=False)
#
#     num_images = train_config['num_samples']
#     ngrid = train_config['num_grid_rows']
#
#     idxs = torch.randint(0, len(im_dataset) - 1, (num_images,))
#     ims = torch.cat([im_dataset[idx][None, :] for idx in idxs]).float()
#     ims = ims.to(device)
#
#     # Load pretrained VAE model
#     model = VAE(im_channels=dataset_config['im_channels'],
#                 model_config=vae_config).to(device)
#     model.load_state_dict(torch.load(os.path.join(train_config['tack_name'],
#                                                   train_config['vae_autoencoder_ckpt_name']),
#                                      map_location=device))
#     model.eval()
#
#     with torch.no_grad():
#
#         ims = (ims + 1) / 2
#
#         encoded_output, _ , _ = model.encode(ims)
#         encoded_output = torch.clamp(encoded_output, min=-1., max=1.)
#         encoded_output = (encoded_output + 1) / 2
#
#         decoded_output = model.decode(encoded_output)
#         decoded_output = torch.clamp(decoded_output, min=-1., max=1.)
#         decoded_output = (decoded_output + 1) / 2
#
#
#         original_grid = make_grid(ims.cpu(), nrow=ngrid)
#         original_grid = torchvision.transforms.ToPILImage()(original_grid)
#
#         encoded_grid = make_grid(encoded_output.cpu(), nrow=ngrid)
#         encoded_grid = torchvision.transforms.ToPILImage()(encoded_grid)
#
#         decoded_grid = make_grid(decoded_output.cpu(), nrow=ngrid)
#         decoded_grid = torchvision.transforms.ToPILImage()(decoded_grid)
#
#         original_grid.save(os.path.join(train_config['task_name'], 'original_samples.png'))
#         encoded_grid.save(os.path.join(train_config['task_name'], 'encoded_samples.png'))
#         decoded_grid.save(os.path.join(train_config['task_name'], 'reconstructed_samples.png'))
#
#         # To save latents, change configuration save_latents = True.
#         if train_config['save_latents']:
#             latent_path = os.path.join(train_config['task_name'], train_config['vae_latent_dir_name'])
#             latent_fnames = glob.glob(os.path.join(train_config['task_name'], train_config['vae_latent_dir_name'], '*.pkl'))
#             assert len(latent_fnames) == 0, 'Latents already present. Delete all latent files and re-run.'
#             if not os.path.exists(latent_path):
#                 os.makedirs(latent_path)
#             print('Saving latents for {}'.format(dataset_config['name']))
#
#             fname_latent_map = {}
#             part_count = 0
#             count = 0
#             for idx, im in enumerate(tqdm(data_loader)):
#                 encoded_output, _, _ = model.encode(im.float().to(device))
#                 fname_latent_map[im_dataset.images[idx]] = encoded_output.cpu()
#                 # Save latents every 1000 images
