import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from dataset.adni_dataset import AdniDataset
from torch.utils.data import DataLoader
from models.unet import UNet
from models.vae import VAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *
from utils.diffusion_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file: # open config file
        try:
            config = yaml.safe_load(file) # YAML file -> python dictionary
        except yaml.YAMLError as exc: # fail to open config file -> exception
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    ldm_config = config['ldm_params']
    vae_config = config['vae_params']
    train_config = config['train_params']

    # Create the noise scheduler #
    scheduler = LinearNoiseScheduler(num_timesteps=train_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Instantiate Condition related components
    condition_types = [] # extends for demogrphic informaitons
    condition_config = get_config_value(ldm_config, key='condition_config', default_value = None)

    # Create the dataset #
    im_dataset_cls = {
        'mnist': MnistDataset,
        'adni': AdniDataset
    }.get(dataset_config['name'])

    im_dataset = im_dataset_cls(split = 'train',
                                im_path = dataset_config['im_path'],
                                im_size = dataset_config['im_size'],
                                im_channels = dataset_config['im_channels'],
                                use_latents=True,
                                latent_path=os.path.join(train_config['task_name'],
                                                         train_config['vae_latent_dir_name']),
                                condition_config = condition_config,
                                )

    # load data by batch unit
    # num_workers: the number of CPU threads
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['ldm_batch_size'],
                             shuffle=True,
                             num_workers=4)

    # Instantiate the model
    unet = UNet(im_channels=vae_config['z_channels'],
                 model_config=ldm_config).to(device) # load model to GPU/CPU
    unet.train()

    # Load VAE only if latents are not present. (use_latents = False)
    if not im_dataset.use_latents:
        print('Loading vae model as latents are not present.')
        vae = VAE(im_channels=vae_config['im_channels'],
                  model_config=ldm_config).to(device)
        vae.eval() # don't need training, just create latents

        # Load trained vae if checkpoint exists
        if os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['vae_autoencoder_ckpt_name'])):
            print('Loaded vae checkpoint')
            vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                        train_config['vae_autoencoder_ckpt_name']),
                                           map_location=device))

    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()

    # Freeze VAE parameters
    # VAE is pre-trained & only used for generating latents.
    # It is relevant only when latents don't exist.
    if not im_dataset.use_latents:
        for param in vae.parameters():
            param.requires_grad = False

    # Run training
    for epoch_idx in range(num_epochs): # repeat for epoch times
        losses = []
        for data in tqdm(data_loader): # train for batch unit
            cond_input = None
            if condition_config is not None:
                im, cond_input = data
            else:
                im = data
            optimizer.zero_grad() # initialize gradient
            im = im.float().to(device) # load data to GPU/CPU

            # It is relevant only when latents don't exist.
            if not im_dataset.use_latents:
                with torch.no_grad():
                    im, _, _ = vae.encode(im) # im: latents

            ########## Handling Conditional Input ##########
            if 'class' in cond_input:
                validate_class_config(condition_config)
                class_condition = torch.nn.functional.one_hot(
                    cond_input['class'],
                    condition_config['class_condition_config']['num_classes']).to(device)
                class_drop_prob = get_config_value(condition_config['class_condition_config'],
                                                'cond_drop_prob', 0.)
                # Drop condition
                cond_input['class'] = drop_class_condition(class_condition, class_drop_prob, im)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t= torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = unet(noisy_im, t, cond_input = cond_input) # U-Net predicts the noise

            loss = criterion(noise_pred, noise) # calculate loss
            losses.append(loss.item())
            loss.backward() # backpropagation
            optimizer.step() # update weights
        print("Finished epoch: {} | Loss: {:.4f}".format(
                epoch_idx + 1,
                np.mean(losses)
        ))
        torch.save(unet.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ldm_ckpt_name'])) # save model

    print('Done Training ...')


if __name__ == '__main__': # runs only when this file script is run (don't run when imported)
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)