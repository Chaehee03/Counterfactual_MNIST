import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.ADNI_dataset import AdniDataset
from torch.utils.data import DataLoader
from models.unet import UNet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

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
    model_config = config['model_params']
    train_config = config['train_params']

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=train_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Create the ataset
    adni = AdniDataset('train', im_path=dataset_config['im_path']) # load ADNI dataset
    # load data by batch unit
    # num_workers: the number of CPU threads
    adni_loader = DataLoader(adni, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)

    # Instantiate the model
    model = UNet(model_config).to(device) # load model to GPU/CPU
    model.train()

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    # Load checkpoint if found -> continue training from checkpoint
    if os.path.exists(os.path.join(train_config['task_name'], train_config['task_name'])):
        print("Loading checkpoint as found one")
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ckpt_name']), map_location=device))

    # Specify training parameters
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    # Run training
    for epoch_idx in range(num_epochs): # repeat for epoch times
        losses = []
        for im in tqdm(adni_loader): # train for batch unit
            optimizer.zero_grad() # initialize gradient
            im = im.float().to(device) # load data to fGPU/CPU

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t= torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t) # model predicts the noise

            loss = criterion(noise_pred, noise) # calculate loss
            losses.append(loss.item())
            loss.backward() # backpropagation
            optimizer.step() # update weights
        print("Finished epoch: {} | Loss: {:.4f}".format(
                epoch_idx + 1,
                np.mean(losses)
        ))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ckpt_name'])) # save model

    print('Done Training ...')


if __name__ == '__main__': # runs only when this file is run (not import)
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)