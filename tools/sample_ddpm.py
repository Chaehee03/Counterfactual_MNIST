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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample(model, scheduler, train_config, model_config, diffusion_config):
    xt = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
    # predict noise at timestep & predict previous, x0 images
    for i in tqdm(reversed(range(train_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device)) # i(scalar) -> i(tensor) -> [i] (1D tensor, shape:(1,))

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, train_config)

        # Save x0
        # restrict value range of tensor
        # detach current tensor from gradient calculation (no more backpropagation)
        # PIL require CPU environment
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        # [-1, 1] (representations for NN training) -> [0, 1] (representations of original images)
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows']) # number of images per 1 row
        img = torchvision.transforms.ToPILImage()(grid) # PIL (pytorch imaging library object
        if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples'))
        img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
        img.close() # free memories for image object

def infer(args):
    # Read the config file #
    with open(args.config, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_config']
    model_config = config['model_config']
    train_config = config['train_config']

    # Load model with checkpoint
    model = UNet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ckpt_name']), map_location=device))
    # evaluation mode (deactivate drop-out, normalize by trained total mean & variance)
    model.eval()

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    with torch.no_grad(): # deactivate autograd (not training -> not save gradient)
        sample(model, scheduler, train_config, model_config, diffusion_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)
























































































































































































if __name__ == '__main__': # Run only when this file is run
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)