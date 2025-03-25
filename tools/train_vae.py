import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
import tqdm

from dataset.adni_dataset import AdniDataset
from models.lpips import LPIPS
from models.vae import VAE
from models.discriminator import Discriminator
from torch.utils.data.dataloader import DataLoader
from dataset.mnist_dataset import MnistDataset
from dataset.adni_dataset import AdniDataset
from torch.optim import Adam
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def kl_divergence(mean, logvar):
    return 0.5 * torch.sum(mean.pow(2) + logvar.exp() - logvar - 1)


def train(args):
    # Read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    vae_config = config['vae_params']
    train_config = config['train_params']

    # Set the desired seed value
    seed = train_config['seed']
    torch.manual_seed(seed) # Pytorch
    np.random.seed(seed) # Numpy
    random.seed(seed) # Python
    if device == 'cuda': # Cuda
        torch.cuda.manual_seed_all(seed) # for cases using several GPUs

    # Create the model
    model = VAE(im_channels=dataset_config['im_channels'],
                model_config=vae_config).to(device)

    # Create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'adni': AdniDataset,
    }.get(dataset_config['name'])

    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size = dataset_config['im_size'],
                                im_channels = dataset_config['im_channels'])

    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['vae_batch_size'],
                             shuffle=True)

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.makedirs(train_config['task_name'])

    num_epochs = train_config['vae_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Discriminator Loss (Least Squares GAN)
    disc_criterion = torch.nn.MSELoss()

    # Instantiate LPIPS and Disc
    # freezing part is in lpips.py -> don't need to freeze lpips
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(in_channels=dataset_config['im_channels']).to(device)

    optimizer_g = Adam(model.parameters(), lr=train_config['vae_lr'], betas = (0.5, 0.999))
    optimizer_d = Adam(discriminator.parameters(), lr = train_config['vae_lr'], betas = (0.5, 0.999))

    # step point when discriminator kicks in. (starts to train generating fake images)
    disc_step_start = train_config['disc_step_start']
    step_count = 0

    # Gradient Accumulation
    # Useful when dealing with high-resolution images
    # And when a large batch size is unavailable.
    acc_steps = train_config['vae_acc_steps']
    image_save_steps = train_config['vae_image_save_steps']
    image_save_count = 0

    for epoch in range(num_epochs):
        recon_losses = []
        kl_losses = []
        perceptual_losses = []
        adversarial_losses = []
        disc_losses = []
        gen_losses = []

        # Initialize gradients to zeros
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        for im in tqdm(data_loader):
            step_count += 1
            im = im.float().to(device)

            # Fetch VAE output(reconstructed images)
            model_output = model(im)
            out, mean, logvar = model_output

            # Image Saving Logic #
            # print [Original Input - Model Output] pair
            if step_count % image_save_steps == 0 or step_count == 1: # save initially & per image_save_steps.
                sample_size = min(8, im.shape[0]) # save max 8 samples
                # model output image normalization for saving
                save_output = torch.clamp(out[:sample_size], -1, 1).detach().cpu()
                save_output = (save_output + 1) / 2 # [-1, 1] -> [0, 1]
                # original input image normalization for saving
                save_input = ((im[:sample_size] + 1) / 2).detach().cpu()

                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid) # Pytorch tensor -> PIL image

                # create output directory
                if not os.path.exists(os.path.join(train_config['task_name'], 'vae_samples')):
                    os.mkdir(os.path.join(train_config['task_name'], 'vae_samples',
                                          'current_vae_sample_{}.png'.format(image_save_count)))
                image_save_count += 1
                img.close()

            ############## Optimize Generator ############
            # L2 Loss
            recon_loss = recon_criterion(out, im)
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps # divide loss by acc_steps before backprop -> effect of calculating gradients partially

            # KL divergence loss(VAE)
            kl_loss = kl_divergence(mean, logvar)/ acc_steps

            g_loss = recon_loss + train_config['kl_weight'] * kl_loss
            kl_losses.append(train_config['kl_weight'] * kl_loss.item())

            # Adversarial loss term is added after step count passed disc_step_start.
            if step_count > disc_step_start:
                disc_fake_pred = discriminator(model_output[0])
                # Generators takes only a batch size of fake samples for training.
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,
                                                           device = disc_fake_pred.device)) # ground truth(G): fake -> 1
                adversarial_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps

            # LPIPS
            lpips_loss = torch.mean(lpips_model(out, im)) # batch-wise mean of (B, 1, 1, 1) tensors.
            perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
            g_loss += train_config['perceptual_weight'] * lpips_loss / acc_steps

            gen_losses.append(g_loss.item()) # total generator's loss (recon + kl + adversarial + lpips)

            g_loss.backward() # calculate gradient
            ##############################################

            ############ Optimize Discriminator ###########
            # Discriminator starts training after step_count passed disc_step_start.
            if step_count > disc_step_start:
                fake = out
                disc_fake_pred = discriminator(fake.detach()) # detach fake(generator's gradient) from backpropagation
                disc_real_pred = discriminator(im)
                # Discriminator takes both fake and real samples and train classifying them.
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.zeros(disc_fake_pred.shape, # ground truth(D): fake -> 0
                                                            device = disc_fake_pred.device))
                disc_real_loss = disc_criterion(disc_real_pred,
                                                torch.ones(disc_real_pred.shape, # ground truth(D): real -> 1
                                                           device = disc_real_pred.device))
                # Total loss = Adding BCE of y=0 (ground truth: fake) & y=1 (ground truth: real)
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps

                disc_loss.backward() # backprop

                # After accumulating for acc_steps, update discriminator's params once.
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad() # Initialize gradient to zero before next batch iteration.
            ################################################

            # After accumulating for acc_steps, update generator's params once.
            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()

        # After entire dataset training
        optimizer_d.step() # Update D's params
        optimizer_d.zero_grad() # Initialize before next epoch
        optimizer_g.step() # Update G's params
        optimizer_g.zero_grad() # Initialize before next epoch

        # print results per 1 epoch
        # disc_losses can be empty if D kicks in after training G for entire dataset.
        if len(disc_losses) > 0:
            print(
                'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                'KL Divergence Loss : {:.4f} | Adversarial Loss: {:.4f} '
                '| G Loss : {:.4f} | D Loss {:.4f}'.
                format(epoch + 1,
                       np.mean(recon_losses),
                       np.mean(perceptual_losses),
                       np.mean(kl_losses),
                       np.mean(adversarial_losses),
                       np.mean(gen_losses),
                       np.mean(disc_losses)))
        else:
            print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                  'KL Divergence Loss : {:.4f} | G Loss : {:.4f}'.
                  format(epoch + 1,
                         np.mean(recon_losses),
                         np.mean(perceptual_losses),
                         np.mean(kl_losses),
                         np.mean(gen_losses)))

        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['vae_autoencoder_ckpt_name']))
        torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                            train_config['vae_discriminator_ckpt_name']))
    print('Done Training...')

    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Arguments for vae training')
        parser.add_argument('--config', dest='config_path',
                            default='config/mnist.yaml', type=str)
        args = parser.parse_args()
        train(args)
