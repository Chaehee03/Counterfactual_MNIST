import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.unet_cond import UNet
from models.vae import VAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *
from utils.diffusion_utils import *
from models.classifier import MnistClassifier
from models.lpips import LPIPS

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
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Instantiate Condition related components
    condition_types = [] # extends for demogrphic informaitons
    condition_config = get_config_value(ldm_config, key='condition_config', default_value = None)

    # Create the dataset #
    im_dataset_cls = {
        'mnist': MnistDataset,
        # 'adni': AdniDataset
    }.get(dataset_config['name'])

    im_dataset = im_dataset_cls(split = 'train',
                                im_path = dataset_config['im_path'],
                                im_size = dataset_config['im_size'],
                                im_channels = dataset_config['im_channels'],
                                use_latents=False,
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
        vae = VAE(im_channels=dataset_config['im_channels'],
                  model_config=vae_config).to(device)
        vae.eval() # don't need training, just create latents

        # Load trained vae if checkpoint exists
        if os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['vae_autoencoder_best_ckpt_name'])):
            print('Loaded vae checkpoint')
            ckpt = torch.load(os.path.join(train_config['task_name'],
                                           train_config['vae_autoencoder_best_ckpt_name']),
                              map_location=device)
            vae.load_state_dict(ckpt['model_state_dict'])

    classifier = MnistClassifier().to(device)
    classifier_path = os.path.join(train_config['task_name'],
                                   train_config['classifier_ckpt_name'])
    classifier.load_state_dict(torch.load(classifier_path))
    classifier.eval()

    lpips_model = LPIPS().to(device)
    lpips_model.eval()

    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(unet.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()

    # Freeze VAE parameters
    # VAE is pre-trained & only used for generating latents.
    # It is relevant only when latents don't exist.
    if not im_dataset.use_latents:
        for param in vae.parameters():
            param.requires_grad = False

    best_val = float("inf")
    min_delta = 1e-4
    patience = 10
    wait = 0

    # Run training
    for epoch_idx in range(num_epochs): # repeat for epoch times
        total_losses = []
        mse_losses = []
        cls_losses = []
        cf_map_losses = []
        perceptual_losses = []
        latent_losses = []
        tv_losses = []

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
                    im, _, _ = vae.encode(im)

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

            ########## hyperparameters ##########
            mse_weight = train_config['mse_weight']
            cls_weight = train_config['cls_weight']
            l1_weight = train_config['cf_l1_weight']
            l2_weight = train_config['cf_l2_weight']
            # cf_perceptual_weight = train_config['cf_perceptual_weight']
            # latent_dist_weight_1 = train_config['latent_dist_weight_1']
            # latent_dist_weight_2 = train_config['latent_dist_weight_2']
            # tv_weight = train_config['tv_weight']

            if epoch_idx < 10:
                cf_perceptual_weight = 0
                latent_dist_weight_1 = 0
                latent_dist_weight_2 = 0
                tv_weight = 0
            elif epoch_idx < 20:
                cf_perceptual_weight = 0.0005
                latent_dist_weight_1 = 0.0005
                latent_dist_weight_2 = 0.00005
                tv_weight = 0.00005
            elif epoch_idx < 30:
                cf_perceptual_weight = 0.001
                latent_dist_weight_1 = 0.001
                latent_dist_weight_2 = 0.0001
                tv_weight = 0.0005
            else:
                cf_perceptual_weight = 0.002
                latent_dist_weight_1 = 0.002
                latent_dist_weight_2 = 0.0002
                tv_weight = 0.001

            # === 기본 MSE loss (DDPM noise prediction loss) ===
            loss_mse = criterion(noise_pred, noise)  # calculate loss
            mse_losses.append(loss_mse.item())

            # === classification loss ===
            xt, x0_pred = scheduler.sample_prev_timestep(noisy_im, noise_pred, t)
            cf_latent = x0_pred
            cf_image = vae.decode(cf_latent)
            logits = classifier(cf_image) # (B, 10)

            target_label = cond_input['class'].argmax(dim=1).long() # (B,)

            # mask for samples that actually have condition
            valid_idx = (cond_input['class'].sum(dim=1) != 0)
            if valid_idx.any():
                logits_valid = logits[valid_idx]
                target_label_valid = target_label[valid_idx]
                loss_cls = F.cross_entropy(logits_valid, target_label_valid)
            else:
                loss_cls = torch.zeros_like(logits.sum())
            cls_losses.append(loss_cls.item())

            # Counterfactual Map Loss
            input_img = vae.decode(im)
            cf_map = cf_image - input_img
            loss_map = (l1_weight * torch.mean(torch.abs(cf_map)) + l2_weight * torch.mean(cf_map ** 2)).to(device)
            cf_map_losses.append(loss_map.item())

            # Perceptual Loss
            perceptual_loss = torch.mean(lpips_model(cf_image, input_img))
            perceptual_losses.append(perceptual_loss.item())

            # Latent Distance Loss
            loss_latent = latent_dist_weight_1 * torch.mean(torch.abs(cf_latent - im)) + \
                          latent_dist_weight_2 * torch.mean((cf_latent - im) ** 2)
            latent_losses.append(loss_latent.item())

            # Total Variation
            tv_loss = torch.mean(torch.abs(cf_image[:, :, :-1, :] - cf_image[:, :, 1:, :])) + \
                      torch.mean(torch.abs(cf_image[:, :, :, :-1] - cf_image[:, :, :, 1:]))
            tv_losses.append(tv_loss.item())


            # === total loss ===
            loss = mse_weight * loss_mse + cls_weight * loss_cls + loss_map + \
                   cf_perceptual_weight * perceptual_loss + loss_latent + tv_weight * tv_loss
            total_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print(
            'Finished epoch: {} | Total Loss: {:.4f}| '
            'MSE Loss : {:.4f} | CLS Loss : {:.4f} | '
            'CF Map Loss : {:.4f} | Perceptual Loss : {:.4f} | '
            'Latent Loss : {:.4f} | Total TV Loss : {:.4f} '.
            format(epoch_idx + 1,
                   np.mean(total_losses),
                   np.mean(mse_losses),
                   np.mean(cls_losses),
                   np.mean(cf_map_losses),
                   np.mean(perceptual_losses),
                   np.mean(latent_losses),
                   np.mean(tv_losses)))

        epoch_loss = np.mean(total_losses)

        if epoch_loss < best_val - min_delta:  # 미세한 수치 오차 허용
            best_val = epoch_loss
            wait = 0
            torch.save(unet.state_dict(), os.path.join(train_config['task_name'],
                                                       train_config['ldm_best_ckpt_name']))
            print(f" Best checkpoint saved (epoch_loss = {best_val:.4g}, epoch = {epoch_idx + 1})")
        else:
            wait += 1
            if wait >= patience:
                print(f" Early-stopping at epoch {epoch_idx + 1} (no val improvement for {patience} epochs)")
                torch.save(unet.state_dict(), os.path.join(train_config['task_name'],
                                                           train_config['ldm_ckpt_name']))
                return

        torch.save(unet.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ldm_ckpt_name'])) # save model

    print('Done Training ...')


if __name__ == '__main__': # runs only when this file script is run (don't run when imported)
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)