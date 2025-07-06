import yaml
import argparse
from models.unet_cond import UNet
from models.vae import VAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from tools.sample_ddpm_cond import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    ldm_config = config['ldm_params']
    vae_config = config['vae_params']
    train_config = config['train_params']

    ################### Validate the config ###################
    condition_config = get_config_value(ldm_config, key='condition_config', default_value=None)
    assert condition_config is not None, ("This sampling script is for class conditioning, "
                                          "but condition config not found")
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'class' in condition_types, ("This sampling script is for class conditioning, "
                                        "but class condition not found in config")
    validate_class_config(condition_config)
    ############################################################

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Create the dataset #
    im_dataset_cls = {
        'mnist': MnistDataset,
        # 'adni': AdniDataset
    }.get(dataset_config['name'])

    im_dataset = im_dataset_cls(split='test',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                use_latents=False,
                                latent_path=os.path.join(train_config['task_name'],
                                                         train_config['vae_latent_dir_name']),
                                condition_config=condition_config,
                                )

    # load data by batch unit
    # num_workers: the number of CPU threads
    data_loader = DataLoader(im_dataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=4)

    # Load pretrained weights to VAE #
    vae = VAE(im_channels=dataset_config['im_channels'],
              model_config=vae_config).to(device)
    vae.eval()

    # Load pretrained vae if checkpoint exists
    vae_checkpoint = os.path.join(train_config['task_name'],
                                  train_config['vae_autoencoder_best_ckpt_name'])
    if os.path.exists(vae_checkpoint):
        print('Loaded vae checkpoint')
        vae.load_state_dict(torch.load(vae_checkpoint, map_location=device)['model_state_dict'])

    # Load pretrained weights to UNet #
    unet = UNet(im_channels=vae_config['z_channels'],
                model_config=ldm_config).to(device)

    # evaluation mode (deactivate drop-out, normalize by trained total mean & variance)
    unet.eval()

    # Checkpoint must exist in infer time.
    # unet_checkpoint = '/mnist/cf_map_best_ldm_4.pth'
    unet_checkpoint = os.path.join(train_config['task_name'], train_config['ldm_best_ckpt_name'])
    if os.path.exists(unet_checkpoint):
        print('Loaded U-Net checkpoint:', unet_checkpoint)
        # torch.load(.pth): Load pretrained wights(OrderedDict type) saved in .pth file.
        # pretrained wights(OrderedDict type) -> state_dict
        # model.load_state_dict(): Load pretrained wights to instantiated model.
        unet.load_state_dict(torch.load(unet_checkpoint, map_location=device), strict=False)


    for idx, data in enumerate(tqdm(data_loader)):
            cond_input = None
            if condition_config is not None:
                im, cond_input = data
            else:
                im = data
            im = im.float().to(device)

            with torch.no_grad():  # deactivate autograd (not training -> not save gradient)
                sample(unet, scheduler, im, train_config, ldm_config, diffusion_config, vae, idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    infer(args)