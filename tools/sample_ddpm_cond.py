import torch
import torchvision
import yaml
import argparse
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_cond import UNet
from models.vae import VAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import get_config_value, validate_class_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample(model, scheduler, train_config, ldm_config,
           vae_config, diffusion_config, dataset_config, vae):

    im_size = dataset_config['im_size'] // 2**sum(vae_config['down_sample'])

    xt = torch.randn((train_config['num_samples'],
                      vae_config['z_channels'],
                      im_size,
                      im_size)).to(device)

    ################### Validate the config ###################
    condition_config = get_config_value(ldm_config, key='condition_config', default_value=None)
    assert condition_config is not None, ("This sampling script is for class conditioning, "
                                          "but condition config not found")
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'class' in condition_types, ("This sampling script is for class conditioning, "
                                         "but class condition not found in config")
    validate_class_config(condition_config)
    ############################################################

    ################# Create Conditional Input ##################
    num_classes = condition_config['class_condition_config']['num_classes']
    sample_classes = torch.randint(0, num_classes, (train_config['num_samples'],))
    print('Generating images for class {}'.format(list(sample_classes.numpy())))
    cond_input = {
        'class': torch.nn.functional.one_hot(sample_classes, num_classes).to(device) # 1D tensor of classes to be one-hot encode, total length of 1 one-hot vector
    }
    # Unconditional input for classifier-free guidance
    uncond_input = {
        'class': cond_input['class'] * 0 # makes one-hot encoded vector all zero vector
    }
    #############################################################

    # By default, classifier-free guidance is disabled. (cf_guidance_scale = 1.0)
    # To enable it, change value in config, or change below default value.
    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)

    # predict noise at timestep & predict previous, x0 images
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        t = (torch.ones(xt.shape[0],)*i).long().to(device)
        noise_pred_cond = model(xt, t, cond_input) # i(scalar) -> i(tensor) -> [i] (1D tensor, shape:(1,))

        if cf_guidance_scale > 1: # enabled
            noise_pred_uncond = model(xt, t, uncond_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        if i == 0:
            # Decode only the final image to save time
            ims = vae.decode(x0_pred)
        else:
            ims = xt
        # restrict value range of tensor
        # detach current tensor from gradient calculation (no more backpropagation)
        # PIL require CPU environment
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        # [-1, 1] (representation for NN training) -> [0, 1] (representation to transform to PIL image)
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows']) # number of images per 1 row
        img = torchvision.transforms.ToPILImage()(grid) # PIL (pytorch imaging library object

        if not os.path.exists(os.path.join(train_config['task_name'], 'samples_cond')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples_cond'))
        img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
        img.close() # free memories for image object

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

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Load pretrained weights to VAE #
    vae = VAE(im_channels=dataset_config['im_channels'],
              model_config=vae_config).to(device)
    vae.eval()

    # Load pretrained vae if checkpoint exists
    vae_checkpoint = os.path.join(train_config['task_name'],
                                  train_config['vae_autoencoder_ckpt_name'])
    if os.path.exists(vae_checkpoint):
        print('Loaded vae checkpoint')
        vae.load_state_dict(torch.load(vae_checkpoint, map_location=device)['model_state_dict'])

    # Load pretrained weights to UNet #
    unet = UNet(im_channels=vae_config['z_channels'],
                 model_config=ldm_config).to(device)

    # evaluation mode (deactivate drop-out, normalize by trained total mean & variance)
    unet.eval()

    # Checkpoint must exist in infer time.
    if os.path.exists(os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])):
        print('Loaded U-Net checkpoint')
        # torch.load(.pth): Load pretrained wights(OrderedDict type) saved in .pth file.
        # pretrained wights(OrderedDict type) -> state_dict
        # model.load_state_dict(): Load pretrained wights to instantiated model.
        unet.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ldm_ckpt_name']), map_location=device), strict=False)


    with torch.no_grad(): # deactivate autograd (not training -> not save gradient)
        sample(unet, scheduler, train_config, ldm_config,
               vae_config, diffusion_config, dataset_config, vae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    infer(args)