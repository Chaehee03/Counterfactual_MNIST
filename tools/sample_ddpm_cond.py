import torch
import torchvision
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from utils.config_utils import get_config_value, validate_class_config
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample(unet, scheduler, im, train_config, ldm_config, diffusion_config, vae, idx):

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
    sample_classes = torch.arange(num_classes, device=device)
    print('Generating images for class {}'.format(list(sample_classes.cpu().numpy())))
    cond_input = {
        'class': torch.nn.functional.one_hot(sample_classes, num_classes).to(device)
    }
    # Unconditional input for classifier-free guidance
    uncond_input = {
        'class': cond_input['class'] * 0
    }
    #############################################################

    # By default, classifier-free guidance is disabled. (cf_guidance_scale = 1.0)
    # To enable it, change value in config, or change below default value.
    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)

    # Repeat input image to match condition batch size
    im = im.repeat(len(sample_classes), 1, 1, 1)  # [10, 1, 28, 28]

    with torch.no_grad():
        z, _, _ = vae.encode(im)

    # Sample random noise
    noise = torch.randn_like(z).to(device)

    # Sample timestep
    t = torch.randint(0, diffusion_config['num_timesteps'], (z.shape[0],)).to(device)

    xt = scheduler.add_noise(z, noise, t)

    # predict noise at timestep & predict previous, x0 images
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        t = torch.full((xt.shape[0],), i, device=device, dtype=torch.long)
        # t = (torch.ones(xt.shape[0],)*i).long().to(device)
        noise_pred_cond = unet(xt, t, cond_input) # i(scalar) -> i(tensor) -> [i] (1D tensor, shape:(1,))

        if cf_guidance_scale > 1: # enabled
            noise_pred_uncond = unet(xt, t, uncond_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, t)

        # Save x0
        cf_image = torch.clamp(xt, -1., 1.).detach().cpu()
        if i == 0:
            # Decode only the final image to save time
            cf_image = vae.decode(x0_pred)
        else:
            cf_image = xt
        # restrict value range of tensor
        # detach current tensor from gradient calculation (no more backpropagation)
        # PIL require CPU environment

    cf_image = torch.clamp(cf_image, -1., 1.).detach().cpu()
    # [-1, 1] (representation for NN training) -> [0, 1] (representation to transform to PIL image)
    cf_image = (cf_image + 1) / 2

    with torch.no_grad():
        input_img = vae.decode(z)
    input_img = torch.clamp(input_img, -1., 1.).detach().cpu()
    cf_map = cf_image - input_img

    # Counterfactual Map Visualization
    save_dir = os.path.join(train_config['task_name'], 'tmp')
    os.makedirs(save_dir, exist_ok=True)
    temp_path = os.path.join(save_dir, 'temp_cf_map.png')
    grid_cf = make_grid(cf_map, nrow=5, normalize=True, scale_each=True).permute(1, 2, 0).squeeze().cpu().numpy()
    plt.figure(figsize=(6, 2))
    plt.axis('off')
    plt.imshow(grid_cf, cmap='seismic', vmin=-1, vmax=1)
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    img_cf = Image.open(temp_path)

    h_target = img_cf.height
    w_target = h_target

    # Counterfactual Image Visualization
    grid_im = make_grid(cf_image, nrow=5)  # 2 rows × 5 columns = 10 classes
    img = torchvision.transforms.ToPILImage()(grid_im) # PIL
    img = img.resize((img_cf.width, h_target))

    # Original Input Image Visualization
    img_input = torchvision.transforms.ToPILImage()(im[0].cpu())
    img_input_resized = img_input.resize((w_target, h_target))

    # 좌-중앙-우 이미지를 이어붙임
    total_width = img_input_resized.width + img_cf.width + img.width
    combined_img = Image.new('L', (total_width,  h_target ))
    combined_img.paste(img_input_resized, (0, 0))
    combined_img.paste(img_cf, (img_input_resized.width, 0))
    combined_img.paste(img, (img_input_resized.width + img_cf.width, 0))

    save_dir = os.path.join(train_config['task_name'], 'cf_im_samples_11')
    os.makedirs(save_dir, exist_ok=True)
    combined_img.save(os.path.join(save_dir, '{}.png'.format(idx)))

    combined_img.close() # free memories for image object

