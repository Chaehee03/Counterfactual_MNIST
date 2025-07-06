import torch

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1-self.alpha_cum_prod)

    # forward process
    def add_noise(self, original, noise, t):
        batch_size = original.shape[0]
        dims = (batch_size,) + (1,) * (original.dim() - 1)  # ex (batch,1,1,1)
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].view(dims)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].view(dims)
        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise

    # reverse process
    def sample_prev_timestep(self, xt, noise_pred, t):
        batch_size = xt.shape[0]
        dims = (batch_size,) + (1,) * (xt.dim() - 1)

        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t].view(dims)
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(xt.device)[t].view(dims)
        betas = self.betas.to(xt.device)[t].view(dims)
        alphas = self.alphas.to(xt.device)[t].view(dims)

        x0 = (xt - sqrt_one_minus_alpha_cum_prod * noise_pred) / sqrt_alpha_cum_prod
        x0 = torch.clamp(x0, -1., 1.)
        mean = (xt - (betas * noise_pred) / sqrt_one_minus_alpha_cum_prod) / torch.sqrt(alphas)

        if isinstance(t, torch.Tensor) and (t == 0).all():
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]).view(dims) / \
                       (1 - self.alpha_cum_prod.to(xt.device)[t]).view(dims)
            variance = variance * betas
            sigma = torch.sqrt(variance)
            z = torch.randn_like(xt)
            return mean + sigma * z, x0

