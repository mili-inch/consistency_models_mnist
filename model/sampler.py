import torch as th


class Sampler:
    def __init__(self, sigma_data=0.5, sigma_max=80, sigma_min=0.002, rho=7, num_scales=200):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.num_scales = num_scales
    
    def get_t(self, indices):
        t = self.sigma_max ** (1 / self.rho) + indices / (self.num_scales - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        t = t ** self.rho
        return t

    def get_scalings(self, sigmas):
        c_skip = self.sigma_data**2 / (sigmas**2 + self.sigma_data**2)
        c_out = sigmas * self.sigma_data / (sigmas**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigmas**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in
    
    def denoise(self, model, x_t, sigmas, conds):
        c_skip, c_out, c_in = [x[..., None, None, None] for x in self.get_scalings(sigmas)]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, conds)
        denoised = c_out * model_output + c_skip * x_t
        return denoised
    
    def training_losses(self, model, x_start, conds):
        noise = th.randn_like(x_start)
        indices = th.randint(0, self.num_scales - 1, (x_start.shape[0],), device=x_start.device)

        t = self.get_t(indices)
        t2 = self.get_t(indices + 1)

        x_t = x_start + noise * t[..., None, None, None]
        x_t2 = x_start + noise * t2[..., None, None, None]

        distiller = self.denoise(model, x_t, t, conds)
        distiller_target = self.denoise(model, x_t2, t2, conds).detach()

        snrs = t ** -2
        weights = snrs + 1 / self.sigma_data ** 2

        diffs = (distiller - distiller_target) ** 2

        loss = diffs.mean(dim=(1, 2, 3), keepdim=False) * weights
        loss = loss.mean()
        return loss
    
    @th.no_grad()
    def sample(self, model, batch_size, conds):
        noise = th.randn(batch_size, 1, 28, 28, device="cuda")
        indices = th.arange(self.num_scales, device="cuda")
        t = self.get_t(indices)
        x_t = noise * t[0].unsqueeze(0)
        for i in range(1, self.num_scales):
            x_t = self.denoise(model, x_t, t[i].unsqueeze(0).repeat(batch_size), conds)
            if i == self.num_scales - 1:
                return x_t
            x_t = x_t + noise * t[i][None, None, None, None]
