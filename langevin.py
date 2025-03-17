import torch
import torch.distributions as D 
import numpy as np
import math

import matplotlib.pyplot as plt


class GaussianMixture:
    def __init__(self, nmodes=5, dim=2, scale=8.0):
        self.nmodes = nmodes
        self.dim = dim
        
        # Equal weights for each mode
        mix_weights = torch.ones(nmodes) / nmodes
        self.mix = D.Categorical(mix_weights)
        
        # Create covariance matrices
        covs = (torch.ones((nmodes, dim)) + 2) / 2 
        self.covs_matrices = torch.diag_embed(covs)
        
        # Create means that are well-separated
        self.means = (torch.rand((nmodes, dim)) * 2 - 1) * scale
        self.comp = D.MultivariateNormal(self.means, self.covs_matrices)
        
        # Create the mixture model
        self.gmm = D.MixtureSameFamily(self.mix, self.comp)
    
    def log_prob(self, x):
        return self.gmm.log_prob(x)
    
    def sample(self, sample_shape=torch.Size([])):
        return self.gmm.sample(sample_shape)


class LangevinSampler:
    def __init__(self, sigma=0.5, step_size=0.1):
        self.sigma = sigma 
        self.step_size = step_size

    def step(self, xt, score):
        drift = 0.5 * (self.sigma ** 2) * score * self.step_size 
        diffusion = self.sigma * torch.randn_like(xt) * math.sqrt(self.step_size) 
        return xt + drift + diffusion 


if __name__ == "__main__":
    gmm = GaussianMixture(nmodes=4)

    n_timesteps = 1000
    final_time = 5.0
    dt = final_time / n_timesteps
    langevin_sampler = LangevinSampler(sigma=1.0, step_size=dt)

    # Sample points to be transformed from N(0, I) to the mixture distribution 
    xt = torch.rand((2000, 2), requires_grad=False) * 20 - 10
    x0s = xt[:, 0]
    x1s = xt[:, 1]

    # These points are used to visualize the mixture distribution 
    grid_x, grid_y = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))
    grid = np.dstack((grid_x, grid_y))
    grid_z = np.exp(gmm.log_prob(torch.tensor(grid)))

    num_subplots = 3
    plot_iters = np.linspace(0, n_timesteps - 1, num_subplots)
    plot_iters_dict = {int(iter) for iter in plot_iters}
    # Add one subplot to plot the final iteration
    fig, axes = plt.subplots(1, num_subplots, figsize=(20, 5))
    plot_frequency = n_timesteps // num_subplots

    plot_idx = 0
    for i in range(n_timesteps):
        if i in plot_iters_dict:

            ax = axes[plot_idx]

            # Plot the filled contour map with pixelated and fuzzy effect
            ax.imshow(grid_z, extent=(-10, 10, -10, 10), origin='lower', cmap='Oranges', alpha=1)
            ax.contour(grid_x, grid_y, grid_z, levels=torch.logspace(-20, 10, 25), colors='black', linewidths=2, alpha=0.1)

            # Plot the points in xt that are being updated by the langevin sampler
            ax.scatter(x0s, x1s, c='grey', edgecolor='black', s=10, alpha=0.5)

            # Set axis limits
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

            ax.set_title(f'Iteration {i+1}')

            plot_idx += 1

        # Compute the Jacobian/score
        xt.requires_grad_(True)
        # Compute the sum of log probs to 1) be able to use autograd backward, and 2) avoid calculating full Jacobian
        log_probs =  gmm.log_prob(xt).sum()
        score = torch.autograd.grad(log_probs, xt)[0]
        xt = xt.detach()
        xt = langevin_sampler.step(xt, score)
        x0s = xt[:, 0]
        x1s = xt[:, 1]

    plt.tight_layout()
    plt.show()