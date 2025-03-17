# Langevin Dynamics

Langevin dynamics describes the evolution of a system influenced by both deterministic and stochastic forces. It is formulated as the stochastic differential equation:

 $$
 dX_t = \frac{1}{2} \nabla \log (p(X_t)) dt + \sigma dW_t 
 $$

where $p(X)$ is the target probability density, $W_t$ is a Wiener process, and $\sigma$ controls the noise intensity. The first term represents drift towards regions of higher probability, while the second term introduces randomness, ensuring exploration of the state space.

As time progresses, the distribution of $X_t$ approaches $p(X)$, meaning that samples drawn from Langevin dynamics asymptotically resemble those from the target distribution.

## Code

### `GaussianMixture` Class Parameters:
- **`nmodes`** (default: `5`):
  - Specifies the number of components (modes) in the Gaussian Mixture Model (GMM). A higher number of modes means the mixture will have more clusters or distributions.
  
- **`dim`** (default: `2`):
  - The dimensionality of each mode in the GMM. For example, a value of 2 represents a 2D Gaussian mixture.

- **`scale`** (default: `8.0`):
  - Controls how far apart the means of each mode are. Larger values result in more widely separated modes in the mixture.

### `LangevinSampler` Class Parameters:
- **`sigma`** (default: `0.5`):
  - Defines the standard deviation for the noise term in the Langevin dynamics update. It controls how much randomness is added to the sampling steps.

- **`step_size`** (default: `0.1`):
  - Specifies the step size for each update in the Langevin sampling process. Smaller values result in finer steps, while larger values make the updates faster but less stable.

### Other Parameters:
- **`n_timesteps`** (set to `1000`):
  - The number of iterations for the Langevin sampling process, controlling how long the sampling procedure runs.

- **`final_time`** (set to `5.0`):
  - The total time for the Langevin sampling process. It is used to determine the `dt` (time step) by dividing `final_time` by `n_timesteps`.

- **`xt`** (initialization with random values):
  - Represents the initial sample points drawn from a standard uniform distribution. These points will be transformed by the Langevin sampler to match the target Gaussian Mixture Model.

- **`grid_x` and `grid_y`** (used for visualization):
  - Create a grid for plotting the probability distribution of the Gaussian Mixture Model. These values are used to generate contour plots that visualize the mixture distribution.

- **`plot_iters`** (set to 3 subplots):
  - Determines the number of subplots used for visualizing the sampling progress. The code plots the state of the Langevin sampling process at different iterations.

You can run the simulation with the following command:

``python langevin.py``
