# Langevin Dynamics

Langevin dynamics describes the evolution of a system influenced by both deterministic and stochastic forces. It is formulated as the stochastic differential equation:

 $$
 dX_t = \frac{1}{2} \nabla \log (p(X_t)) dt + \sigma dW_t 
 $$

where $p(X)$ is the target probability density, $W_t$ is a Wiener process, and $\sigma$ controls the noise intensity. The first term represents drift towards regions of higher probability, while the second term introduces randomness, ensuring exploration of the state space.

As time progresses, the distribution of $X_t$ approaches $p(X)$, meaning that samples drawn from Langevin dynamics asymptotically resemble those from the target distribution.


