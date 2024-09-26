import jax
from jax import numpy as jnp
from typing import Callable, Dict, Tuple

def sample_sender_distribution(x: jax.Array, t: float, beta_1: float, key: jax.random.PRNGKey) -> jax.Array:
    """ Sample from the sender distribution
    Args:
        x (jax.Array): original sample of shape [N] where N is the length of the sample
        t (float): time at which to sample
        beta_1 (float): final precision of the BFN
        key (jax.random.PRNGKey): random key
    """
    beta = beta_1 * t ** 2.0
    K = 32
    e_x = jax.nn.one_hot(x, num_classes=K, axis = -1)
    mu = beta * (K * e_x - 1)
    sigma = jnp.sqrt(beta * K)
    return mu + sigma * jax.random.normal(key, mu.shape)

def compute_continuous_time_loss(x: jax.Array, phi_logits: jax.Array, t: float, beta_1: float) -> float:
    """ Compute the continuous-time loss for a given output distribution phi and original sample x
    Args:
        x (jax.Array): original sample of shape [N] where N is the length of the sample
        phi_logits (jax.Array): logits of the output distribution of shape [N, K] where K is the number of classes
        t (float): time at which to compute the continuous-time loss
        beta_1 (float): final precision of the BFN
    Returns:
        float: The continuous-time loss at time t
    """
    # Compute the target distribution
    K = 32
    target = jax.nn.one_hot(x, num_classes=K, axis = -1)
    # beta(t) = beta_1 * t^2 -> alpha = 2 * beta_1 * t
    alpha = 2.0 * beta_1 * t 
    # Compute the continuous-time loss
    phi = jax.nn.softmax(phi_logits, axis = -1)
    loss = jnp.sum(jnp.sum(0.5 * K * alpha * (target - phi) ** 2, axis=-1))
    return loss

def compute_reconstruction_loss(x: jax.Array, phi_logits: jax.Array) -> float:
    """ Compute the reconstruction loss for a given output distribution phi and original sample x
    Args:
        x (jax.Array): original sample of shape [N] where N is the length of the sample
        phi (jax.Array): output distribution of shape [N, K] where K is the number of classes
    Returns:
        float: The reconstruction loss
    """
    x_one_hot = jax.nn.one_hot(x, num_classes=32, axis = -1)
    loss_per_variable = -jnp.sum(x_one_hot * jax.nn.log_softmax(phi_logits), axis = -1)
    loss = jnp.sum(loss_per_variable, axis = 0)
    return loss

def approximate_loss(
    x: jax.Array,
    transformer_fn: Callable[[Dict, jax.random.PRNGKey, jax.Array], jax.Array],
    parameters: Dict,
    key: jax.random.PRNGKey,
    beta_1: float = 2.0,
    num_approximations: int = 1000,
    ) -> float:
    """ Compute the approximate loss for a given sample x by doing a monte-carlo approximation over time
    Args:
        x (jax.Array): original sample of shape [N] where N is the length of the sample
        transformer_fn (Callable[[Dict, jax.random.PRNGKey, jax.Array]): function to apply the BFN
        parameters (Dict): parameters of the BFN
        key (jax.random.PRNGKey): random key
        beta_1 (float): final precision of the BFN
        num_approximations (int): number of approximations to use
    Returns:
        float: The approximate loss
    """
    continuous_time_loss = 0.0

    def step_loss_fn(continuous_time_loss: float, key: jax.random.PRNGKey) -> Tuple[float, float]:
        """ Step function for computing the loss. This method samples a random point in time,
        samples the sender distribution, and computes the contiuous-time loss at that point
        Args:
            total_loss (float): current loss
            key (jax.random.PRNGKey): random key
        Returns:
            Tuple[float, float]: new loss (twice)
        """
        key_t, key_sender, key_model = jax.random.split(key, 3)
        # Pick a time to sample
        t = jax.random.uniform(key_t, (1,))[0]

        # Sample from the sender distribution
        sender_sample = sample_sender_distribution(x, t, beta_1, key_sender)
        theta = jax.nn.softmax(sender_sample, axis=-1)

        # Apply the transformer to get the output distribution
        phi_logits = transformer_fn(parameters, key_model, theta)
        # Compute the continous-time loss and average over the approximations
        continuous_time_loss += compute_continuous_time_loss(x, phi_logits, t, beta_1) / num_approximations

        return continuous_time_loss, continuous_time_loss
    
    key_continuous_time, key_reconstruction = jax.random.split(key, 2)
    # Compute the continous-time loss
    continuous_time_loss, _ = jax.lax.scan(step_loss_fn, continuous_time_loss, jax.random.split(key_continuous_time, num_approximations))

    # Compute the reconstruction loss at t = 1 -- this will be very small
    key_sender, key_model = jax.random.split(key_reconstruction, 2)
    t = 1.0
    sender_sample = sample_sender_distribution(x, t, beta_1, key_sender)
    theta = jax.nn.softmax(sender_sample, axis=-1)
    phi_logits = transformer_fn(parameters, key_model, theta)
    reconstruction_loss = compute_reconstruction_loss(x, phi_logits)

    return continuous_time_loss + reconstruction_loss





