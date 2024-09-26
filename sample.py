import os
import argparse
import time
from tqdm import tqdm
from typing import Tuple, Callable

import jax
from jax import numpy as jnp
import haiku as hk
import flax
from huggingface_hub import list_repo_files, hf_hub_download

from model import get_transformer_fn
from loss import approximate_loss
from utils import load_pytree_from_dir, sample_to_string, repetition_score
from Bio import SeqIO, Seq

# argparse setup
parser = argparse.ArgumentParser(description='Sample from a BFN')
parser.add_argument('--model', type=str, default='ProtBFN', help='Name of the model; this can be either ProtBFN or AbBFN')
parser.add_argument('--force_reload', action = 'store_true', help='Force reload the model parameters')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--num_steps', type=int, default=10000, help='Number of sampling steps')
parser.add_argument('--num_samples_per_batch', type=int, default=1, help='Number of samples to generate per batch')
parser.add_argument('--num_batches', type=int, default=1, help='Number of batches to generate')
parser.add_argument('--filter_samples', action = 'store_true', help='Filter samples by perplexity and repetivity')
parser.add_argument('--perplexity_threshold', type=float, default=7.786, help='perplexity threshold for filtering')
parser.add_argument('--repetivity_threshold', type=float, default=0.0207, help='Repetivity threshold for filtering')
parser.add_argument('--output_dir', type=str, default='samples', help='Output directory for samples')
args = parser.parse_args()


def make_sample_fn(
    params: jax.Array,
    transformer: Callable[[jax.Array], jax.Array],
    num_steps: int = 10000,
    sample_length: int = 512,
) -> Callable[[jax.random.PRNGKey], jax.Array]:
    """ Create a function to sample from the model
    Args:
        params (jax.Array): parameters of the BFN
        transformer (Callable[[jax.Array], jax.Array]): function to apply the BFN
        num_steps (int): number of steps to sample
        sample_length (int): length of the sample
        K (int): number of classes
    Returns:
        Callable[[jax.random.PRNGKey], jax.Array]: function to sample from the model
    """
    K = 32
    beta_1 = 2.0

    def sample_fn(
        key: jax.random.PRNGKey,
    ) -> jax.Array:
        """ Sample from the model using Algorithm 2 from the paper
        Args:
            key (jax.random.PRNGKey): random key
        Returns:
            jax.Array: sample from the model, of shape (sample_length, K)
        """
        # Fixed isotropic noise
        z = jax.random.normal(key, (sample_length, K))
        # Uniform prior expressed in logit space
        y_0 = jnp.zeros((sample_length, K))

        def step_fn(y: jax.Array, args: Tuple[int, jax.random.PRNGKey]) -> Tuple[jax.Array, jax.Array]:
            """ Step function for sampling
            Args:
                y (jax.Array): current state, in logit space, of shape (sample_length, K)
                args (Tuple[int, jax.random.PRNGKey]): tuple of step index and random key
            Returns:
                Tuple[jax.Array, jax.Array]: new state, returned twice for API compatibility with scan
            """
            step_index, key = args
            s = (step_index + 1) / num_steps
            # Theta is the distribution over the variables, given as the softmax over latent variable y
            theta = jax.nn.softmax(y, axis=-1)
            # The transformer returns the logits of the output distribution
            phi_logits = transformer(
                params, key, theta,
            )
            phi = jax.nn.softmax(phi_logits, axis=-1)

            # Compute the beta value for this step
            beta_s = beta_1 * s ** 2.0

            # Update the state
            y = beta_s * (K * phi - 1) + jnp.sqrt(beta_s * K) * z

            return y, y
        
        # Run the sampling loop to get y at time 1
        y_1, _ = jax.lax.scan(step_fn, y_0, (jnp.arange(num_steps), jax.random.split(key, num_steps)), length=num_steps)

        # Perform a final inference step
        theta = jax.nn.softmax(y_1, axis=-1)
        phi_logits = transformer(
            params, key, theta,
        )
        phi = jax.nn.softmax(phi_logits, axis=-1)

        # Argmax over phi to get the most likely class for each variable
        return jnp.argmax(phi, axis=-1)
    
    return sample_fn

def make_loss_fn(
    params: jax.Array,
    transformer: Callable[[jax.Array], jax.Array],
    ) -> Callable[[jax.Array], float]:
    """ Create a function to compute the loss of a sample
    Args:
        params (jax.Array): parameters of the BFN
        transformer (Callable[[jax.Array], jax.Array]): function to apply the BFN
        beta_1 (float): final precision of the BFN
    Returns:
        Callable[[jax.Array], float]: function to compute the loss of a sample
    """
    beta_1 = 2.0

    def loss_fn(
        x: jax.Array, key_loss: jax.random.PRNGKey
    ) -> float:
        """ Compute the loss of a sample
        Args:
            x (jax.Array): sample of shape (sample_length,)
        Returns:
            float: loss of the sample
        """
        # Compute the loss
        return approximate_loss(
            x=x,
            transformer_fn=transformer,
            parameters=params,
            key=key_loss,
            beta_1=beta_1,
            num_approximations=1000,
        )
    
    return loss_fn

if __name__ == "__main__":

    # Download the model parameters if needed
    if not os.path.exists("parameters"):
        os.makedirs("parameters")
    if os.path.exists(f"parameters/{args.model}") and not args.force_reload:
        print(f"Model weights for {args.model} already downloaded and stored in parameters/{args.model}")
    else:
        # Download the model from huggingface hub
        print(f"Downloading model weights to local directory parameters/{args.model}")
        t1 = time.time()
        # List all files in the repository
        repo_id = "InstaDeepAI/protein-sequence-bfn"
        files = list_repo_files(repo_id)
        # Filter files in the specific folder
        folder_files = [file for file in files if file.startswith(args.model)]
        # Download each file in the folder
        for file in tqdm(folder_files):
            hf_hub_download(repo_id, file, local_dir="parameters")
        t2 = time.time()
        print(f"\nModel weights downloaded successfully in {t2-t1:.2f} seconds")

    # Number of classes for the BFN
    K = 32
    # Sample length is fixed dependent on the model
    sample_length = 512 if args.model == "ProtBFN" else 256

    # Instantiate the model
    transformer = get_transformer_fn(
        output_dim = K,
    )
    transformer = hk.transform(transformer)
    # We don't need the parameter dictionary returned by this as we are loading a checkpoint
    _ = transformer.init(
        jax.random.PRNGKey(0), jnp.ones((sample_length, K))
    )

    # Load the parameters from a checkpoint
    model_path = f"parameters/{args.model}"
    params = load_pytree_from_dir(model_path)

    # Set up devices.
    num_hosts = jax.device_count() // jax.local_device_count()

    devices = jax.local_devices()
    num_devices = len(devices)
    print(f"Sampling across {num_hosts} hosts.")

    if num_hosts > 1:
        # Warning
        print("Warning! This code is not configured to work in a multi-host setting. You should not expect consistent results. ")

    print(f"Found {num_devices} local devices.")
    
    num_samples_per_device = int(jnp.ceil(args.num_samples_per_batch / num_devices))
    print(f"Sampling {num_samples_per_device} samples per device per batch.")


    # Prepare the sampling function
    sample_fn = make_sample_fn(
        params=params,
        transformer=transformer.apply,
        num_steps=args.num_steps,
        sample_length=sample_length,
    )

    # Prepare the loss function
    loss_fn = make_loss_fn(
        params=params,
        transformer=transformer.apply,
    )

    def sample_and_approximate_loss(
        key: jax.random.PRNGKey,
    ) -> Tuple[jax.Array, float]:
        """ Sample from the model and compute the approximate loss
        Args:
            key (jax.random.PRNGKey): random key
        Returns:
            Tuple[jax.Array, float]: sample from the model and approximate loss
        """
        key_sample, key_loss = jax.random.split(key)
        x = sample_fn(
            key=key_sample,
        )
        loss = loss_fn(
            x=x,
            key_loss=key_loss,
        )
        return x, loss
    
    def batched_sample_and_approximate_loss(
        key: jax.random.PRNGKey,
    ) -> Tuple[jax.Array, float]:
        """ Batched version of sample_and_approximate_loss
        Args:
            key (jax.random.PRNGKey): random key
        Returns:
            Tuple[jax.Array, float]: sample from the model and approximate loss
        """
        keys = jax.random.split(key, num_samples_per_device)
        samples, losses = jax.vmap(sample_and_approximate_loss)(keys)
        return samples, losses

    pmapped_sample_and_approximate_loss = jax.pmap(batched_sample_and_approximate_loss, in_axes=(0,))
    params = flax.jax_utils.replicate(params)
    key = jax.random.PRNGKey(args.seed)

    # Sample the model
    print("Begin sampling")
    all_samples = []
    all_losses = []
    for batch in tqdm(range(args.num_batches)):
        key, key_batch = jax.random.split(key, 2)
        keys = jax.device_put_sharded(list(jax.random.split(key_batch, num_devices)), devices)
        samples, losses = pmapped_sample_and_approximate_loss(
            keys,
        )
        # Gather and flatten across devices
        samples = jax.device_get(samples)
        losses = jax.device_get(losses)
        samples = samples.reshape(-1, sample_length)
        losses = losses.reshape(-1)
        all_samples.append(samples)
        all_losses.append(losses)
    samples = jnp.concatenate(all_samples, axis=0)
    losses = jnp.concatenate(all_losses, axis=0)

    # Convert samples to string format
    samples = [
        sample_to_string(sample) for sample in samples
    ]
    print("Sampling completed")

    # Filter samples
    filtered_samples = []
    filtered_losses = []
    filtered_perplexities = []
    num_filtered = 0
    for sample, loss in zip(samples, losses):
        sample_length = len(sample)
        sample_perplexity = jnp.exp(loss / sample_length)
        if not args.filter_samples or (sample_perplexity < args.perplexity_threshold and repetition_score(sample) < args.repetivity_threshold):
            filtered_samples.append(sample)
            filtered_losses.append(loss)
            filtered_perplexities.append(sample_perplexity)
        else:
            num_filtered += 1
    samples = filtered_samples
    losses = filtered_losses
    perplexities = filtered_perplexities

    print("Number of samples filtered: ", num_filtered)
    print("Number of samples remaining: ", len(samples))
    print(f"Writing samples to disk: {args.output_dir}/samples.fasta")

    # Save the samples using SeqIO
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    seqs = []

    for i, sample in enumerate(samples):
        seq = SeqIO.SeqRecord(Seq.Seq(sample), id=f"sample_{i}", description=f"loss: {losses[i]:.2f}, perplexity: {perplexities[i]:.2f}")
        seqs.append(seq)
    SeqIO.write(seqs, f"{args.output_dir}/samples.fasta", "fasta")

