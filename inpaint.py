import jax
from jax import numpy as jnp
import numpy as np
from typing import Tuple, Callable
from model import get_transformer_fn
import haiku as hk
from utils import load_pytree_from_dir, sample_to_string, string_to_sample
import argparse
from Bio import SeqIO, Seq
from huggingface_hub import list_repo_files, hf_hub_download
import os
import time
from tqdm import tqdm
import flax

# argparse setup
parser = argparse.ArgumentParser(description='Inpaint with a BFN')
parser.add_argument('--model', type=str, default='AbBFN', help='Name of the model; this can be either ProtBFN or AbBFN')
parser.add_argument('--force_reload', action = 'store_true', help='Force reload the model parameters')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--num_steps', type=int, default=100, help='Number of sampling steps')
parser.add_argument('--num_particles', type=int, default=128, help='Number of particles')
parser.add_argument('--num_samples_per_batch', type=int, default=1, help='Number of samples to inpaint per batch')
parser.add_argument('--input_file', type=str, default='example_inputs/sequences.fasta', help='Path to the input file')
parser.add_argument('--numbering_file', type=str, default='example_inputs/numbering.npy', help='Path to the ANARCI numbering file')
parser.add_argument('--region', type=str, default='CDR3', help='Region to inpaint')
parser.add_argument('--output_dir', type=str, default='samples', help='Output directory for samples')
parser.add_argument('--verbose', action = 'store_true', help='Whether to print every inpainted sample')
args = parser.parse_args()


def make_inpaint_fn(
    params: jax.Array,
    transformer: Callable[[jax.Array], jax.Array],
    num_steps: int = 100,
    num_particles: int = 1024,
    sample_length: int = 256,
) -> Callable[[jax.random.PRNGKey], jax.Array]:
    """ Create a function to sample from the model
    Args:
        params (jax.Array): parameters of the BFN
        transformer (Callable[[jax.Array], jax.Array]): function to apply the BFN
        num_steps (int): number of steps to sample
        num_particles (int): number of particles
        sample_length (int): length of the sample
    Returns:
        Callable[[jax.random.PRNGKey], jax.Array]: function to sample from the model
    """
    K = 32
    beta_1 = 2.0
    
    def inpaint_fn(
        key: jax.random.PRNGKey,
        x: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        """ Inpaint from the model using Algorithm 3 from the paper
        Args:
            key (jax.random.PRNGKey): random key
            x (jax.Array): observed sequence, of shape (sample_length,)
            mask (jax.Array): mask of the observed sequence, of shape (sample_length,)
        Returns:
            jax.Array: inpainted sample from the model, of shape (sample_length, K)
        """


        def step_particle(y: jax.Array, z: jax.Array, alpha: float, beta_s: float, key: jax.random.PRNGKey) -> Tuple[jax.Array, float]:
            """ Step an individual particle
            Args:
                y (jax.Array): current state, in logit space, of shape (sample_length, K)
                z (jax.Array): noise, of shape (sample_length, K)
                alpha (float): the change in beta for this step
                beta_s (float): beta at the end of the step, used for constructing y
                key (jax.random.PRNGKey): random key
            Returns:
                Tuple[jax.Array, float]: new state, logit for SMC
            """
            # Predict using the model 
            theta = jax.nn.softmax(y, axis=-1)
            phi_logits = transformer(
                params, key, theta,
            )
            phi = jax.nn.softmax(phi_logits, axis=-1)
            e_x = jax.nn.one_hot(x, num_classes=K, axis=1)
            squared_errors = jnp.sum((e_x - phi) ** 2, axis=-1)
            logit = -jnp.sum((alpha * K / 2) * squared_errors, axis=0, where=mask)
            # Force phi to x where the mask is 1
            phi = jnp.where(
                mask[:, None],
                jax.nn.one_hot(x, K),
                phi,
            )
            y = beta_s * (K * phi - 1) + jnp.sqrt(beta_s * K) * z
            return y, logit
        
        vectorised_step_particle = jax.vmap(
            step_particle, in_axes=(0,0,None,None,0),
        )

        # Fixed isotropic noise
        zs = jax.random.normal(key, (num_particles, sample_length, K))
        # Uniform prior expressed in logit space
        ys_0 = jnp.zeros((num_particles, sample_length, K))
        def step_fn(ys: jax.Array, args: Tuple[int, jax.random.PRNGKey]) -> Tuple[jax.Array, jax.Array]:
            """ Step function for sampling
            Args:
                ys (jax.Array): current state, in logit space, of shape (num_particles, sample_length, K)
                args (Tuple[int, jax.random.PRNGKey]): tuple of step index and random key
            Returns:
                Tuple[jax.Array, jax.Array]: new state, returned twice for API compatibility with scan
            """
            step_index, key = args
            t = step_index / num_steps
            s = (step_index + 1) / num_steps
            beta_t = beta_1 * t ** 2.0
            beta_s = beta_1 * s ** 2.0
            alpha = beta_s - beta_t
            
            # step each particle
            key_step, key_select = jax.random.split(key)
            ys, log_probs = vectorised_step_particle(ys, zs, alpha, beta_s, jax.random.split(key_step, num_particles))

            # Resample particles
            probs = jax.nn.softmax(log_probs)
            indices = jax.random.categorical(key_select, probs, axis = -1, shape = (num_particles,))

            ys = ys[indices]
            return ys, None
        
        # Run the sampling loop
        ys_1, _ = jax.lax.scan(step_fn, ys_0, (jnp.arange(num_steps), jax.random.split(key, num_steps)), length=num_steps)

        # Take the first particle
        y_1 = ys_1[0]

        # Perform a final inference step
        theta = jax.nn.softmax(y_1, axis=-1)
        phi_logits = transformer(
            params, key, theta,
        )
        phi = jax.nn.softmax(phi_logits, axis=-1)
        # Argmax over phi to get the most likely class for each variable
        return jnp.argmax(phi, axis=-1)
    return inpaint_fn

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
    num_samples_per_batch = num_samples_per_device * num_devices
    print(f"Sampling {num_samples_per_device} samples per device per batch of {num_samples_per_batch}.")

    # Prepare the inpainting function
    inpaint_fn = make_inpaint_fn(
        params=params,
        transformer=transformer.apply,
        num_steps=args.num_steps,
        num_particles=args.num_particles,
        sample_length=sample_length,
    )
    
    def batched_inpaint_fn(
        key: jax.random.PRNGKey,
        xs: jax.Array,
        masks: jax.Array,
    ) -> jax.Array:
        """ Batched version of inpaint_fn
        Args:
            key (jax.random.PRNGKey): random key
            xs (jax.Array): observed sequences, of shape (num_samples_per_device, sample_length)
            masks (jax.Array): masks of the observed sequences, of shape (num_samples_per_device, sample_length)
        Returns:
            jax.Array: inpainted sample from the model
        """
        keys = jax.random.split(key, num_samples_per_device)
        samples = jax.vmap(inpaint_fn)(keys, xs, masks)
        return samples

    pmapped_inpaint = jax.pmap(batched_inpaint_fn, in_axes=(0, 0, 0,))
    params = flax.jax_utils.replicate(params)
    key = jax.random.PRNGKey(args.seed)

    # Load inputs and convert to integer representations + masks
    numberings = np.load(args.numbering_file)
    samples =  SeqIO.parse(args.input_file, "fasta")
    seqs = [sample for sample in samples]

    xs = []
    masks = []
    for seq, numbering in zip(seqs, numberings):
        if args.region == "CDR":
            mask = np.sum(numbering[[0, 1, 2]], axis=0)
        elif args.region == "FR":
            mask = np.sum(numbering[[3, 4, 5, 6]], axis=0)
        else:
            numbering_index = ["CDR1", "CDR2", "CDR3", "FR1", "FR2", "FR3", "FR4"].index(args.region)
            mask = numbering[numbering_index]
        mask = jnp.clip(mask, a_min = 0, a_max = 1)
        mask = 1 - jnp.pad(mask, (0, sample_length - len(mask)))
        masks.append(mask)
        x = string_to_sample(str(seq.seq), sample_length)
        xs.append(x)

    masks = jnp.stack(masks, axis = 0)
    xs = jnp.stack(xs, axis = 0)

    # Handle mismatch between batch size and number of samples
    num_samples = len(xs)
    print(f"Number of samples to inpaint is {num_samples}")
    num_batches = int(jnp.ceil(num_samples / num_samples_per_batch))
    effective_sequences = num_batches * num_samples_per_batch
    print(f"Padding input to length {effective_sequences} for batching")

    xs = jnp.concatenate(
        [
            xs, 
            jnp.zeros((effective_sequences - num_samples, sample_length), dtype = xs.dtype)
        ], axis = 0
    )
    masks = jnp.concatenate(
        [
            masks, 
            jnp.zeros((effective_sequences - num_samples, sample_length), dtype = masks.dtype)
        ], axis = 0
    )

    # Sample the model
    print("Begin sampling")
    all_samples = []
    for batch in tqdm(range(num_batches)):
        key, key_batch = jax.random.split(key, 2)
        start = batch * num_samples_per_batch
        end = (batch + 1) * num_samples_per_batch
        x_batch = xs[start: end]
        mask_batch = masks[start: end]
        keys = jax.device_put_sharded(list(jax.random.split(key_batch, num_devices)), devices)
        x_batch = jax.device_put_sharded(list(x_batch.reshape(num_devices, num_samples_per_device, sample_length)), devices)
        mask_batch = jax.device_put_sharded(list(mask_batch.reshape(num_devices, num_samples_per_device, sample_length)), devices)
        batch_inpainted = pmapped_inpaint(keys, x_batch, mask_batch)
        # Gather and flatten across devices
        batch_inpainted = jax.device_get(batch_inpainted)
        batch_inpainted = batch_inpainted.reshape(-1, sample_length)
        all_samples.append(batch_inpainted)
    samples = jnp.concatenate(all_samples, axis=0)
    samples = samples[:num_samples]

    # Convert samples to string format
    samples = [
        sample_to_string(sample) for sample in samples
    ]
    print("Sampling completed")

    print("Number of inpainted samples: ", len(samples))
    print(f"Writing samples to disk: {args.output_dir}/inpainted_samples.fasta")

    # Save the samples using SeqIO
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    write_seqs = []
    aars = []

    for i in range(num_samples):
        original_seq = seqs[i]
        x = sample_to_string(xs[i])
        x_inpainted = samples[i]
        mask = masks[i]
        # Calculate the number of mismatches
        errors = sum([0 if x1 == x2 else 1 for x1, x2 in zip(x, x_inpainted)])
        # AAR is number of correct predictions
        aar = 1.0 - errors / sum(1 - mask)
        aars.append(aar)
        if args.verbose: 
            # Print each inpainted sequence
            match_string = "".join([
                " " if m == 1 else ("|" if x1 == x2 else "-") for x1, x2, m in zip(x, x_inpainted, mask)
            ])
            mask_string = "".join([
                "X" if m == 0 else " " for i, m in enumerate(mask) if i < len(x_inpainted)
            ])
            print(x)
            print(mask_string)
            print("\nInpainted to:\n")
            print(x_inpainted)
            print(match_string)
            print(f"\nAAR: {aar}\n")
        seq = SeqIO.SeqRecord(Seq.Seq(x_inpainted), id=f"{original_seq.id}-inpainted", description=f"inpainted with AAR {aar}")
        write_seqs.append(seq)
    SeqIO.write(write_seqs, f"{args.output_dir}/inpainted_samples.fasta", "fasta")
    aars = jnp.array(aars)
    print(f"Average AAR: {jnp.mean(aars)}")

