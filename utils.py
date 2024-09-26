import os
import numpy as np
from jax import numpy as jnp
import jax

def load_pytree_from_dir(base_dir: str) -> dict:
    """ Load a pytree from a directory
    Args:
        base_dir (str): path to the directory
    Returns:
        dict: pytree
    """
    tree_def = np.load(os.path.join(base_dir, 'tree_def.npy'), allow_pickle=True).item()
    flat = []
    i = 0
    while True:
        array_path = os.path.join(base_dir, f'array_{i}.npy')
        if not os.path.exists(array_path):
            break
        flat.append(jnp.array(np.load(array_path)))
        i += 1
    return jax.tree_util.tree_unflatten(tree_def, flat)

id_to_token = [
   "<unk>",
   "<pad>",
   "<mask>",
   "<cls>",
   "<eos>",
   "<bos>",
   "A",
   "R",
   "N",
   "D",
   "C",
   "Q",
   "E",
   "G",
   "H",
   "I",
   "L",
   "K",
   "M",
   "F",
   "P",
   "S",
   "T",
   "W",
   "Y",
   "V",
   "X",
   "B",
   "Z",
   "J",
   "U",
   "O",
]

def sample_to_string(sample: jax.Array) -> str:
    """ Convert a sample to a string
    Args:
        sample (jax.Array): sample 
    Returns:
        str: string representation of the sample
    """
    string = ""
    for i in sample:
        if i == 4:
            break
        if i > 5:
            string += id_to_token[i]
    return string

def string_to_sample(sample: str, length: int) -> jax.Array:
    """ Convert a string to a sample
    Args:
        sample (str): string representation of the sample
        length (int): length of the sample
    Returns:
        jax.Array: sample of shape (length,)
    """
    result = id_to_token.index("<pad>") * np.ones(length, dtype=jnp.int32)
    for i, token in enumerate(sample):
        result[i] = id_to_token.index(token)
    # End with EOS
    if len(sample) < length:
        result[len(sample)] = 4
    return jnp.array(result)

def count_sequential_repeats(string: str) -> dict:
    """ Count the number of sequential repeats in a string
    Args:
        string (str): input string
    Returns:
        dict: dictionary with the counts of sequential repeats
    """

    if not string:
        return {}

    repeats = {}
    current_char = string[0]
    count = 1

    for char in string[1:]:
        if char == current_char:
            count += 1
        else:
            if count > 3:
                if current_char in repeats:
                    repeats[current_char].append(count)
                else:
                    repeats[current_char] = [count]
            current_char = char
            count = 1

    if count > 3:
        if current_char in repeats:
            repeats[current_char].append(count)
        else:
            repeats[current_char] = [count]
    return repeats

def repetition_score(string: str) -> float:
    """ Compute the repetition score of a string
    Args:
        string (str): input string
    Returns:
        float: repetition score
    """
    counts = count_sequential_repeats(string)
    per_char_score = [np.sum(counts[char]) for char in counts.keys()]
    return np.sum(per_char_score) / len(string)