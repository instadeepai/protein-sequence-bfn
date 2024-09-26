import haiku as hk
from haiku import initializers
import jax
from jax import numpy as jnp
from typing import Callable, Optional
from functools import partial

# Partial instantiation of the GELU activation function without approximation
gelu_activation_fn = partial(jax.nn.gelu, approximate=False)

class RobertaHead(hk.Module):

    def __init__(self, embed_dim: int, num_outputs: int, name: Optional[str] = None):
        """ Roberta head. Transform final attention layer output into a
        distribution over tokens at each position.
        Args:
            embed_dim (int): Embedding dimension.
            num_outputs (int): Number of output classes.
            name (Optional[str]): Name of the layer. Defaults to None.
        """
        super().__init__(name=name)

        # Define layers
        self._first_layer_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="emb_layer_norm_after"
        )
        self._fc1 = hk.Linear(embed_dim, name="lm_head_fc_1")
        self._final_fc = hk.Linear(num_outputs, name="lm_final_fc")
        self._second_layer_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="lm_head_layer_norm"
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self._first_layer_norm(x)
        x = self._fc1(x)
        x = gelu_activation_fn(x)
        x = self._second_layer_norm(x)
        logits = self._final_fc(x)
        return logits

class MultiHeadAttention(hk.Module):
    def __init__(
        self,
        num_heads: int,
        qkv_size: int,
        name: Optional[str] = None,
    ):
        """ Multi-head attention with Rotary embeddings (see RoFormer https://arxiv.org/pdf/2104.09864.pdf)
        Args:
            num_heads: Number of independent attention heads.
            qkv_size: The size of keys and queries used for attention.
            name: Optional name for this module.
        """
        super().__init__(
            name=name,
        )
        self._num_heads = num_heads
        self._qkv_size = qkv_size

        # Constant used in Sinusoidal/Rotary Embeddings, reference to this value can be found
        # on page 6 of https://arxiv.org/pdf/1706.03762.pdf and page 5 of
        # https://arxiv.org/abs/2104.09864
        UPPER_FREQ = 10000
        self._inv_freq = 1.0 / (UPPER_FREQ ** (jnp.arange(0, self._qkv_size, 2) / self._qkv_size))


    def rotary_embedding(self, heads: jax.Array) -> jax.Array:
        """ Applies rotary embeddings to the given heads
        Args:
            heads (jax.Array): The heads tensor.
        Returns:
            (jax.Array): The heads tensor with rotary embeddings applied.
        """

        seq_len = heads.shape[0]
        t = jnp.arange(seq_len)
        freqs = jnp.einsum("i,j->ij", t, self._inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)

        # Compute cos and cast is as (seq_len, 1, key_size) to be applied to queries
        # of shape (seq_len, num_heads, key_size)
        sin_emb = jnp.sin(emb)[:,None,:]
        cos_emb = jnp.cos(emb)[:,None,:]
        x1, x2 = heads[..., : heads.shape[-1] // 2], heads[..., heads.shape[-1] // 2 :]
        heads_rotated = jnp.concatenate((-x2, x1), axis=-1)

        embedded_heads = (heads * cos_emb) + (heads_rotated * sin_emb)
        return embedded_heads

    @hk.transparent
    def attention_weights(
        self,
        x: jax.Array,
    ) -> jax.Array:
        """
        Computes the attention weights.

        Args:
            x (jax.Array): Embedding sequence to compute keys/queries of shape [sequence_length, embed_dim].

        Returns:
            (jax.Array): Attention weights of shape [num_heads, sequence_length, sequence_length].
        """
        # Project query and keys into [sequence_length, num_heads, qkv_size] shape
        query_heads = self._linear_projection_he_init(x, self._qkv_size, "query")
        key_heads = self._linear_projection_he_init(x, self._qkv_size, "key")

        # Apply rotary embedding
        query_heads = self.rotary_embedding(query_heads)
        key_heads = self.rotary_embedding(key_heads)

        # Compute attention logits and rescale
        attention_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        sqrt_key_size = jnp.sqrt(self._qkv_size).astype(x.dtype)
        attention_logits = attention_logits / sqrt_key_size

        # Softmax to give attention weights
        attention_weights = jax.nn.softmax(attention_logits)

        return attention_weights

    @hk.transparent
    def compute_embeddings(
        self,
        value: jax.Array,
        attention_weights: jax.Array,
    ) -> jax.Array:
        """
        Computes the output embeddings.

        Args:
            x (jax.Array): Embedding sequence to compute values of shape [sequence_length, embed_dim].
            attention_weights (jax.Array): Attention weights of shape [num_heads, sequence_length, sequence_length].

        Returns:
            (jax.Array): Output embeddings.
        """

        # He initialization
        w_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")
        b_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")

        value_heads = self._linear_projection_he_init(value, self._qkv_size, "value")

        attention = jnp.einsum("...htT,...Thd->...thd", attention_weights, value_heads)

        # Concatenate attention matrix of all heads into a single vector.
        attention_vec = jnp.reshape(attention, (*attention.shape[:-2], -1))
        return hk.Linear(
            self._qkv_size * self._num_heads, w_init=w_init, b_init=b_init, name="mha_output"
        )(attention_vec)

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        """
        Computes Multi-head attention mechanism

        Args:
            x (jax.Array): Embedding sequence to apply attention mechanism.
        Returns:
            (jax.Array): Output embeddings.
        """

        attention_weights = self.attention_weights(x)
        embeddings = self.compute_embeddings(x, attention_weights)
        return embeddings

    @hk.transparent
    def _linear_projection_he_init(
        self, x: jax.Array, head_size: int, name: Optional[str] = None
    ) -> jax.Array:
        """
        Linear layer for multi-head attention mechanism. Initialized with the He method.

        Args:
            x (jax.Array): Input embeddings.
            head_size (int): Embedding size of each attention head.
            name (Optional[str]): Name of the linear layer.

        Returns:
            (jax.Array): Multi-head embeddings.
        """

        # He initialization
        w_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")
        b_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")

        y = hk.Linear(
            self._num_heads * head_size, w_init=w_init, b_init=b_init, name=name
        )(x)
        return y.reshape((*x.shape[:-1], self._num_heads, head_size))


class SelfAttentionBlock(hk.Module):

    def __init__(
        self, 
        num_heads: int,
        embed_dim: int,
        ffn_embed_dim: int,
        name: Optional[str] = None,
    ) -> None:
        """ Self attention block
        Args:
            num_heads (int): The number of attention heads
            embed_dim (int): The embedding dimension
            ffn_embed_dim (int): The embedding dimension of the feed-forward network
            name (Optional[str]): The name of the module
        """
        super().__init__(name=name)

        self._num_heads = num_heads
        self._embed_dim = embed_dim
        self._ffn_embed_dim = ffn_embed_dim

        # Even split of embed dim into num heads
        self._qkv_size = self._embed_dim // self._num_heads


        self.fc1 = hk.Linear(ffn_embed_dim, name="fc1")
        self.fc2 = hk.Linear(embed_dim, name="fc2")

        self.layer_norm_self_attention = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name="self_attention_layer_norm",
        )
        self.layer_norm_mlp = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="final_layer_norm"
        )
        self.sa_layer = MultiHeadAttention(
            num_heads=self._num_heads, 
            qkv_size=self._qkv_size,
            name="self_attention",
        )

    @hk.transparent
    def mlp(self, x: jax.Array) -> jax.Array:
        x = self.layer_norm_mlp(x)
        x = gelu_activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x

    def __call__(self, x: jax.Array) -> jax.Array:
        """ Forward pass through the self-attention block.
        Args:
            x (jax.Array): The input sequence of shape [sequence_length, embed_dim]
        Returns:
            (jax.Array): The output sequence of shape [sequence_length, embed_dim]
        """
        # Store x for residual connection
        res = x

        # Self-Attention
        x = self.layer_norm_self_attention(x)
        x = self.sa_layer(
            x=x,
        )
        x = res + x

        # MLP
        x = x + self.mlp(x)
        return x
    
class Transformer(hk.Module):

    def __init__(
            self, 
            output_dim: int,
            embed_dim: int = 1280,
            ffn_embed_dim: int = 5120,
            num_layers: int = 33,
            attention_heads: int = 20,
            name: Optional[str] = None) -> None:
        """ Transformer architecture as described in Methods -> Pre-training ProtBFN -> Model

        Args:
            output_dim (int): The output dimension e.g. number of classes
            embed_dim (int): The embedding dimension
            ffn_embed_dim (int): The embedding dimension of the feed-forward network
            num_layers (int): The number of transformer layers
            attention_heads (int): The number of attention heads
            name (Optional[str]): The name of the module
        """
        super().__init__(name = name)

        self._embed_dim = embed_dim
        self._ffn_embed_dim = ffn_embed_dim
        self._num_layers = num_layers
        self._attention_heads = attention_heads
        self._output_dim = output_dim

        # Linear layer embeds theta into a higher-dimensional space
        self._embed_layer = hk.Linear(self._embed_dim)
        self._emb_ln_before = hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True,
                name="emb_layer_norm_before",
            )

        # Final MLP that transforms the output to the target output dimension
        self._lm_head = RobertaHead(
            embed_dim=self._embed_dim,
            num_outputs=self._output_dim,
            name="roberta_lm_head",
        )

    @hk.transparent
    def apply_attention_blocks(
        self,
        x: jax.Array,
    ) -> jax.Array:
        """
        Create the blocks of attention layers and applies them.

        Args:
            x (jax.Array): The sequence embedding of shape (sequence_length, embed_dim)

        Returns:
            The output sequence embedding of shape (sequence_length, embed_dim)
        """

        layers = [
            self._attention_block(layer_idx)
            for layer_idx in range(self._num_layers)
        ]
        for layer in layers:
            x = layer(
                x,
            )
        return x
    
    @hk.transparent
    def _attention_block(self, layer_idx: int) -> SelfAttentionBlock:
        return SelfAttentionBlock(
            num_heads=self._attention_heads,
            embed_dim=self._embed_dim,
            ffn_embed_dim=self._ffn_embed_dim,
            name=f"attention_layer_{layer_idx}",
        )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """ Forward pass through the model.
        Args:
            x (jax.Array): The input of shape [sequence_length, input_dim]
        Returns:
            logits (jax.Array): The output logits of shape [sequence_length, num_classes]
        """
        # Embed x in self._embed_dim space
        x = self._embed_layer(x)
        x = self._emb_ln_before(x)

        # Pass x through the attention blocks
        x = self.apply_attention_blocks(x)

        # Apply the LM head
        logits = self._lm_head(x)
        return logits
    
def apply_entropy_encoding(theta: jax.Array) -> jax.Array:
    """ Apply a per-variable entropy encoding to theta.
    Args:
        theta (jax.Array): The input theta tensor
    Returns:
        jax.Array: The encoded theta tensor
    """
    # Compute the per-variable entropy of theta
    entropy = -jnp.sum(theta * jnp.log(theta + 1e-12), axis=-1)
    # Compute maximum possible entropy of theta
    max_entropy = -jnp.log(theta.shape[-1])
    # Express entropy as a fraction of the maximum possible entropy
    entropy_param = jnp.sqrt(1.0 - entropy / max_entropy)
    # Pass through 32-dimensional fourier encoding
    fourier_base = jnp.pi * 2.0 ** jnp.arange(
        -1, (32 // 2) - 1
    )
    fourier_base = jnp.broadcast_to(fourier_base, theta.shape[:-1] + fourier_base.shape)
    fourier_embedding = jnp.concatenate(
        [
            jnp.sin(entropy_param[:, None] * fourier_base),
            jnp.cos(entropy_param[:, None] * fourier_base),
        ],
        axis=-1,
    )
    return  jnp.concatenate([theta, fourier_embedding], axis=-1)
    
def get_transformer_fn(
    output_dim: int,
    embed_dim: int = 1280,
    ffn_embed_dim: int = 5120,
    num_layers: int = 33,
    attention_heads: int = 20,
    ) -> Callable[[jax.Array], jax.Array]:
    """ Returns a callable Transformer model function.
    Args:
        output_dim (int): The output dimension e.g. number of classes
        embed_dim (int): The embedding dimension
        ffn_embed_dim (int): The embedding dimension of the feed-forward network
        num_layers (int): The number of transformer layers
        attention_heads (int): The number of attention heads
    Returns:
        Callable[[jax.Array], jax.Array]: The callable model function
    """

    def transformer_fn(theta: jax.Array) -> jax.Array:
        """
        """
        x = apply_entropy_encoding(theta)
        transformer = Transformer(
            output_dim=output_dim,
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            num_layers=num_layers,
            attention_heads=attention_heads,
            name = "transformer",
        )
        return transformer(x)

    return transformer_fn