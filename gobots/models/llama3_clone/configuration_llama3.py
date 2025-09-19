from transformers import PretrainedConfig


class Llama3Config(PretrainedConfig):
    """
    Args:
       vocab_size (`int`, *optional*, defaults to 32000):
          Vocabulary size for the model.
        hidden_size: (`int`, *optional* default to 2048)
          The embedding dimension for tokens.
       intermediate_dim: (`int` *optional* defaults to 8192)
          dimension of the feedforward layers
       attention_bias (`bool` *optional* defaults to False):
          whether to include a bias term in the attention blocks.
       attention_dropout (`float` *optional* defaults to 0.0):
          dropout rate in the attention blocks.
       mlp_bias (`bool` *optional* defaults to False):
          whether to include bias in the feedforward networks.
       num_attention_heads (`int` *optional* defaults to 32):
          number of attention heads per query.
       num_hidden_layers (`int` *optional* ddefaults to 16):
          number of attention blocks.
       num_key_value_heads (`int` *optional* defaults to 8):
          number of key/value heads in the attention mechanism if num_key_value_heads < num_attention_heads uses grouped query attention.
       max_position_embeddings (`int` *optional* defaults to 4096):
          maximum number of positional embeddings supported.
       rope_base (`float` *optional* defaults to 500000.0):
          the base for the RoPE embedding.
       rms_norm_eps (`float` *optional* defaults to 1E-05):
          regularizer for rms norm layers
       pad_token_id (`int | None` *optional* defaults to None):
          id of the padding token.
       tie_word_embeddings (`bool` *optional* defaults to False):
          whether to tie the embedding and lm_head weights.
       initializer_range (`float` *optional* defaults to 0.02):
          value to use when initializing model weights.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        pad_token_id: int | None = None,
        hidden_size: int = 2048,
        intermediate_dim: int = 8192,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 16,
        num_key_value_heads: int = 8,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        rms_norm_eps: float = 1e-05,
        max_position_embeddings: int = 4096,
        rope_base: float = 500000.0,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_dim = intermediate_dim
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_base = rope_base
        self.initializer_range = initializer_range
