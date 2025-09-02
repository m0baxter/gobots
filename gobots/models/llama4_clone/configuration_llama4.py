from transformers import PretrainedConfig


class Llama4Config(PretrainedConfig):
    """
    Args:
       vocab_size (`int`, *optional*, defaults to 32000):
          Vocabulary size for the model.
        hidden_dim: (`int`, *optional* default to 2048)
          The embedding dimension for tokens.
       intermediate_dim (`int` *optional* defaults to 8192):
          dimension of the feedforward MOE layers
       intermediate_size_ml (`int` *optional* defaults to 16384):
          dimension of the dense feedforward layers.
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
       interleave_moe_layer_step (`int` *optional* defaults to 1):
          rate at which dense and MOE layers alternate.
       use_qk_norm (`bool` *optional* defaults to True):
          whether to apply query/key normalizzation in the attention block.
       num_experts_per_tok (`int` *optional* defaults to 1):
          number of experts choosen per token.
       num_local_experts (`int` *optional` defaults to 16):
          total number of experts per MOE layer.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 2048,
        interleave_moe_layer_step: int = 1,
        intermediate_dim: int = 8192,
        intermediate_size_mlp: int = 16384,
        num_experts_per_tok: int = 1,
        num_local_experts: int = 16,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 16,
        num_key_value_heads: int = 8,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        use_qk_norm: bool = True,
        rms_norm_eps: float = 1e-05,
        max_position_embeddings: int = 4096,
        rope_base: float = 500000.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.intermediate_size_mlp = intermediate_size_mlp
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_base = rope_base
        self.interleave_moe_layer_step = interleave_moe_layer_step
        self.use_qk_norm = use_qk_norm

        super().__init__(**kwargs)
