from transformers import PretrainedConfig
from typing import Any


class DeepSeekV3Config(PretrainedConfig):
    """
    Args:
       vocab_size (`int`, *optional*, defaults to 32000):
          Vocabulary size for the model.
        pad_token_id (`int | None` defaults to None):
          id of the padding token.
        hidden_size: (`int`, *optional* default to 2048)
          The embedding dimension for tokens.
       moe_intermediate_size (`int` *optional* defaults to 8192):
          dimension of the feedforward MOE layers
       intermediate_dim (`int` *optional* defaults to 16384):
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
       max_position_embeddings (`int` *optional* defaults to 4096):
          maximum number of positional embeddings supported.
       rope_base (`float` *optional* defaults to 500000.0):
          the base for the RoPE embedding.
       rms_norm_eps (`float` *optional* defaults to 1E-05):
          regularizer for rms norm layers
       num_experts_per_tok (`int` *optional* defaults to 1):
          number of experts choosen per token.
       n_routed_experts (`int` *optional` defaults to 16):
          total number of experts per MOE layer.
       n_shared_experts (`int` *optional* defaults to 1):
          number of shared experts in moe layers.
       first_k_dense_replace (`int` *optional* defaults to 3):
          number of dense layers to start the model.
       kv_lora_rank (`int`*optional* defaults to 512):
          latent dimension for the keys and values in multihead latent attention.
       q_lora_rank (`int` *optional* defaults to 1536):
          latent dimension for the queries in multihead latent attention.
       qk_nope_head_dim (`int` *optional* defaults to 128):
          dimension of the NoPE portion of vectors in multihead latent attention.
       qk_rope_head_dim (`int` *optional* defaults to 64):
          dimension of the RoPE portion of vectors in multihead latent attention.
       v_head_dim (`int` *optional* defaults to 128):
          number of key and value heads in multihead latent attention.
       mtp_config (`dict` *optional* defaults to None):
          parameters for the multitoken prediction heads.
       num_nextn_predict_layers (`int` *optional* defaults to 0):
          number of multi token prediction layers to add.
       tie_word_embeddings (`bool` *optional* defaults to False):
          whether to tie the weights of the embedding layer and the lm_head.
       initializer_range (`float` *optional* defaults to 0.02):
          value to use when initializing model weights.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        pad_token_id: int | None = 0,
        hidden_size: int = 2048,
        intermediate_dim: int = 8192,
        moe_intermediate_size: int = 16384,
        num_experts_per_tok: int = 1,
        n_routed_experts: int = 16,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 16,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        rms_norm_eps: float = 1e-05,
        max_position_embeddings: int = 4096,
        rope_base: float = 500000.0,
        n_shared_experts: int = 1,
        first_k_dense_replace: int = 3,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        num_nextn_predict_layers: int = 0,
        mtp_config: dict[str, Any] | None = None,
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
        self.moe_intermediate_size = moe_intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_base = rope_base
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.first_k_dense_replace = first_k_dense_replace
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_config = mtp_config
        self.initializer_range = initializer_range
