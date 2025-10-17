from transformers import PretrainedConfig


class BaGLMTPConfig(PretrainedConfig):
    model_type = "bagl_mpt"

    def __init__(
        self,
        tie_word_embeddings: bool = False,
        vocab_size: int = 32000,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        attention_type: str = "grouped_query_attention",
        feedforward_type: str = "dense",
        E_k: int | None = None,
        E_q: int | None = None,
        E_total: int | None = None,
        E_v: int | None = None,
        attention_bias: bool = False,
        bias: bool | None = None,
        d_model: int | None = None,
        dropout: float = 0.0,
        hidden_size: int | None = None,
        input_dim: int | None = None,
        intermediary_dim: int | None = None,
        intermediate_size: int | None = None,
        kv_lora_rank: int | None = None,
        n_routed_experts: int | None = None,
        n_shared_experts: int | None = None,
        num_experts_per_token: int | None = None,
        num_heads: int | None = None,
        num_kv_groups: int | None = None,
        q_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_norm: bool | None = None,
        qk_rope_head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        v_head_dim: int | None = None,
        num_nextn_predict_layers: int = 1,
        initializer_range: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.attention_type = attention_type
        self.feedforward_type = feedforward_type
        self.E_k = E_k
        self.E_q = E_q
        self.E_total = E_total
        self.E_v = E_v
        self.attention_bias = attention_bias
        self.bias = bias
        self.d_model = d_model
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.intermediary_dim = intermediary_dim
        self.intermediate_size = intermediate_size
        self.kv_lora_rank = kv_lora_rank
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_token = num_experts_per_token
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_norm = qk_norm
        self.qk_rope_head_dim = qk_rope_head_dim
        self.rms_norm_eps = rms_norm_eps
        self.v_head_dim = v_head_dim
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.initializer_range = initializer_range


class BaGLConfig(PretrainedConfig):
    """
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
           Vocabulary size for the model.
        pad_token_id (`int | None` defaults to None):
           id of the padding token.
        bos_token_id (`int` | `None` default to `None`):
           id of the beginning of stream token.
        eos_token_id (`int` | `None` default to `None`):
           id of the end of stream token.
        hidden_size: (`int`, *optional* default to 2048)
           The embedding dimension for tokens.
        intermediate_size (`int` *optional* defaults to 8192):
           dimension of the feedforward MOE layers
        attention_bias (`bool` *optional* defaults to False):
           whether to include a bias term in the attention blocks.
        attention_dropout (`float` *optional* defaults to 0.0):
          dropout rate in the attention blocks.
        mlp_bias (`bool` *optional* defaults to False):
           whether to include bias in the feedforward networks.
        num_attention_heads (`int` *optional* defaults to 32):
           number of attention heads per query.
        num_key_value_heads (`int` *optional* defaults to 8):
           number of key/value heads in the attention mechanism if num_key_value_heads < num_attention_heads uses grouped query attention.
        use_qk_norm (`bool` *optional* defaults to `True`):
           whether to normalize the query and key in the attention mechanism.
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
        mtp_config (`dict` *optional* defaults to None):
           parameters for the multitoken prediction heads.
        tie_word_embeddings (`bool` *optional* defaults to False):
           whether to tie the weights of the embedding layer and the lm_head.
        initializer_range (`float` *optional* defaults to 0.02):
           value to use when initializing model weights.
    """

    model_type = "bagl"

    def __init__(
        self,
        vocab_size: int = 32000,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        nope_layers: list[int] = [],
        dense_layers: list[int] = [],
        hidden_size: int = 2048,
        intermediate_size: int = 16384,
        num_experts_per_tok: int = 1,
        n_routed_experts: int = 16,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        use_qk_norm: bool = True,
        num_hidden_layers: int = 16,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        rms_norm_eps: float = 1e-05,
        max_position_embeddings: int = 4096,
        rope_base: float = 500000.0,
        n_shared_experts: int = 1,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.nope_layers = nope_layers
        self.dense_layers = dense_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.use_qk_norm = use_qk_norm
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
        self.initializer_range = initializer_range


class BaGLWithMTPConfig(PretrainedConfig):
    model_type = "bagl_with_mpt"
    sub_configs = {"bagl_config": BaGLConfig, "mtp_config": BaGLMTPConfig}

    def __init__(
        self,
        bagl_config: BaGLConfig = BaGLConfig(),
        mtp_config: BaGLMTPConfig = BaGLMTPConfig(),
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.2,
        **kwargs,
    ):
        self.bagl_config = bagl_config
        self.mtp_config = mtp_config
        self.initializer_range = initializer_range
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
