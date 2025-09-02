from transformers import AutoTokenizer


def build_llama_3p2_1b():
    from .models.llama3_clone import Llama3Config, Llama3Model

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    config = Llama3Config(
        vocab_size=128256,
        hidden_dim=2048,
        intermediate_dim=8192,
        num_attention_heads=32,
        num_hidden_layers=16,
        num_key_value_heads=8,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        rms_norm_eps=1e-05,
        max_position_embeddings=131072,
        rope_base=500000.0,
    )
    model = Llama3Model(config)

    return config, tokenizer, model


def build_qwen3_dense_4b():
    from .models.qwen3_dense_clone import Qwen3DenseConfig, Qwen3DenseModel

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    config = Qwen3DenseConfig(
        vocab_size=151936,
        hidden_dim=2560,
        intermediate_dim=9728,
        num_attention_heads=32,
        num_hidden_layers=36,
        num_key_value_heads=8,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        rms_norm_eps=1e-06,
        max_position_embeddings=40960,
        rope_base=1000000,
    )
    model = Qwen3DenseModel(config)

    return config, tokenizer, model


def build_smollm3_3b():
    from .models.smollm3_clone import SmolLM3Config, SmolLM3Model

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    config = SmolLM3Config(
        vocab_size=128256,
        hidden_dim=2048,
        intermediate_dim=11008,
        num_attention_heads=16,
        num_hidden_layers=36,
        num_key_value_heads=4,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        rms_norm_eps=1e-06,
        no_rope_layer_interval=4,
        max_position_embeddings=65536,
        rope_base=5000000.0,
    )
    model = SmolLM3Model(config)

    return config, tokenizer, model


def build_llama4_scout_17b_16e():
    from .models.llama4_clone import Llama4Config, Llama4Model

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E")
    config = Llama4Config(
        vocab_size=202048,
        hidden_dim=5120,
        interleave_moe_layer_step=1,
        intermediate_dim=8192,
        intermediate_size_mlp=16384,
        num_experts_per_tok=1,
        num_local_experts=16,
        num_attention_heads=40,
        num_hidden_layers=48,
        num_key_value_heads=8,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        use_qk_norm=True,
        rms_norm_eps=1e-05,
        max_position_embeddings=262144,
        rope_base=500000.0,
    )
    model = Llama4Model(config)

    return config, tokenizer, model


def build_deepseek_v3():
    from .models.deepseek_v3_clone import DeepSeekV3Config, DeepSeekV3Model

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    config = DeepSeekV3Config(
        vocab_size=129280,
        pad_token_id=2,
        hidden_dim=7168,
        intermediate_dim=18432,
        moe_intermediate_size=2048,
        num_experts_per_tok=8,
        n_routed_experts=256,
        num_attention_heads=128,
        num_hidden_layers=61,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        rms_norm_eps=1e-06,
        max_position_embeddings=163840,
        rope_base=10000.0,
        n_shared_experts=1,
        first_k_dense_replace=3,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_nextn_predict_layers=1,
        mtp_config={
            "attention_type": "multi_head_latent_attention",
            "d_model": 7168,
            "num_heads": 128,
            "v_head_dim": 128,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 128,
            "qk_nope_head_dim": 64,
            "dropout": 0.0,
            "attention_bias": False,
            "feedforward_type": "moe",
            "n_shared_experts": 1,
            "n_routed_experts": 256,
            "intermediate_size": 2048,
            "num_experts_per_token": 1,
        },
    )
    model = DeepSeekV3Model(config)

    return config, tokenizer, model
