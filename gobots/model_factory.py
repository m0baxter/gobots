from transformers import AutoTokenizer


def build_bagl_hybrid():
    from .models.bagl import (
        BaGLConfig,
        BaGLMTPConfig,
        BaGLWithMTPConfig,
        BaGLWithMTPModel,
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    config = BaGLWithMTPConfig(
        bagl_config=BaGLConfig(
            tie_word_embeddings=True,
            vocab_size=128256,
            bos_token_id=128000,
            eos_token_id=128001,
            hidden_size=768,
            dense_layers=[
                0,
                1,
                2,
                3,
                5,
                7,
                9,
                11,
                13,
                15,
                17,
            ],
            nope_layers=[
                6,
                10,
                14,
                18,
            ],
            intermediate_size=256,
            intermediate_size_mlp=3072,
            use_qk_norm=True,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=19,
            attention_bias=False,
            attention_dropout=0.0,
            shared_expert=True,
            n_routed_experts=32,
            num_experts_per_token=3,
            mlp_bias=False,
            rms_norm_eps=1e-05,
            # max_position_embeddings=131072,
            max_position_embeddings=2048 * 4,
            rope_base=500000.0,
            initializer_range=0.1,
        ),
        mtp_config=BaGLMTPConfig(
            tie_word_embeddings=True,
            vocab_size=128256,
            bos_token_id=128000,
            eos_token_id=128001,
            hidden_size=768,
            # max_position_embeddings=131072,
            max_position_embeddings=2048 * 4,
            rope_base=500000.0,
            initializer_range=0.1,
            attention_type="grouped_query_attention",
            input_dim=768,
            intermediary_dim=3072,
            bias=False,
            E_q=768,
            E_k=768,
            E_v=768,
            E_total=768,
            num_heads=32,
            num_kv_groups=8,
            qk_norm=True,
            dropout=0.0,
            attention_bias=False,
            rms_norm_eps=1e-05,
            feedforward_type="dense",
            intermediate_dim=3072,
            num_nextn_predict_layers=1,
        ),
        initializer_range=0.1,
        tie_word_embeddings=True,
    )
    model = BaGLWithMTPModel(config)

    return config, tokenizer, model


def build_bagl_moe():
    from .models.bagl import (
        BaGLConfig,
        BaGLMTPConfig,
        BaGLWithMTPConfig,
        BaGLWithMTPModel,
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    config = BaGLWithMTPConfig(
        bagl_config=BaGLConfig(
            tie_word_embeddings=True,
            vocab_size=128256,
            bos_token_id=128000,
            eos_token_id=128001,
            hidden_size=768,
            dense_layers=[
                0,
                1,
                2,
            ],
            nope_layers=[
                5,
                11,
                15,
                19,
            ],
            intermediate_size=256,
            intermediate_size_mlp=3072,
            use_qk_norm=True,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=20,
            attention_bias=False,
            attention_dropout=0.0,
            shared_experts=True,
            n_routed_experts=32,
            num_experts_per_token=3,
            mlp_bias=False,
            rms_norm_eps=1e-05,
            # max_position_embeddings=131072,
            max_position_embeddings=2048 * 4,
            rope_base=500000.0,
            initializer_range=0.1,
        ),
        mtp_config=BaGLMTPConfig(
            tie_word_embeddings=True,
            vocab_size=128256,
            bos_token_id=128000,
            eos_token_id=128001,
            hidden_size=768,
            # max_position_embeddings=131072,
            max_position_embeddings=2048 * 4,
            rope_base=500000.0,
            initializer_range=0.1,
            attention_type="grouped_query_attention",
            E_q=768,
            E_k=768,
            E_v=768,
            E_total=768,
            num_heads=32,
            num_kv_groups=8,
            qk_norm=True,
            dropout=0.0,
            attention_bias=False,
            rms_norm_eps=1e-05,
            feedforward_type="moe",
            shared_expert=True,
            n_routed_experts=32,
            intermediate_size=256,
            num_experts_per_token=3,
            num_nextn_predict_layers=1,
        ),
        initializer_range=0.1,
        tie_word_embeddings=True,
    )
    model = BaGLWithMTPModel(config)

    return config, tokenizer, model


def build_bagl_dense():
    from .models.bagl import (
        BaGLConfig,
        BaGLMTPConfig,
        BaGLWithMTPConfig,
        BaGLWithMTPModel,
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    config = BaGLWithMTPConfig(
        bagl_config=BaGLConfig(
            tie_word_embeddings=True,
            vocab_size=128256,
            bos_token_id=128000,
            eos_token_id=128001,
            hidden_size=1024,
            dense_layers=[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
            ],
            nope_layers=[
                3,
                7,
                11,
                15,
            ],
            intermediate_size_mlp=4096,
            use_qk_norm=True,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=16,
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
            rms_norm_eps=1e-05,
            max_position_embeddings=131072,
            rope_base=500000.0,
            initializer_range=0.1,
        ),
        mtp_config=BaGLMTPConfig(
            tie_word_embeddings=True,
            vocab_size=128256,
            bos_token_id=128000,
            eos_token_id=128001,
            hidden_size=1024,
            max_position_embeddings=131072,
            rope_base=500000.0,
            initializer_range=0.1,
            attention_type="grouped_query_attention",
            input_dim=1024,
            intermediary_dim=3072,
            bias=False,
            E_q=1024,
            E_k=1024,
            E_v=1024,
            E_total=1024,
            num_heads=32,
            num_kv_groups=8,
            qk_norm=True,
            dropout=0.0,
            attention_bias=False,
            rms_norm_eps=1e-05,
            feedforward_type="dense",
            intermediate_dim=4096,
            num_nextn_predict_layers=1,
        ),
        initializer_range=0.1,
        tie_word_embeddings=True,
    )
    model = BaGLWithMTPModel(config)

    return config, tokenizer, model


def build_llama_3p2_1b():
    from .models.llama3_clone import Llama3Config, Llama3Model

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    config = Llama3Config(
        vocab_size=128256,
        pad_token_id=128004,
        hidden_size=2048,
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
        initializer_range=0.013975424859373685,
    )
    model = Llama3Model(config)

    return config, tokenizer, model


def build_qwen3_dense_4b():
    from .models.qwen3_dense_clone import Qwen3DenseConfig, Qwen3DenseModel

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    config = Qwen3DenseConfig(
        vocab_size=151936,
        hidden_size=2560,
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
        initializer_range=0.02,
    )
    model = Qwen3DenseModel(config)

    return config, tokenizer, model


def build_smollm3_3b():
    from .models.smollm3_clone import SmolLM3Config, SmolLM3Model

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    config = SmolLM3Config(
        vocab_size=128256,
        pad_token_id=128004,
        hidden_size=2048,
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
        initializer_range=0.02,
    )
    model = SmolLM3Model(config)

    return config, tokenizer, model


def build_llama4_scout_17b_16e():
    from .models.llama4_clone import Llama4Config, Llama4Model

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E")
    config = Llama4Config(
        vocab_size=202048,
        hidden_size=5120,
        interleave_moe_layer_step=2,
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
        initializer_range=0.02,
    )
    model = Llama4Model(config)

    return config, tokenizer, model


def build_deepseek_v3():
    from .models.deepseek_v3_clone import DeepSeekV3Config, DeepSeekV3Model

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    config = DeepSeekV3Config(
        vocab_size=129280,
        pad_token_id=2,
        hidden_size=7168,
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
        initializer_range=0.02,
        mtp_config={
            "attention_type": "multi_head_latent_attention",
            "d_model": 7168,
            "hidden_size": 7168,
            "num_heads": 128,
            "v_head_dim": 128,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "dropout": 0.0,
            "attention_bias": False,
            "feedforward_type": "moe",
            "n_shared_experts": 1,
            "n_routed_experts": 256,
            "intermediate_size": 2048,
            "num_experts_per_token": 8,
        },
    )
    model = DeepSeekV3Model(config)

    return config, tokenizer, model
