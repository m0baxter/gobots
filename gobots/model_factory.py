from transformers import AutoTokenizer


def build_llama_3p2_1b():
    from gobots.models.llama3_clone import Llama3Config, Llama3Model

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
    from gobots.models.qwen3_dense_clone import Qwen3DenseConfig, Qwen3DenseModel

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
    from gobots.models.smollm3_clone import SmolLM3Config, SmolLM3Model

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
