import torch.nn as nn
from transformers import GradientCheckpointingLayer, PreTrainedModel
from torchtune.modules import RotaryPositionalEmbeddings
from ..attention_mechanisms import GroupedQueryAttention
from ..feedforward_layers import SwiGLUFeedForward
from ..mask_utils import generate_block_mask
from .configuration_qwen3_dense import Qwen3DenseConfig


class QwenBlock(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3DenseConfig):
        super().__init__()
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(
            E_q=config.hidden_size,
            E_k=config.hidden_size,
            E_v=config.hidden_size,
            E_total=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_groups=config.num_key_value_heads,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            dropout=config.attention_dropout,
            attention_bias=config.attention_bias,
        )
        self.mid_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feedforward = SwiGLUFeedForward(
            input_dim=config.hidden_size,
            intermediary_dim=config.intermediate_dim,
            bias=config.mlp_bias,
        )

    def forward(self, x, mask=None, pos_embedding=None, input_pos=None):
        skip = x
        x = self.input_norm(x)
        attention_score = self.attention(
            x,
            x,
            x,
            mask,
            pos_embedding,
            input_pos,
        )

        x = skip + attention_score
        skip = x

        x = self.mid_norm(x)
        x = self.feedforward(x)

        x = skip + x

        return x


class Qwen3DenseModel(PreTrainedModel):
    _tied_weights_keys = ["embedding_layer.weight", "lm_head.weight"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def __init__(self, config: Qwen3DenseConfig):
        super().__init__(config)
        self.config = config
        self.pad_token_id = config.pad_token_id

        self.embedding_layer = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        self.transformer_blocks = nn.ModuleList(
            [QwenBlock(config) for _ in range(config.num_hidden_layers)]
        )

        self.rope = RotaryPositionalEmbeddings(
            dim=config.hidden_size // config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_base,
        )

        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self._tie_or_clone_weights(self.embedding_layer, self.lm_head)

        self.post_init()

    def forward(self, x, mask=None, document_ids=None, input_pos=None):
        b, s = x.shape
        x = self.embedding_layer(x)

        if mask is None:
            mask = generate_block_mask(
                batch_size=b,
                query_length=s,
                key_value_length=s,
                document_ids=document_ids,
            )

        for block in self.transformer_blocks:
            x = block(x, mask, self.rope, input_pos)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return {"logits": logits}
