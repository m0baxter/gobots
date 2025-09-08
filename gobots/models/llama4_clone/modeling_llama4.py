import torch.nn as nn
from transformers import GradientCheckpointingLayer, PreTrainedModel
from torchtune.modules import RotaryPositionalEmbeddings
from ..attention_mechanisms import GroupedQueryAttention
from ..feedforward_layers import SwiGLUFeedForward
from ..mixture_of_experts import MixtureOfExperts
from .configuration_llama4 import Llama4Config


class Llama4Block(GradientCheckpointingLayer):
    def __init__(self, config: Llama4Config, index: int):
        super().__init__()
        self.is_dense = index % config.interleave_moe_layer_step == 0
        self.input_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(
            E_q=config.hidden_dim,
            E_k=config.hidden_dim,
            E_v=config.hidden_dim,
            E_total=config.hidden_dim,
            num_heads=config.num_attention_heads,
            num_kv_groups=config.num_key_value_heads,
            qk_norm=config.use_qk_norm,
            rms_norm_eps=config.rms_norm_eps,
            dropout=config.attention_dropout,
            attention_bias=config.attention_bias,
        )
        self.mid_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        if self.is_dense:
            self.feedforward = SwiGLUFeedForward(
                input_dim=config.hidden_dim,
                intermediary_dim=config.intermediate_size_mlp,
                bias=config.mlp_bias,
            )

        else:
            self.feedforward = MixtureOfExperts(
                n_shared_experts=1,
                n_routed_experts=config.num_local_experts,
                hidden_size=config.hidden_dim,
                intermediate_size=config.intermediate_size,
                num_experts_per_token=config.num_experts_per_tok,
            )

    def forward(self, x, mask=None, pos_embedding=None):
        skip = x
        x = self.input_norm(x)
        attention_score = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=mask,
            pos_embedding=pos_embedding,
            is_causal=mask is None,
        )

        x = skip + attention_score
        skip = x

        x = self.mid_norm(x)
        auxiliary_losses = None

        if self.is_dense:
            x = self.feedforward(x)

        else:
            x, auxiliary_losses = self.feedforward(x)

        x = skip + x

        return x, auxiliary_losses


class Llama4Model(PreTrainedModel):
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

        elif isinstance(module, MixtureOfExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.gate_down_proj.data.normal_(mean=0.0, std=std)

    def __init__(self, config: Llama4Config):
        super().__init__(config)
        self.config = config
        self.pad_token_id = config.pad_token_id

        self.embedding_layer = nn.Embedding(
            config.vocab_size, config.hidden_dim, config.pad_token_id
        )

        self.transformer_blocks = nn.ModuleList(
            [Llama4Block(config, index) for index in range(config.num_hidden_layers)]
        )
        self.rope = RotaryPositionalEmbeddings(
            dim=config.hidden_dim // config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_base,
        )

        self.final_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self._tie_or_clone_weights(self.embedding_layer, self.lm_head)

        self.post_init()

    def forward(self, x, mask=None):
        x = self.embedding_layer(x)
        auxiliary_losses = []

        for block in self.transformer_blocks:
            x, aux_loss = block(x, mask, self.rope)

            if aux_loss is not None:
                auxiliary_losses.append(aux_loss)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return {"logits": logits, "auxiliary_losses": auxiliary_losses}
