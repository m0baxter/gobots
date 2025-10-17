import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import GradientCheckpointingLayer, PreTrainedModel, GenerationMixin
from transformers.utils import ModelOutput
from torchtune.modules import RotaryPositionalEmbeddings
from ..attention_mechanisms import GroupedQueryAttention
from ..feedforward_layers import SwiGLUFeedForward
from ..mask_utils import generate_block_mask
from ..mixture_of_experts import MixtureOfExperts
from ..multitoken_prediction_layer import MultiTokenPredictionHead
from .configuration_bagl import BaGLConfig, BaGLMTPConfig, BaGLWithMTPConfig


@dataclass
class BaGLOutput(ModelOutput):
    logits: torch.FloatTensor | None = None
    hidden_states: torch.FloatTensor | None = None
    auxiliary_losses: list[torch.FloatTensor] = None
    z_losses: list[torch.FloatTensor] | None = None


@dataclass
class BaGLMTPOutput(ModelOutput):
    mtp_logits: torch.FloatTensor | None = None
    auxiliary_losses: list[torch.FloatTensor] = None
    z_losses: list[torch.FloatTensor] | None = None


@dataclass
class BaGLWithMTPOutput(ModelOutput):
    logits: torch.FloatTensor | None = None
    mtp_logits: torch.FloatTensor | None = None
    auxiliary_losses: list[torch.FloatTensor] = None
    z_losses: list[torch.FloatTensor] | None = None


class BaGLBlock(GradientCheckpointingLayer):
    def __init__(self, config: BaGLConfig, is_dense: bool):
        super().__init__()
        self.is_dense = is_dense
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(
            E_q=config.hidden_size,
            E_k=config.hidden_size,
            E_v=config.hidden_size,
            E_total=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_groups=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            dropout=config.attention_dropout,
            attention_bias=config.attention_bias,
            qk_norm=config.use_qk_norm,
        )
        self.mid_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.is_dense:
            self.feedforward = SwiGLUFeedForward(
                input_dim=config.hidden_size,
                intermediary_dim=config.intermediate_size_mlp,
                bias=config.mlp_bias,
            )

        else:
            self.feedforward = MixtureOfExperts(
                n_shared_experts=config.n_shared_experts,
                n_routed_experts=config.n_routed_experts,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts_per_token=config.num_experts_per_tok,
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
        auxiliary_losses = None
        z_loss = None

        if self.is_dense:
            x = self.feedforward(x)

        else:
            x, auxiliary_losses, z_loss = self.feedforward(x)

        x = skip + x

        return x, auxiliary_losses, z_loss


class BaGLModel(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["embedding_layer.weight", "lm_head.weight"]
    supports_gradient_checkpointing = True
    config_class = BaGLConfig

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            w_fan_in = module.weight.shape[-1]
            std = torch.math.sqrt(self.config.initializer_range / w_fan_in)

            module.weight.data.normal_(mean=0.0, std=std)
            module.weight.data.clamp_(min=-2 * std, max=2 * std)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            w_fan_in = module.weight.shape[-1]
            std = torch.math.sqrt(self.config.initializer_range / w_fan_in)
            module.weight.data.normal_(mean=0.0, std=std)

            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def __init__(self, config: BaGLConfig):
        super().__init__(config)

        self.config = config
        self.hidden_size = config.hidden_size
        self.nope_layers = set(config.nope_layers)
        self.dense_layers = set(config.dense_layers)
        self.pad_token_id = config.pad_token_id

        self.embedding_layer = nn.Embedding(
            config.vocab_size, config.hidden_size, self.pad_token_id
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BaGLBlock(config, is_dense=idx in self.dense_layers)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.rope = RotaryPositionalEmbeddings(
            dim=config.hidden_size // config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_base,
        )

        self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self._tie_or_clone_weights(self.embedding_layer, self.lm_head)

        self.post_init()

    def forward(
        self,
        input_ids,
        mask=None,
        document_ids=None,
        input_pos=None,
        labels=None,
    ):
        b, s = input_ids.shape
        x = self.embed_norm(self.embedding_layer(input_ids))
        auxiliary_losses = []
        z_losses = []

        if mask is None:
            mask = generate_block_mask(
                batch_size=b,
                query_length=s,
                key_value_length=s,
                document_ids=document_ids,
            )

        for i, block in enumerate(self.transformer_blocks):
            pos_embed = None if i in self.nope_layers else self.rope
            x, aux_loss, z_loss = block(x, mask, pos_embed, input_pos)

            if aux_loss is not None:
                auxiliary_losses.append(aux_loss)

            if z_loss is not None:
                z_losses.append(z_loss)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return BaGLOutput(
            logits=logits,
            hidden_states=x,
            auxiliary_losses=auxiliary_losses,
            z_losses=z_losses,
        )


class BaGLMTPModel(PreTrainedModel):
    _tied_weights_keys = ["embedding_layer.weight", "lm_head.weight"]
    supports_gradient_checkpointing = True
    config_class = BaGLMTPConfig

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            w_fan_in = module.weight.shape[-1]
            std = torch.math.sqrt(self.config.initializer_range / w_fan_in)

            module.weight.data.normal_(mean=0.0, std=std)
            module.weight.data.clamp_(min=-2 * std, max=2 * std)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            w_fan_in = module.weight.shape[-1]
            std = torch.math.sqrt(self.config.initializer_range / w_fan_in)
            module.weight.data.normal_(mean=0.0, std=std)

            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def __init__(self, config: BaGLMTPConfig):
        super().__init__(config)

        self.config = config
        self.hidden_size = config.hidden_size
        self.pad_token_id = config.pad_token_id
        self.num_nextn_predict_layers = config.num_nextn_predict_layers

        self.embedding_layer = nn.Embedding(
            config.vocab_size, config.hidden_size, self.pad_token_id
        )

        self.rope = RotaryPositionalEmbeddings(
            dim=config.hidden_size // config.num_heads,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_base,
        )

        self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self._tie_or_clone_weights(self.embedding_layer, self.lm_head)

        self.mtp_heads = nn.ModuleList(
            [
                MultiTokenPredictionHead(config)
                for _ in range(config.num_nextn_predict_layers)
            ]
        )

        self.post_init()

    def forward(
        self,
        hidden_states,
        logits,
        input_ids,
        document_ids=None,
        input_pos=None,
        labels=None,
        mtp_sampling=False,
        mtp_temperature=1.0,
    ):
        b, s, d = hidden_states.shape
        mtp_logits = []
        auxiliary_losses = []
        z_losses = []

        current_input_ids = input_ids
        current_hidden = hidden_states
        current_logits = logits
        shifted_document_ids = document_ids
        shifted_input_pos = input_pos

        for mtp_head in self.mtp_heads:
            # shift document_ids to create new mask
            if shifted_document_ids is not None:
                shifted_document_ids = torch.nn.functional.pad(
                    shifted_document_ids, (0, 1), mode="replicate"
                )[:, 1:]

                mask = generate_block_mask(
                    batch_size=b,
                    query_length=s,
                    key_value_length=s,
                    document_ids=shifted_document_ids,
                )
                mask = torch._dynamo.mark_dynamic(mask, index=0)

            # shift the input positional ids:
            if shifted_input_pos is not None:
                shifted_input_pos = torch.nn.functional.pad(
                    shifted_input_pos, (0, 1), mode="replicate"
                )
                shifted_input_pos[:, -1] += 1
                shifted_input_pos = shifted_document_ids[:, 1:]

            # determine next token:
            if mtp_sampling:
                probs = F.softmax(current_logits[:, -1, :] / mtp_temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            else:
                next_token = current_logits[:, -1, :].argmax(dim=-1)

            # add to sequence and drop first:
            current_input_ids = torch.cat(
                [current_input_ids, next_token.view(b, 1)], dim=-1
            )[:, 1:]

            # apply mpt head
            embeds = self.embed_norm(self.embedding_layer(current_input_ids))
            current_hidden, aux_loss, z_loss = mtp_head(
                current_hidden,
                embeds,
                mask,
                self.rope,
                shifted_document_ids,
            )

            if aux_loss is not None:
                auxiliary_losses.append(aux_loss)

            if z_loss is not None:
                z_losses.append(z_loss)

            # add new logit to output
            current_logits = self.lm_head(self.final_norm(current_hidden))
            mtp_logits.append(current_logits)

        return BaGLMTPOutput(
            mtp_logits=mtp_logits, auxiliary_losses=auxiliary_losses, z_losses=z_losses
        )


class BaGLWithMTPModel(PreTrainedModel):
    _tied_weights_keys = [
        "bagl_model.embedding_layer.weight",
        "bagl_model.lm_head.weight",
        "mtp_model.embedding_layer.weight",
        "mtp_model.lm_head.weight",
    ]
    supports_gradient_checkpointing = True
    config_class = BaGLMTPConfig

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            w_fan_in = module.weight.shape[-1]
            std = torch.math.sqrt(self.config.initializer_range / w_fan_in)

            module.weight.data.normal_(mean=0.0, std=std)
            module.weight.data.clamp_(min=-2 * std, max=2 * std)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            w_fan_in = module.weight.shape[-1]
            std = torch.math.sqrt(self.config.initializer_range / w_fan_in)
            module.weight.data.normal_(mean=0.0, std=std)

            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def __init__(self, config: BaGLWithMTPConfig):
        super().__init__(config)
        self.config = config
        self.bagl_model = BaGLModel(config.bagl_config)
        self.mtp_model = BaGLMTPModel(config.mtp_config)

        if config.tie_word_embeddings:
            self._tie_or_clone_weights(
                self.bagl_model.embedding_layer, self.bagl_model.lm_head
            )
            self._tie_or_clone_weights(
                self.mtp_model.embedding_layer, self.mtp_model.lm_head
            )
            self._tie_or_clone_weights(
                self.mtp_model.embedding_layer, self.bagl_model.embedding_layer
            )
            self._tie_or_clone_weights(self.mtp_model.lm_head, self.bagl_model.lm_head)

        self.post_init

    def forward(
        self,
        input_ids,
        mask=None,
        document_ids=None,
        input_pos=None,
        labels=None,
        mtp_sampling=False,
        mtp_temperature=1.0,
    ):
        bagl_outputs = self.bagl_model(
            input_ids=input_ids,
            mask=mask,
            document_ids=document_ids,
            input_pos=input_pos,
            labels=labels,
        )

        mtp_outputs = self.mtp_model(
            hidden_states=bagl_outputs.hidden_states,
            logits=bagl_outputs.logits,
            input_ids=input_ids,
            document_ids=document_ids,
            input_pos=input_pos,
            labels=labels,
            mtp_sampling=mtp_sampling,
            mtp_temperature=mtp_temperature,
        )

        return BaGLWithMTPOutput(
            logits=bagl_outputs.logits,
            mtp_logits=mtp_outputs.mtp_logits,
            auxiliary_losses=bagl_outputs.auxiliary_losses
            + mtp_outputs.auxiliary_losses,
            z_losses=bagl_outputs.z_losses + mtp_outputs.z_losses,
        )
