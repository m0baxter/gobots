import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from ..losses import fast_cross_entropy_loss


class MTPTrainer(Trainer):
    def __init__(
        self,
        auxiliary_loss_weight: float = 1e-4,
        mtp_weight: float = 0.01,
        mtp_depth: int = 1,
        z_loss_weight: float = 0.001,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mtp_weight = mtp_weight
        self.z_loss_weight = z_loss_weight
        self.mtp_depth = mtp_depth
        self.auxiliary_loss_weight = auxiliary_loss_weight

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        _, s = inputs["input_ids"][:, : -(1 + self.mtp_depth)].shape

        outputs = model(
            input_ids=inputs["input_ids"][:, : -(1 + self.mtp_depth)],
            document_ids=inputs["document_ids"][:, : -(1 + self.mtp_depth)],
            input_pos=inputs["input_pos"][:, : -(1 + self.mtp_depth)],
        )
        # logits, mtp_logits, auxiliary_losses, z_losses = (
        logits, mtp_logits, auxiliary_losses, _ = (
            outputs["logits"],
            outputs.get("mtp_logits", []),
            outputs.get("auxiliary_losses", []),
            outputs.get("z_losses", []),
        )
        main_loss = fast_cross_entropy_loss(
            logits, inputs["input_ids"][:, 1 : s + 1].contiguous()
        )

        mtp_loss = torch.tensor(0.0).to(logits.device)
        z_loss = torch.tensor(0.0).to(logits.device)
        auxiliary_loss = torch.tensor(0.0).to(logits.device)

        for d, logits_d in enumerate(mtp_logits, start=1):
            mtp_loss += fast_cross_entropy_loss(
                logits_d, inputs["input_ids"][:, 1 + d : d + s + 1].contiguous()
            )

        if len(mtp_logits) > 0:
            mtp_loss *= self.mtp_weight / self.mtp_depth

        if len(auxiliary_losses) > 0:
            auxiliary_loss = (
                self.auxiliary_loss_weight * torch.cat(auxiliary_losses).sum()
            )

        # if len(z_losses) > 0:
        #    z_loss = self.z_loss_weight * torch.sum(torch.stack(z_losses))

        total_loss = main_loss + mtp_loss + auxiliary_loss  # + z_loss

        if return_outputs:
            output = {
                "total_loss": total_loss,
                "mtp_loss": mtp_loss,
                "auxiliary_loss": auxiliary_loss,
                "z_loss": z_loss,
                "logits": logits,
                "mtp_logits": mtp_logits,
            }

            return total_loss, output

        return total_loss

    def get_decay_parameter_names(self, model) -> list[str]:
        """
        Get all parameter names that weight decay will be applied to.

        This function filters out parameters in two ways:
        1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
        2. By parameter name patterns (containing 'bias', or variation of 'norm')
        """
        forbidden_name_patterns = [
            r"bias",
            r"layernorm",
            r"rmsnorm",
            r"(?:^|\.)norm(?:$|\.)",
            r"_norm(?:$|\.)",
            "embedding",
        ]

        if (
            hasattr(model.config, "tie_word_embeddings")
            and model.config.tie_word_embeddings
        ):
            forbidden_name_patterns = forbidden_name_patterns + ["lm_head"]

        decay_parameters = get_parameter_names(
            model, [nn.LayerNorm], forbidden_name_patterns
        )

        return decay_parameters
