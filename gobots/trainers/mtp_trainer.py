import torch
import torch.nn as nn
from transformers import Trainer


class MTPTrainer(Trainer):
    def __init__(
        self,
        auxiliary_loss_weight: float = 1e-4,
        mtp_weight: float = 0.01,
        mtp_depth: int = 1,
        ignore_index: int = -100,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mtp_weight = mtp_weight
        self.mtp_depth = mtp_depth
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        torch.compiler.cudagraph_mark_step_begin()
        _, s = inputs["input_ids"][:, : -(1 + self.mtp_depth)].shape

        outputs = model(
            input_ids=inputs["input_ids"][:, : -(1 + self.mtp_depth)],
            document_ids=inputs["document_ids"][:, : -(1 + self.mtp_depth)],
            input_pos=inputs["input_pos"][:, : -(1 + self.mtp_depth)],
        )
        logits, mtp_logits, auxiliary_losses = (
            outputs["logits"],
            outputs.get("mtp_logits", []),
            outputs.get("auxiliary_losses", []),
        )
        main_loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            inputs["input_ids"][:, 1 : s + 1].contiguous().view(-1),
        )

        mtp_loss = 0

        for d, logits_d in enumerate(mtp_logits, start=1):
            mtp_loss += self.loss_fn(
                logits_d.view(-1, logits_d.size(-1)),
                inputs["input_ids"][:, 1 + d : d + s + 1].contiguous().view(-1),
            )

        auxiliary_loss = 0

        if len(auxiliary_losses) > 0:
            auxiliary_loss = (
                self.auxiliary_loss_weight * torch.cat(auxiliary_losses).sum()
            )

        mtp_loss *= self.mtp_weight / self.mtp_depth
        total_loss = main_loss + mtp_loss + auxiliary_loss

        if return_outputs:
            output = {
                "mtp_loss": mtp_loss,
                "auxiliary_loss": auxiliary_loss,
                "logits": logits,
                "mtp_logits": mtp_logits,
            }

            return total_loss, output

        return total_loss
