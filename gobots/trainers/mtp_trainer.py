import torch.nn as nn
from transformers import Trainer


class MTPTrainer(Trainer):
    def __init__(
        self,
        mtp_weight: float = 0.01,
        mtp_depth: int = 1,
        ignore_index: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mtp_weight = mtp_weight
        self.mtp_depth = mtp_depth
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=1)

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        _, s = inputs["input_ids"][:, : -(1 + self.mtp_depth)].shape

        logits, mtp_logits = model(x=inputs["input_ids"][:, : -(1 + self.mtp_depth)])
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

        mtp_loss *= self.mtp_weight / self.mtp_depth
        total_loss = main_loss + mtp_loss

        return (total_loss, (logits, mtp_logits)) if return_outputs else total_loss
