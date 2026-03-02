import torch.nn as nn
from transformers import TrainerCallback, TrainerState, TrainerControl, PreTrainedModel
from transformers.training_args import TrainingArguments
from .zclip import ZClip


class ZClipCallback(TrainerCallback):
    """
    Huggingface callback for ZClip.
    Applies adaptive gradient clipping before optimizer step.
    """

    def __init__(self, **zclip_kwargs):
        super().__init__()
        self.zclip = ZClip(**zclip_kwargs)

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel | nn.Module,
        **kwargs,
    ):
        self.zclip.step(model)
