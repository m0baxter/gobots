import yaml
from enum import Enum
from pydantic import BaseModel
from typing import Any


class ModelType(str, Enum):
    DENSE = "dense"
    MOE = "moe"
    HYBRID = "hybrid"


class LRScheduleType(str, Enum):
    linear = "linear"
    cosine = "cosine"
    cosine_with_restarts = "cosine_with_restarts"
    polynomial = "polynomial"
    constant = "constant"
    constant_with_warmup = "constant_with_warmup"
    inverse_sqrt = "inverse_sqrt"
    reduce_lr_on_plateau = "reduce_lr_on_plateau"
    cosine_with_min_lr = "cosine_with_min_lr"
    cosine_warmup_with_min_lr = "cosine_warmup_with_min_lr"
    warmup_stable_decay = "warmup_stable_decay"


class LogLevel(str, Enum):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical"
    passive = "passive"


class LoggingStrategy(str, Enum):
    no = "no"
    epoch = "epoch"
    steps = "steps"


class SaveStrategy(str, Enum):
    no = "no"
    epoch = "epoch"
    steps = "steps"
    best = "best"


class Optimizer(str, Enum):
    adamw_torch = "adamw_torch"
    adamw_torch_fused = "adamw_torch_fused"
    adamw_torch_xla = "adamw_torch_xla"
    adamw_torch_npu_fused = "adamw_torch_npu_fused"
    adamw_apex_fused = "adamw_apex_fused"
    adafactor = "adafactor"
    adamw_anyprecision = "adamw_anyprecision"
    adamw_torch_4bit = "adamw_torch_4bit"
    adamw_torch_8bit = "adamw_torch_8bit"
    ademamix = "ademamix"
    sgd = "sgd"
    adagrad = "adagrad"
    adamw_bnb_8bit = "adamw_bnb_8bit"
    adamw_8bit = "adamw_8bit"
    ademamix_8bit = "ademamix_8bit"
    lion_8bit = "lion_8bit"
    lion_32bit = "lion_32bit"
    paged_adamw_32bit = "paged_adamw_32bit"
    paged_adamw_8bit = "paged_adamw_8bit"
    paged_ademamix_32bit = "paged_ademamix_32bit"
    paged_ademamix_8bit = "paged_ademamix_8bit"
    paged_lion_32bit = "paged_lion_32bit"
    paged_lion_8bit = "paged_lion_8bit"
    rmsprop = "rmsprop"
    rmsprop_bnb = "rmsprop_bnb"
    rmsprop_bnb_8bit = "rmsprop_bnb_8bit"
    rmsprop_bnb_32bit = "rmsprop_bnb_32bit"
    galore_adamw = "galore_adamw"
    galore_adamw_8bit = "galore_adamw_8bit"
    galore_adafactor = "galore_adafactor"
    galore_adamw_layerwise = "galore_adamw_layerwise"
    galore_adamw_8bit_layerwise = "galore_adamw_8bit_layerwise"
    galore_adafactor_layerwise = "galore_adafactor_layerwise"
    lomo = "lomo"
    adalomo = "adalomo"
    grokadamw = "grokadamw"
    schedule_free_radam = "schedule_free_radam"
    schedule_free_adamw = "schedule_free_adamw"
    schedule_free_sgd = "schedule_free_sgd"
    apollo_adamw = "apollo_adamw"
    apollo_adamw_layerwise = "apollo_adamw_layerwise"
    stable_adamw = "stable_adamw"


class ZClipMode(str, Enum):
    zscore = "zscore"
    percentile = "percentile"


class ClipOption(str, Enum):
    adaptive_scaling = "adaptive_scaling"
    mean = "mean"


class ZClipConfig(BaseModel):
    mode: ZClipMode = ZClipMode.zscore
    alpha: float = 0.97
    clip_option: ClipOption = ClipOption.adaptive_scaling
    z_thresh: float = 2.5
    clip_factor: float = 1.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 25


class TrainingConfig(BaseModel):
    run_name: str
    model_type: ModelType
    max_steps: int
    warmup_steps: int
    lr_scheduler_type: LRScheduleType
    lr_scheduler_kwargs: dict[str, Any]
    use_bf16: bool = True
    use_tf32: bool = True
    log_level: LogLevel
    logging_steps: int
    output_dir: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    save_strategy: SaveStrategy
    save_steps: int
    save_total_limit: int
    logging_strategy: LoggingStrategy
    optim: Optimizer
    learning_rate: float
    max_grad_norm: float
    weight_decay: float
    auxiliary_loss_weight: float
    z_loss_weight: float
    mtp_weight: float
    zclip_config: ZClipConfig
    context_length: int = 2048


def load_config_file(path):
    with open(path, "r") as stream:
        config_dict = yaml.safe_load(stream)

    return TrainingConfig.validate(config_dict["training"])
