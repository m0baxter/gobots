import argparse
import torch
from multiprocessing import cpu_count
from pathlib import Path
from transformers import TrainingArguments
from gobots.callbacks import ZClipCallback
from gobots.model_factory import build_bagl_hybrid
from gobots.trainers import MTPTrainer, LossAccumulator
from gobots.utils.data_utils import prepare_pretraining_datasets
from gobots.utils.parse_config import load_config_file

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cudnn.rnn.fp32_precision = "tf32"
torch.set_float32_matmul_precision("high")


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", choices=["train", "eval"])

    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    training_config = load_config_file(args.training_config)
    num_shards = int(cpu_count() * 0.75)
    loss_accumlator = LossAccumulator()

    def compute_metrics(pred, compute_result: bool = False):
        total_loss, mtp_loss, auxiliary_loss, z_loss, logits, mtp_logits = (
            pred.predictions
        )
        loss_accumlator.update(
            total_loss,
            mtp_loss,
            auxiliary_loss,
            z_loss,
            logits.shape[0] * logits.shape[1],
        )

        if compute_result:
            total_main_loss, total_mtp_loss, total_auxiliary_loss, total_z_loss = (
                loss_accumlator.compute()
            )

            return {
                "main_loss": total_main_loss,
                "mtp_loss": total_mtp_loss,
                "auxiliary_loss": total_auxiliary_loss,
                "z_loss": total_z_loss,
                "perplexity": torch.math.exp(total_main_loss),
            }

        return

    config, tokenizer, model = build_bagl_hybrid()

    train_dataset = prepare_pretraining_datasets(
        tokenizer,
        context_length=training_config.context_length
        + 1
        + config.mtp_config.num_nextn_predict_layers,
        num_shards=num_shards,
    )

    training_args = TrainingArguments(
        disable_tqdm=True,
        max_steps=training_config.max_steps,
        warmup_steps=training_config.warmup_steps,
        lr_scheduler_type=training_config.lr_scheduler_type,
        lr_scheduler_kwargs=training_config.lr_scheduler_kwargs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        eval_strategy="no",
        save_strategy=training_config.save_strategy,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        logging_strategy=training_config.logging_strategy,
        logging_steps=training_config.logging_steps,
        output_dir=training_config.output_dir,
        log_level=training_config.log_level,
        bf16=training_config.use_bf16,
        tf32=training_config.use_tf32,
        optim=training_config.optim,
        learning_rate=training_config.learning_rate,
        dataloader_num_workers=num_shards,
        dataloader_prefetch_factor=8,
        include_num_input_tokens_seen=True,
        max_grad_norm=training_config.max_grad_norm,
        weight_decay=training_config.weight_decay,
        run_name=training_config.run_name,
        report_to="trackio",
        trackio_space_id=None,
        dataloader_drop_last=True,
    )
    trainer = MTPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        auxiliary_loss_weight=training_config.auxiliary_loss_weight,
        z_loss_weight=training_config.z_loss_weight,
        mtp_weight=training_config.mtp_weight,
        mtp_depth=config.mtp_config.num_nextn_predict_layers,
        compute_metrics=compute_metrics,
        callbacks=[
            ZClipCallback(
                mode=training_config.zclip_config,
                alpha=training_config.zclip_config,
                clip_option=training_config.zclip_config,
                z_thresh=training_config.zclip_config,
                clip_factor=training_config.zclip_config,
                max_grad_norm=training_config.zclip_config,
                warmup_steps=training_config.zclip_config,
            )
        ],
    )

    if any(Path(training_config.output_dir).iterdir()):
        trainer.train(resume_from_checkpoint=True)

    else:
        trainer.train()

    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    tokenizer.save_pretrained(training_config.output_dir)
    trainer.save_model(training_config.output_dir)

    trainer.accelerator.end_training()
