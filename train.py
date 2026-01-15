import torch
from multiprocessing import cpu_count
from pathlib import Path
from transformers import TrainingArguments
from gobots.callbacks import ZClipCallback
from gobots.model_factory import build_bagl_hybrid
from gobots.trainers import MTPTrainer, LossAccumulator
from gobots.utils.data_utils import prepare_pretraining_datasets

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cudnn.rnn.fp32_precision = "tf32"
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
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
        context_length=2048 + 1 + config.mtp_config.num_nextn_predict_layers,
        num_shards=num_shards,
    )

    training_args = TrainingArguments(
        disable_tqdm=True,
        max_steps=92000,
        warmup_steps=2000,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs={
            "num_decay_steps": 9200,
            "decay_type": "linear",
        },
        per_device_train_batch_size=10,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=5,
        logging_strategy="steps",
        logging_steps=1,
        output_dir="./output_dir",
        log_level="info",
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        learning_rate=1.5e-03,
        dataloader_num_workers=num_shards,
        dataloader_prefetch_factor=8,
        batch_eval_metrics=True,
        include_num_input_tokens_seen=True,
        max_grad_norm=10.0,
        weight_decay=0.01,
        run_name="bagl_hybrid_08",
        report_to="trackio",
        trackio_space_id=None,
        dataloader_drop_last=True,
    )
    trainer = MTPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        auxiliary_loss_weight=1e-4,
        z_loss_weight=0.001,
        mtp_weight=0.01,
        mtp_depth=config.mtp_config.num_nextn_predict_layers,
        compute_metrics=compute_metrics,
        callbacks=[
            ZClipCallback(
                mode="zscore",
                alpha=0.97,
                clip_option="adaptive_scaling",
                z_thresh=2.5,
                clip_factor=0.95,
                max_grad_norm=1.0,
                warmup_steps=25,
            )
        ],
    )

    if any(Path("./output_dir").iterdir()):
        trainer.train(resume_from_checkpoint=True)

    else:
        trainer.train()

    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    tokenizer.save_pretrained("./output_dir")
    trainer.save_model("./output_dir")

    trainer.accelerator.end_training()
