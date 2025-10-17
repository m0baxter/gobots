import argparse
import torch
from huggingface_hub import login
from transformers import TrainingArguments, AutoTokenizer, AutoModel, AutoConfig
from gobots.model_factory import build_bagl_qmb
from gobots.trainers import MTPTrainer, LossAccumulator
from gobots.utils.model_utils import count_trainable_parameters
from gobots.utils.data_utils import prepare_pretraining_datasets


def generate_argparser():
    parser = argparse.ArgumentParser(prog="bagl_trainer")
    parser.add_argument("--mode", choices=["train", "eval"])

    return parser


if __name__ == "__main__":
    args = generate_argparser().parse_args()
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

    login(token="TOKEN GOES HERE")
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        config, tokenizer, model = build_bagl_qmb()

        print(f"main model parameters: {count_trainable_parameters(model.bagl_model)}")
        print(f"mtp model parameters: {count_trainable_parameters(model.mtp_model)}")
        print(f"total trainable parameters: {count_trainable_parameters(model)}")

        train_dataset = prepare_pretraining_datasets(
            tokenizer,
            context_length=2048 + 1 + config.mtp_config.num_nextn_predict_layers,
        )

    else:
        from gobots.models.bagl import BaGLConfig, BaGLModel

        AutoConfig.register("bagl", BaGLConfig)
        AutoModel.register(BaGLConfig, BaGLModel)

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        config = AutoConfig.from_pretrained("./bagl")
        model = AutoModel.from_pretrained("./bagl")
        model._tie_or_clone_weights(model.embedding_layer, model.lm_head)

    training_args = TrainingArguments(
        disable_tqdm=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=64,
        per_device_eval_batch_size=2,
        max_steps=10000,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=5000,
        logging_strategy="steps",
        logging_steps=1,
        output_dir="./output_dir",
        log_level="info",
        bf16=True,
        bf16_full_eval=False,
        optim="adamw_torch",
        learning_rate=2.0e-03,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=16,
        batch_eval_metrics=True,
        include_num_input_tokens_seen=True,
        max_grad_norm=1.0,
        weight_decay=0.01,
        run_name="bagl_00",
        report_to="trackio",
        torch_empty_cache_steps=1000,
        trackio_space_id=None,
    )
    trainer = MTPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        auxiliary_loss_weight=1e-4,
        z_loss_weight=0.001,
        mtp_weight=0.01,
        mtp_depth=config.mtp_config.num_nextn_predict_layers,
        ignore_index=-100,
        compute_metrics=compute_metrics,
    )

    if args.mode == "train":
        trainer.train()

    else:
        res = trainer.evaluate()
