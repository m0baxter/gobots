import time
import torch
from datasets import load_dataset
from functools import partial
from transformers import TrainingArguments
from gobots.model_factory import build_bagl_qmb
from gobots.trainers import MTPTrainer, LossAccumulator
from gobots.utils.model_utils import count_trainable_parameters
from gobots.utils.data_utils import parse_examples

if __name__ == "__main__":
    from huggingface_hub import login

    loss_accumlator = LossAccumulator()

    def compute_metrics(pred, compute_result: bool = False):
        mtp_loss, auxiliary_loss, logits, mtp_logits = pred.predictions
        loss_accumlator.update(mtp_loss, auxiliary_loss, logits.shape[0])

        if compute_result:
            total_mtp_loss, total_auxiliary_loss = loss_accumlator.compute()

            return {"mtp_loss": total_mtp_loss, "auxiliary_loss": total_auxiliary_loss}

        return

    login(token="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")
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config, tokenizer, model = build_bagl_qmb()
    model = model.to(device)

    print(f"trainable parameters: {count_trainable_parameters(model)}")
    print(config)

    t_start = time.time()

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
    )

    document_count = len(dataset)
    train_size = int(document_count * 0.01)
    val_size = int(train_size * 0.1)
    sequence_length = 2048
    context_length = sequence_length + config.num_nextn_predict_layers + 1

    print(document_count, val_size, train_size)

    dataset = dataset.to_iterable_dataset(num_shards=8).shuffle(seed=1234)
    eval_dataset = dataset.take(val_size)
    train_dataset = dataset.skip(val_size).take(train_size)
    tokenize_fn = partial(
        parse_examples,
        tokenizer=tokenizer,
        eos_token="<|end_of_text|>",
        context_length=context_length,
        eos_token_id=128001,
    )
    eval_dataset = eval_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

    print(dataset)
    print("document count:", document_count)

    training_args = TrainingArguments(
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        max_steps=10000,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=100,
        output_dir="./output_dir",
        log_level="info",
        # fp16=True,
        bf16=True,
        bf16_full_eval=True,
        optim="adamw_torch",
        learning_rate=5.0e-04,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        batch_eval_metrics=True,
        include_num_input_tokens_seen=True,
        max_grad_norm=1.0,
        weight_decay=0.01,
        run_name="bagl",
        report_to="trackio"
    )

    trainer = MTPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        auxiliary_loss_weight=1e-4,
        mtp_weight=0.01,
        mtp_depth=1,
        ignore_index=-100,
        compute_metrics=compute_metrics,
    )
    trainer.train()
