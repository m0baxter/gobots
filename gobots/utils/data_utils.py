import boto3
import botocore

# import re
import smart_open
from botocore.exceptions import ClientError
from datasets import load_dataset, interleave_datasets
from functools import partial

drop_columns = []


def download_contents(
    blob_id,
    bucket_name="softwareheritage",
    s3=boto3.client(
        "s3",
        region_name="us-west-2",
        config=botocore.config.Config(signature_version=botocore.UNSIGNED),
    ),
):
    s3_url = f"s3://softwareheritage/content/{blob_id}"

    try:
        with smart_open.open(
            s3_url, "rb", compression=".gz", transport_params={"client": s3}
        ) as s3bucket:
            content = s3bucket.read().decode("utf-8", errors="ignore")

        return {"text": content, "download_success": True}

    except (ClientError, OSError, AttributeError) as e:
        if isinstance(e, OSError | AttributeError):
            return {"text": "", "download_success": False}

        elif e.response["Error"]["Code"] == "NoSuchKey":
            return {"text": "", "download_success": False}


def mark_documents(token_ids, eos_token_id):
    doc_id = []
    pos_id = []
    current_doc_id = 0
    current_pos = 0

    for tid in token_ids:
        doc_id.append(current_doc_id)
        pos_id.append(current_pos)

        if tid == eos_token_id:
            current_doc_id += 1
            current_pos = 0

        else:
            current_pos += 1

    return doc_id, pos_id


def format_cosmopedia(example, tokenizer):
    text = f"{example['prompt']}\n\nResponse:\n{example['text']}"

    return {"text": text}


def parse_examples(examples, tokenizer, eos_token, eos_token_id, context_length):
    text = eos_token.join(examples["text"])

    outputs = tokenizer(
        text,
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    input_ids = []
    document_ids = []
    input_pos = []

    for sequence_lenght, token_ids in zip(outputs["length"], outputs["input_ids"]):
        if sequence_lenght == context_length:
            doc_id, pos_id = mark_documents(token_ids, eos_token_id=eos_token_id)

            input_ids.append(token_ids)
            document_ids.append(doc_id)
            input_pos.append(pos_id)

    return {
        "input_ids": input_ids,
        "document_ids": document_ids,
        "input_pos": input_pos,
        "labels": input_ids,
    }


def prepare_pretraining_datasets(
    tokenizer,
    context_length: int,
    num_shards: int = 8,
    seed: int = 1234,
):
    dataset_python_edu = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "python-edu",
        split="train",
        num_proc=num_shards,
    )
    dataset_python_edu = dataset_python_edu.map(
        download_contents,
        input_columns="blob_id",
        num_proc=32,
    ).to_iterable_dataset(num_shards=num_shards)
    dataset_python_edu = dataset_python_edu.filter(lambda x: x["download_success"])

    dataset_fineweb_edu = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup",
        split="train",
        num_proc=4,
    ).to_iterable_dataset(num_shards=num_shards)
    # dataset_cosmopedia_v2 = load_dataset(
    #     "HuggingFaceTB/smollm-corpus",
    #     "cosmopedia-v2",
    #     split="train",
    #     num_proc=num_shards,
    # ).to_iterable_dataset(num_shards=num_shards)
    dataset_finepdfs = load_dataset(
        "HuggingFaceFW/finepdfs",
        revision="v1.6.0",
        name="eng_Latn",
        split="train",
        num_proc=num_shards,
    ).to_iterable_dataset(num_shards=num_shards)
    # dataset_dclm_edu = load_dataset(
    #     "HuggingFaceTB/dclm-edu",
    #     split="train",
    #     num_proc=num_shards,
    # ).to_iterable_dataset(num_shards=num_shards)
    dataset_finemath1 = load_dataset(
        "HuggingFaceTB/finemath",
        "finemath-3plus",
        split="train",
        num_proc=num_shards,
    ).to_iterable_dataset(num_shards=num_shards)
    dataset_finemath2 = load_dataset(
        "HuggingFaceTB/finemath",
        "infiwebmath-3plus",
        split="train",
        num_proc=num_shards,
    ).to_iterable_dataset(num_shards=num_shards)

    dataset_fineweb_edu = dataset_fineweb_edu.select_columns(column_names="text")
    dataset_finepdfs = dataset_finepdfs.select_columns(column_names="text")
    dataset_python_edu = dataset_python_edu.select_columns(column_names="text")
    dataset_finemath1 = dataset_finemath1.select_columns(column_names="text")
    dataset_finemath2 = dataset_finemath2.select_columns(column_names="text")
    # dataset_dclm_edu = dataset_dclm_baseline.select_columns(column_names="text")
    # dataset_cosmopedia_v2 = dataset_cosmopedia_v2.map(
    #     partial(format_cosmopedia, tokenizer=tokenizer),
    #     remove_columns=[
    #         "prompt",
    #         "token_length",
    #         "audience",
    #         "format",
    #         "seed_data",
    #         "metadata",
    #         "language",
    #     ],
    # )

    tokenize_fn = partial(
        parse_examples,
        tokenizer=tokenizer,
        eos_token="<|end_of_text|>",
        context_length=context_length,
        eos_token_id=128001,
    )

    merged_dataset = interleave_datasets(
        [
            dataset_fineweb_edu,
            # dataset_dclm_edu,
            # dataset_cosmopedia_v2,
            dataset_finepdfs,
            dataset_python_edu,
            dataset_finemath1,
            dataset_finemath2,
        ],
        # probabilities=[0.30, 0.30, 0.125, 0.125, 0.12, 0.015, 0.015],
        # probabilities=[0.60, 0.125, 0.125, 0.12, 0.015, 0.015],
        probabilities=[0.60, 0.25, 0.12, 0.015, 0.015],
        stopping_strategy="first_exhausted",
        seed=seed,
    )
    merged_dataset = merged_dataset.map(
        tokenize_fn, batched=True, remove_columns=["text"]
    ).with_format("torch")

    return merged_dataset
