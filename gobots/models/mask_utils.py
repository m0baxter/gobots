import torch
from functools import partial
from torch.nn.attention.flex_attention import create_block_mask, and_masks


def causal_mask_fn(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def document_masking_fn(b, h, q_idx, kv_idx, document_ids):
    return document_ids[b][q_idx] == document_ids[b][kv_idx]


@torch.compile()
def create_causal_mask(
    batch_size: int | None = None,
    num_heads: int | None = None,
    query_length: int | None = None,
    key_value_length: int | None = None,
):
    return create_block_mask(
        causal_mask_fn,
        B=batch_size,
        H=num_heads,
        Q_LEN=query_length,
        KV_LEN=key_value_length,
    )


@torch.compile()
def create_causal_document_mask(
    batch_size: int | None = None,
    num_heads: int | None = None,
    query_length: int | None = None,
    key_value_length: int | None = None,
    document_ids=None,
):
    mask_fn = and_masks(
        causal_mask_fn, partial(document_masking_fn, document_ids=document_ids)
    )

    return create_block_mask(
        mask_fn,
        B=batch_size,
        H=num_heads,
        Q_LEN=query_length,
        KV_LEN=key_value_length,
    )


def generate_block_mask(
    batch_size: int | None = None,
    num_heads: int | None = None,
    query_length: int | None = None,
    key_value_length: int | None = None,
    document_ids=None,
):
    if document_ids is None:
        return create_causal_mask(
            batch_size,
            num_heads,
            query_length,
            key_value_length,
        )

    return create_causal_document_mask(
        batch_size,
        num_heads,
        query_length,
        key_value_length,
        document_ids,
    )
