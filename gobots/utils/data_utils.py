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
