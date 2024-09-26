import sys
sys.path.append("..")

import torch

from utils import longest_overlapping_substring


def get_tagged_spans(input, output, tokenizer):
    cidx = input["context_indices"][0]
    input_ids = input["input_ids"][0]
    context_ids = input_ids[cidx[0] : cidx[1]]
    spans_indices = torch.where(output["span_preds"][0] == 1)[0]

    spans_list = []
    cur_span_idx = [spans_indices[0]]
    for i in range(1, len(spans_indices)):
        if spans_indices[i] - spans_indices[i - 1] == 1:
            cur_span_idx.append(spans_indices[i])
        else:
            spans_list.append(tokenizer.decode(context_ids[torch.stack(cur_span_idx)]))
            cur_span_idx = [spans_indices[i]]

    return spans_list


def process_multispanqa_original_data(original_input, input, tokenizer, prompter, has_label=True):
    input_text = tokenizer.decode(input["input_ids"][0])
    question = prompter.get_question(input_text)
    context = prompter.get_context(input_text)

    if has_label:
        spans = []
        cur_span = []
        for tok, l in zip(original_input["context"], original_input["label"]):
            if l == "B" or l == "I":
                cur_span.append(tok)
            else:
                if cur_span:
                    spans.append(" ".join(cur_span))
                    cur_span = []

    processed_obj = {
        "id": original_input["id"],
        "question": question,
        "context": context,
        "spans": spans if has_label else None,
    }

    return processed_obj


def postprocess_multispanqa_prediction(gold_obj, generated_response):
    generated_spans = [s.strip() for s in generated_response.split(" ; ") if s.strip()]
    generated_spans = [
        longest_overlapping_substring(gold_obj["context"], s).strip()
        for s in generated_spans
    ]
    return {gold_obj["id"]: [s for s in generated_spans if s]}


def postprocess_squad_predictions(gold_obj, generated_response):
    prediction_text = longest_overlapping_substring(
        gold_obj["context"], generated_response
    ).strip()
    return {"id": gold_obj["id"], "prediction_text": prediction_text}


def postprocess_quoref_predictions(gold_obj, generated_response):
    generated_spans = [s.strip() for s in generated_response.split(" ; ") if s.strip()]
    generated_spans = [
        longest_overlapping_substring(gold_obj["context"], s).strip()
        for s in generated_spans
    ]
    return {gold_obj["id"]: [s for s in generated_spans if s]}
